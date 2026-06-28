"""
Design Assistant — Modal Deployment
====================================
Deploys the HelixDesign therapeutic reasoning assistant as a Modal service
with FastAPI endpoints. Runs on CPU (no GPU needed — just calls Nemotron API).

Usage:
    modal deploy backend/design_assistant/modal_deploy.py

Endpoints:
    POST /analyze  — One-shot therapeutic strategy analysis
    POST /chat     — SSE streaming conversational Q&A (Phase 12)

Secrets (set in Modal dashboard):
    nvidia-key  — NVIDIA_API_KEY for Nemotron-3 Ultra 550B
    groq-key    — GROQ_API_KEY for Llama 3.3 70B fallback
"""

import modal
from pathlib import Path

# =============================================================================
# Image: Python 3.12 + design_assistant + dependencies
# =============================================================================

design_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.115.6",
        "requests>=2.31.0",
        "groq>=0.4.0",
    )
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/app/design_assistant",
    )
)

app = modal.App(
    "design-assistant",
    image=design_image,
    secrets=[
        modal.Secret.from_name("nvidia-key"),
        # Optional: only needed if Groq fallback is used
        # modal.Secret.from_name("groq-key"),
    ],
)

# =============================================================================
# Service class
# =============================================================================


@app.cls(
    cpu=2.0,
    memory=2048,
    scaledown_window=600,  # Allow up to 10 min for long LLM calls
)
@modal.concurrent(max_inputs=5)
class DesignAssistantService:
    """
    CPU-only Modal service for therapeutic design reasoning.

    Each container handles up to 5 concurrent requests.
    Containers scale to zero when idle (after 10 min scaledown window).
    """

    @modal.enter()
    def setup(self):
        """Load the design assistant on container start."""
        import sys
        sys.path.insert(0, "/app")

        from design_assistant.agent import DesignAssistant
        from design_assistant.nvidia_client import get_client

        self.assistant = DesignAssistant(use_llm=True)
        self.nvidia = get_client()
        print("DesignAssistantService ready.")

    # ------------------------------------------------------------------
    # POST /analyze — One-shot therapeutic strategy
    # ------------------------------------------------------------------

    @modal.fastapi_endpoint(method="POST")
    async def analyze(self, report: dict) -> dict:
        """
        Analyze a variant report and return therapeutic recommendations.

        Request body: Full variant report dict (already snake_case normalized).
        Response: { strategy_class, confidence, reasoning_summary, evidence_bullets, ... }
        """
        try:
            result = self.assistant.analyze(report)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "strategy_class": "observe_and_reassess",
                "confidence": "low",
                "reasoning_summary": f"Design assistant encountered an error: {e}",
                "evidence_bullets": [],
                "recommended_next_steps": [],
                "limitations": ["Analysis failed due to an internal error."],
                "generated_at": "",
                "evidence_summary": "Error during analysis",
                "rule_based": True,
            }

    # ------------------------------------------------------------------
    # POST /chat — SSE streaming conversational Q&A (Phase 12)
    # ------------------------------------------------------------------

    @modal.fastapi_endpoint(method="POST")
    async def chat(self, request: dict):
        """
        Streaming chat endpoint. Returns SSE (text/event-stream).

        Request body:
        {
            "messages": [{"role": "user", "content": "Why is this a VUS?"}],
            "variantContext": { "geneSymbol": "BRCA1", "position": 43094169, "report": {...} }
        }

        Response: SSE stream with events: thinking, tool_call, answer, done, error
        """
        from fastapi.responses import StreamingResponse
        import json
        import asyncio

        messages = request.get("messages", [])
        variant_context = request.get("variantContext", {})

        async def event_stream():
            try:
                # Build system prompt with full report context
                report = variant_context.get("report", {})
                system_prompt = self._build_chat_system_prompt(report)

                # Call Nemotron with streaming
                full_messages = [
                    {"role": "system", "content": system_prompt},
                    *messages,
                ]

                # Stream thinking + answer
                async for event in self._stream_nemotron(full_messages):
                    yield event

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @modal.fastapi_endpoint(method="GET")
    async def health(self) -> dict:
        return {"status": "healthy", "service": "design-assistant"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chat_system_prompt(self, report: dict) -> str:
        """Build the system prompt with full variant report context."""
        import json as _json

        gene = report.get("gene_symbol", "Unknown")
        position = report.get("position", "Unknown")
        prediction = report.get("prediction", "Unknown")
        delta = report.get("delta_score", "N/A")

        return f"""You are HelixDesign, an expert genomics and therapeutic design assistant
powered by Nemotron-3 Ultra 550B. You help researchers understand genetic variants
and their therapeutic implications.

## Current Variant
- Gene: {gene}
- Position: {position}
- Evo2 Prediction: {prediction}
- Evo2 Delta Score: {delta}

## Full Report
{_json.dumps(report, indent=2, default=str)}

## Instructions
- Show your reasoning before answering (chain-of-thought)
- Cite specific evidence sources for every claim
- Be honest about uncertainty — don't fabricate evidence
- Suggest actionable next steps when appropriate
- Keep answers clear and evidence-based
"""

    async def _stream_nemotron(self, messages: list):
        """Stream Nemotron response as SSE events."""
        import json as _json
        import requests

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        api_key = self.nvidia.api_key

        payload = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2048,
            "top_p": 0.95,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=300
        )

        thinking_done = False
        message_id = "msg_" + str(hash(str(messages[-1])))

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                data = _json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {})

                # Check for reasoning_content (chain-of-thought)
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield f"event: thinking\ndata: {_json.dumps({'token': reasoning, 'messageId': message_id})}\n\n"
                    continue

                # Check for content (final answer)
                content = delta.get("content")
                if content:
                    if not thinking_done:
                        thinking_done = True
                        yield f"event: thinking_done\ndata: {_json.dumps({'messageId': message_id})}\n\n"
                    yield f"event: answer\ndata: {_json.dumps({'token': content, 'messageId': message_id})}\n\n"

            except _json.JSONDecodeError:
                continue

        yield f"event: done\ndata: {_json.dumps({'messageId': message_id})}\n\n"


# =============================================================================
# Local dev entry point (for testing without Modal)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Design Assistant API locally...")
    print("Endpoints: POST /analyze, POST /chat, GET /health")
    uvicorn.run(
        "modal_deploy:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
