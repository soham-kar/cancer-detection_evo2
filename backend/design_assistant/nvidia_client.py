"""
NVIDIA Nemotron API Client
===========================
Thin wrapper around the NVIDIA Nemotron-3 Ultra 550B API.
Uses OpenAI-compatible REST endpoint at integrate.api.nvidia.com.

Verified working: 2026-06-28
- Model: nvidia/nemotron-3-ultra-550b-a55b
- Returns reasoning_content + content
- ~24s latency for 600-token responses
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# Fallback model (Groq Llama 3.3 70B) for lower latency / cost-sensitive paths
FALLBACK_MODEL = "llama-3.3-70b-versatile"


def _load_api_key() -> Optional[str]:
    """Load NVIDIA_API_KEY from environment or frontend .env.local file."""
    key = os.getenv("NVIDIA_API_KEY")
    if key:
        return key

    # Try reading from frontend .env.local (development convenience)
    env_paths = [
        Path(__file__).parent.parent.parent / "frontend" / ".env.local",
        Path("frontend/.env.local"),
    ]
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("NVIDIA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class NemotronClient:
    """Minimal Nemotron-3 Ultra 550B client for the design assistant."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _load_api_key()
        if not self.api_key:
            logger.warning(
                "NVIDIA_API_KEY not found. Nemotron calls will fail. "
                "Set the key in frontend/.env.local or as an environment variable."
            )

    # ------------------------------------------------------------------
    # Low-level call
    # ------------------------------------------------------------------

    def _call(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        timeout: int = 180,
    ) -> Dict[str, Any]:
        """Raw REST call to the Nemotron chat completions endpoint."""
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set")

        url = f"{NVIDIA_BASE_URL}/chat/completions"
        payload = {
            "model": NVIDIA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start

        if resp.status_code != 200:
            logger.error(
                "Nemotron API error status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise RuntimeError(
                f"Nemotron API returned {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        logger.info(
            "Nemotron call completed in %.1fs, tokens=%s",
            elapsed,
            data.get("usage", {}).get("total_tokens", "?"),
        )
        return data

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def reason(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Dict[str, Optional[str]]:
        """
        Send a reasoning prompt and return both reasoning_content and content.

        Returns:
            dict with keys:
                - reasoning: str or None (model's internal thinking)
                - content: str or None (model's final answer)
                - usage: dict (token counts)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        data = self._call(messages, temperature=temperature, max_tokens=max_tokens)

        choice = data["choices"][0]["message"]
        return {
            "reasoning": choice.get("reasoning_content"),
            "content": choice.get("content"),
            "usage": data.get("usage", {}),
        }

    def answer(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a prompt and return only the content string.
        Convenience wrapper for simple Q&A.
        """
        result = self.reason(
            system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens
        )
        return result["content"] or ""

    def reason_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Send a prompt and attempt to parse the response as JSON.
        Falls back to raw content if JSON parsing fails.
        """
        result = self.reason(
            system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens
        )
        content = result["content"] or ""

        # Try to extract JSON from the response
        import re

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return {
                    "parsed": parsed,
                    "raw": content,
                    "reasoning": result["reasoning"],
                    "usage": result["usage"],
                }
            except json.JSONDecodeError:
                pass

        return {
            "parsed": None,
            "raw": content,
            "reasoning": result["reasoning"],
            "usage": result["usage"],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: Optional[NemotronClient] = None


def get_client() -> NemotronClient:
    """Return a shared NemotronClient instance."""
    global _client
    if _client is None:
        _client = NemotronClient()
    return _client
