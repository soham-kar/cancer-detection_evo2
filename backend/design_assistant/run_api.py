"""
Design Assistant API Entry Point
================================
Reads a variant report from stdin (JSON), runs the design assistant,
and prints the result to stdout (JSON).

Called by the Next.js API route: POST /api/design-assistant
"""

import sys
import json
import os

# Ensure the backend directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_assistant.agent import DesignAssistant


def main():
    try:
        # Read report from stdin
        raw = sys.stdin.read()
        report = json.loads(raw)

        # Run the design assistant
        assistant = DesignAssistant(use_llm=True)
        result = assistant.analyze(report)

        # Output result as JSON to stdout
        print(json.dumps(result))

    except Exception as e:
        # Print error as JSON so the frontend can parse it
        error_result = {
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
        print(json.dumps(error_result))
        sys.exit(0)  # Exit cleanly so the Node process doesn't crash


if __name__ == "__main__":
    main()
