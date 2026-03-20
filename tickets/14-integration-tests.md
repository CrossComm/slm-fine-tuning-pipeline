# Ticket 14: Integration Test Suite

**Dependencies:** Ticket 10, Ticket 12, Ticket 13
**Estimated time:** 30 minutes
**Spec reference:** Section 11.3 (Test Cases to Validate)

---

## Objective

Create `tests/test_tool_calling.py` — a comprehensive test suite that validates the fine-tuned model against all 9 test scenarios from the engineering spec. These are end-to-end integration tests that load the model and run actual inference.

---

## Step 1: Create the test suite

Create `slm-tool-calling-finetune/tests/test_tool_calling.py`:

```python
"""
Integration test suite for the fine-tuned tool-calling model.

Tests all 9 scenarios from the engineering spec Section 11.3.
Each test loads the model, runs inference with specific tools and queries,
and validates the output meets expectations.

Usage:
    python -m pytest tests/test_tool_calling.py -v
    python -m pytest tests/test_tool_calling.py -v -k "test_single"  # Run one test
    python tests/test_tool_calling.py                                 # Run without pytest

Prerequisites:
    - Training complete (Ticket 09)
    - LoRA adapters at outputs/lora_adapters/ OR merged model at outputs/merged_model/
"""

import json
import os
import sys
import unittest

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    load_finetuned_model,
    generate_response,
    extract_tool_calls,
    has_tool_call,
)

# =========================================================================
# TEST FIXTURES: Tool definitions used across tests
# =========================================================================

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information on a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}

EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content"}
            },
            "required": ["to", "subject", "body"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '2 + 2'"
                }
            },
            "required": ["expression"]
        }
    }
}

ALL_TOOLS = [WEATHER_TOOL, SEARCH_TOOL, EMAIL_TOOL, CALCULATOR_TOOL]


# =========================================================================
# SHARED MODEL LOADING (loaded once for all tests)
# =========================================================================

_model = None
_tokenizer = None


def get_model():
    """Lazy-load the model once for all tests."""
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_finetuned_model()
    return _model, _tokenizer


# =========================================================================
# TEST CASES
# =========================================================================

class TestToolCalling(unittest.TestCase):
    """Integration tests for all 9 spec scenarios."""

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        cls.model, cls.tokenizer = get_model()

    def _generate(self, query, tools=None, max_tokens=256):
        """Helper to generate response."""
        if tools is None:
            tools = ALL_TOOLS
        return generate_response(
            self.model, self.tokenizer,
            query, tools,
            max_new_tokens=max_tokens,
            temperature=0.0,
        )

    # -----------------------------------------------------------------
    # Spec Test 1: Single tool call
    # -----------------------------------------------------------------
    def test_01_single_tool_call(self):
        """Model correctly identifies a single tool and provides valid args."""
        response = self._generate("What's the weather in San Francisco?")
        calls = extract_tool_calls(response)

        self.assertTrue(len(calls) >= 1, f"Expected tool call, got: {response[:200]}")
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertIn("location", calls[0].get("arguments", {}))

        # Verify the JSON is valid
        for call in calls:
            self.assertIsInstance(call, dict)
            self.assertIn("name", call)
            self.assertIn("arguments", call)

    # -----------------------------------------------------------------
    # Spec Test 2: Multiple tool calls (parallel)
    # -----------------------------------------------------------------
    def test_02_parallel_tool_calls(self):
        """Model emits multiple tool calls when query requires it."""
        response = self._generate(
            "What's the weather in New York and also search for best restaurants there?"
        )
        calls = extract_tool_calls(response)

        self.assertTrue(len(calls) >= 2,
            f"Expected >= 2 tool calls, got {len(calls)}: {response[:200]}")

        call_names = [c["name"] for c in calls]
        self.assertIn("get_weather", call_names)
        self.assertIn("search_web", call_names)

    # -----------------------------------------------------------------
    # Spec Test 3: No tool needed
    # -----------------------------------------------------------------
    def test_03_no_tool_needed(self):
        """Model responds naturally without calling tools for general questions."""
        response = self._generate("What is the capital of France?")

        self.assertFalse(has_tool_call(response),
            f"Expected no tool call for general knowledge question, got: {response[:200]}")
        self.assertTrue(len(response) > 10,
            "Response should be a meaningful answer")

    # -----------------------------------------------------------------
    # Spec Test 4: Missing required parameter
    # -----------------------------------------------------------------
    def test_04_missing_required_param(self):
        """Model asks for clarification when required info is missing."""
        response = self._generate(
            "Send an email about the project update",
            tools=[EMAIL_TOOL],
        )

        # The model should either:
        # a) Ask for the recipient (no tool call), OR
        # b) Make a tool call but with a reasonable guess
        calls = extract_tool_calls(response)

        if calls:
            # If it does call, it should have valid required params
            for call in calls:
                args = call.get("arguments", {})
                self.assertIn("to", args, "Missing 'to' in email args")
                self.assertIn("subject", args, "Missing 'subject' in email args")
                self.assertIn("body", args, "Missing 'body' in email args")
        # If no call, that's also acceptable (asking for clarification)

    # -----------------------------------------------------------------
    # Spec Test 5: Tool not in available tools
    # -----------------------------------------------------------------
    def test_05_tool_not_available(self):
        """Model does NOT hallucinate a tool that wasn't defined."""
        # Only provide weather tool, ask for email
        response = self._generate(
            "Send an email to john@example.com about the meeting",
            tools=[WEATHER_TOOL],
        )
        calls = extract_tool_calls(response)

        if calls:
            # If any calls made, they should only be for available tools
            for call in calls:
                self.assertEqual(call["name"], "get_weather",
                    f"Model hallucinated tool: {call['name']}")
        # Ideally: no tool call at all, natural language explaining it can't send email

    # -----------------------------------------------------------------
    # Spec Test 6: Ambiguous query matching multiple tools
    # -----------------------------------------------------------------
    def test_06_ambiguous_query(self):
        """Model selects the most appropriate tool for an ambiguous query."""
        response = self._generate(
            "What is 15% of 250?",
            tools=[CALCULATOR_TOOL, SEARCH_TOOL],
        )
        calls = extract_tool_calls(response)

        if calls:
            # Should prefer calculator over search for math
            self.assertEqual(calls[0]["name"], "calculate",
                f"Expected calculator for math query, got: {calls[0]['name']}")

    # -----------------------------------------------------------------
    # Spec Test 7: Complex nested args
    # -----------------------------------------------------------------
    def test_07_complex_nested_args(self):
        """Model produces valid JSON for complex arguments."""
        complex_tool = {
            "type": "function",
            "function": {
                "name": "create_event",
                "description": "Create a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title"},
                        "start_time": {"type": "string", "description": "ISO 8601 start time"},
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of attendee emails"
                        },
                        "location": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "address": {"type": "string"}
                            },
                            "description": "Event location"
                        }
                    },
                    "required": ["title", "start_time"]
                }
            }
        }

        response = self._generate(
            "Create a meeting called 'Sprint Review' tomorrow at 2pm "
            "with alice@co.com and bob@co.com at Conference Room B, 123 Main St",
            tools=[complex_tool],
        )
        calls = extract_tool_calls(response)

        self.assertTrue(len(calls) >= 1, f"Expected tool call, got: {response[:200]}")
        self.assertEqual(calls[0]["name"], "create_event")

        args = calls[0].get("arguments", {})
        self.assertIn("title", args)
        # JSON must be valid (extract_tool_calls already validated this)

    # -----------------------------------------------------------------
    # Spec Test 8: Multi-turn with tool results
    # -----------------------------------------------------------------
    def test_08_multi_turn_format(self):
        """Model output follows expected format for multi-turn tool use."""
        # This tests that the tool call format is parseable
        response = self._generate("What's the weather like in Tokyo?", tools=[WEATHER_TOOL])
        calls = extract_tool_calls(response)

        self.assertTrue(len(calls) >= 1, "Expected weather tool call")
        self.assertEqual(calls[0]["name"], "get_weather")

        # Verify the call has the expected structure for multi-turn integration
        call = calls[0]
        self.assertIsInstance(call["arguments"], dict)
        # The response should be parseable so a tool result can be injected

    # -----------------------------------------------------------------
    # Spec Test 9: Empty tool list
    # -----------------------------------------------------------------
    def test_09_empty_tool_list(self):
        """Model responds naturally when no tools are available."""
        response = self._generate(
            "What's the weather in London?",
            tools=[],
        )

        self.assertFalse(has_tool_call(response),
            f"Should not call tools when none are available: {response[:200]}")
        self.assertTrue(len(response) > 5, "Should give some response")


# =========================================================================
# JSON VALIDITY STRESS TEST
# =========================================================================

class TestJSONValidity(unittest.TestCase):
    """Test that all tool call outputs are valid JSON."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = get_model()

    def test_json_validity_batch(self):
        """Run 20 diverse queries and verify all tool calls produce valid JSON."""
        queries = [
            "What's the weather in Paris?",
            "Search for Python tutorials",
            "Email john@test.com about the report with a summary",
            "Calculate 127 * 38",
            "What's the temperature in Berlin in celsius?",
            "Look up recent news about AI",
            "What's the weather in Tokyo and search for sushi restaurants",
            "Send a meeting reminder to team@company.com",
            "Calculate the square root of 144",
            "Search for flights to London",
        ]

        valid = 0
        invalid = 0

        for query in queries:
            response = generate_response(
                self.model, self.tokenizer,
                query, ALL_TOOLS,
                max_new_tokens=256,
                temperature=0.0,
            )

            if has_tool_call(response):
                calls = extract_tool_calls(response)
                if calls:
                    valid += 1
                else:
                    invalid += 1
                    print(f"  INVALID JSON for: {query}")
                    print(f"  Response: {response[:200]}")

        total = valid + invalid
        if total > 0:
            rate = valid / total
            print(f"\nJSON validity: {valid}/{total} ({rate:.0%})")
            self.assertGreaterEqual(rate, 0.9,
                f"JSON validity too low: {rate:.0%} (target: ≥90%)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

## Step 2: Install pytest (if not already installed)

```bash
pip install pytest
```

---

## Step 3: Run the test suite

```bash
cd slm-tool-calling-finetune

# Run all tests with verbose output
python -m pytest tests/test_tool_calling.py -v 2>&1 | tee test_results.log

# Or run specific tests
python -m pytest tests/test_tool_calling.py -v -k "test_01"   # Single tool call
python -m pytest tests/test_tool_calling.py -v -k "test_03"   # No tool needed
python -m pytest tests/test_tool_calling.py -v -k "test_05"   # Tool not available

# Or run without pytest
python tests/test_tool_calling.py
```

---

## Step 4: Interpret results

### Expected pass rate for a well-trained model:

| Test | Expected | Acceptable |
|------|----------|-----------|
| test_01_single_tool_call | PASS | Must pass |
| test_02_parallel_tool_calls | PASS | May fail — hardest test |
| test_03_no_tool_needed | PASS | Must pass |
| test_04_missing_required_param | PASS | Either behavior OK |
| test_05_tool_not_available | PASS | Must pass |
| test_06_ambiguous_query | PASS | May fail |
| test_07_complex_nested_args | PASS | May fail on structure |
| test_08_multi_turn_format | PASS | Must pass |
| test_09_empty_tool_list | PASS | Must pass |
| test_json_validity_batch | PASS | ≥90% rate |

**Minimum passing:** Tests 01, 03, 05, 08, 09 MUST pass. Others are stretch goals.

---

## Step 5: Document results

Record the test results:

```
Test Results:
  Passed: ___ / 10
  Failed: ___
  Specific failures:
    - test_XX: [description of failure]

Action items for failures:
  - [list any needed improvements]
```

---

## Acceptance Criteria

- [ ] `tests/test_tool_calling.py` runs without crashes
- [ ] At least 7/10 tests pass (including all "must pass" tests)
- [ ] JSON validity stress test shows ≥ 90% valid rate
- [ ] Results are logged to `test_results.log`
- [ ] Any failures have documented explanations

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| All tests fail | Model may not have trained well. Run `python inference.py` interactively to check. |
| test_02 (parallel) fails | Parallel calling is hard for SLMs. Consider this a stretch goal. |
| test_03 (no tool) fails | Insufficient negative examples in training. Re-run Ticket 06 with more examples. |
| test_05 (hallucination) fails | Model is hallucinating tools. Add more diverse negative examples. |
| Tests are very slow | Each test runs inference (~2-5 sec). 10 tests ≈ 30-50 sec. If slower, check GPU. |
| Import errors | Run from the `slm-tool-calling-finetune/` directory with the venv activated. |
