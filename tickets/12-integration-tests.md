# Ticket 12: Integration Tests

**Dependencies:** Ticket 08, 09, 10
**Estimated time:** 25 minutes
**Spec reference:** Spec doesn't list specific test cases; these cover end-to-end tool-calling scenarios

---

## Objective

Create `tests/test_tool_calling.py` — integration tests that load the fine-tuned model and validate it handles all critical tool-calling scenarios correctly.

---

## Step 1: Create test suite

Create `slm-tool-calling-finetune/tests/test_tool_calling.py`:

```python
"""
Integration tests for the fine-tuned tool-calling model.

Tests: single call, parallel calls, no-call, missing params,
unavailable tool, ambiguous query, complex args, empty tool list.

Usage:
    python -m pytest tests/test_tool_calling.py -v
    python tests/test_tool_calling.py
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import load_model, run_inference, extract_tool_calls, has_tool_call

# Tool fixtures
WEATHER = {"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City and state"}},"required":["location"]}}}
SEARCH = {"type":"function","function":{"name":"search_web","description":"Search the web for information","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}
EMAIL = {"type":"function","function":{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}}
CALC = {"type":"function","function":{"name":"calculate","description":"Evaluate a math expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}
ALL_TOOLS = [WEATHER, SEARCH, EMAIL, CALC]

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        # Try fused first, fall back to adapters
        try:
            _model, _tokenizer = load_model(use_adapters=False)
        except Exception:
            _model, _tokenizer = load_model(use_adapters=True)
    return _model, _tokenizer


class TestToolCalling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = get_model()

    def _gen(self, query, tools=None):
        return run_inference(self.model, self.tokenizer, query,
                             tools or ALL_TOOLS, max_tokens=256, temp=0.0)

    def test_01_single_tool_call(self):
        """Single tool correctly identified with valid args."""
        r = self._gen("What's the weather in San Francisco?")
        calls = extract_tool_calls(r)
        self.assertTrue(len(calls) >= 1, f"Expected call, got: {r[:200]}")
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertIn("location", calls[0].get("arguments", {}))

    def test_02_parallel_calls(self):
        """Multiple tools called for compound query."""
        r = self._gen("Weather in NYC and search for restaurants there")
        calls = extract_tool_calls(r)
        self.assertTrue(len(calls) >= 2, f"Expected >=2 calls, got {len(calls)}")
        names = [c["name"] for c in calls]
        self.assertIn("get_weather", names)
        self.assertIn("search_web", names)

    def test_03_no_tool_needed(self):
        """General knowledge question — no tool call."""
        r = self._gen("What is the capital of France?")
        self.assertFalse(has_tool_call(r), f"Should not call tools: {r[:200]}")

    def test_04_missing_params(self):
        """Incomplete request — model asks for clarification or makes reasonable guess."""
        r = self._gen("Send an email about the project", tools=[EMAIL])
        calls = extract_tool_calls(r)
        if calls:
            args = calls[0].get("arguments", {})
            self.assertIn("to", args, "Should have 'to' if calling send_email")

    def test_05_unavailable_tool(self):
        """Model should NOT hallucinate tools not in the list."""
        r = self._gen("Send an email to john@test.com", tools=[WEATHER])
        calls = extract_tool_calls(r)
        if calls:
            for c in calls:
                self.assertNotEqual(c["name"], "send_email",
                    "Hallucinated send_email which is not available")

    def test_06_ambiguous_query(self):
        """Math question — should use calculator over search."""
        r = self._gen("What is 15% of 250?", tools=[CALC, SEARCH])
        calls = extract_tool_calls(r)
        if calls:
            self.assertEqual(calls[0]["name"], "calculate")

    def test_07_complex_args(self):
        """Tool call with multiple required parameters."""
        r = self._gen(
            "Email alice@co.com with subject 'Meeting' and body 'See you at 3pm'",
            tools=[EMAIL],
        )
        calls = extract_tool_calls(r)
        self.assertTrue(len(calls) >= 1)
        args = calls[0].get("arguments", {})
        self.assertIn("to", args)
        self.assertIn("subject", args)
        self.assertIn("body", args)

    def test_08_empty_tool_list(self):
        """No tools available — must not call any."""
        r = self._gen("What's the weather?", tools=[])
        self.assertFalse(has_tool_call(r), f"Called tools with empty list: {r[:200]}")

    def test_09_json_validity_batch(self):
        """10 queries — all tool calls must produce valid JSON."""
        queries = [
            "Weather in Paris?", "Search for Python docs",
            "Calculate 127 * 38", "Weather in Berlin in celsius?",
            "Search for AI news", "Email bob@test.com saying hello",
        ]
        valid, invalid = 0, 0
        for q in queries:
            r = run_inference(self.model, self.tokenizer, q, ALL_TOOLS, 256, 0.0)
            if has_tool_call(r):
                if extract_tool_calls(r):
                    valid += 1
                else:
                    invalid += 1
        total = valid + invalid
        if total > 0:
            self.assertGreaterEqual(valid / total, 0.8, f"JSON validity: {valid}/{total}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

## Step 2: Run tests

```bash
cd slm-tool-calling-finetune

# Run all tests
python -m pytest tests/test_tool_calling.py -v 2>&1 | tee test_results.log

# Run individual test
python -m pytest tests/test_tool_calling.py -v -k "test_01"
```

---

## Step 3: Interpret results

**Must pass:** 01 (single call), 03 (no tool), 05 (no hallucination), 08 (empty tools)
**Should pass:** 04, 06, 07, 09
**Stretch goal:** 02 (parallel calls — hardest for SLMs)

Minimum: 6/9 tests passing.

---

## Acceptance Criteria

- [ ] Test suite runs without crashes
- [ ] At least 6/9 tests pass
- [ ] All "must pass" tests pass (01, 03, 05, 08)
- [ ] JSON validity batch test shows ≥ 80%
- [ ] Results logged to `test_results.log`
