# Ticket 10: Inference Script & Tool Call Parser

**Dependencies:** Ticket 02, Ticket 03
**Estimated time:** 20 minutes
**Spec reference:** Sections 11.1 (Inference Pattern), 11.2 (Parsing Tool Calls)

---

## Objective

Create `inference.py` — a script that loads the fine-tuned model (from LoRA adapters), accepts user queries with tool definitions, generates responses, and parses tool calls from the output. This script is used for both interactive testing and by the evaluation script.

---

## Step 1: Create `inference.py`

Create `slm-tool-calling-finetune/inference.py`:

```python
"""
Inference script for the fine-tuned Qwen 3.5-4B tool-calling model.

Loads the LoRA-adapted model and provides:
  1. Tool call parsing from model output
  2. Single-turn tool-calling inference
  3. Multi-turn tool-calling with tool result injection
  4. Interactive REPL for testing

Usage:
    python inference.py                    # Interactive mode
    python inference.py --query "..."      # Single query mode

Prerequisites:
    Training must be complete (Ticket 09) with adapters in outputs/lora_adapters/
"""

import argparse
import json
import re
import torch
from unsloth import FastLanguageModel
from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT,
    LORA_OUTPUT_DIR, SYSTEM_PROMPT,
)


# =========================================================================
# TOOL CALL PARSER
# =========================================================================

def extract_tool_calls(response: str) -> list[dict]:
    """
    Extract tool calls from Qwen 3.5 native format.

    Looks for content between <|function_calls|> and <|/function_calls|> tokens.
    The content should be a JSON array of {"name": ..., "arguments": ...} objects.

    Returns:
        list of dicts, each with "name" and "arguments" keys.
        Returns empty list if no tool calls found or parsing fails.
    """
    pattern = r'<\|function_calls\|>\s*(.*?)\s*<\|/function_calls\|>'
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        return []

    try:
        calls = json.loads(match.group(1))
        if isinstance(calls, dict):
            calls = [calls]  # Single call wrapped in dict instead of array
        if isinstance(calls, list):
            return calls
        return []
    except json.JSONDecodeError:
        return []


def has_tool_call(response: str) -> bool:
    """Check if a response contains a tool call."""
    return "<|function_calls|>" in response


# =========================================================================
# MODEL LOADING
# =========================================================================

def load_finetuned_model(adapter_path: str = None):
    """
    Load the fine-tuned model from LoRA adapters.

    Args:
        adapter_path: Path to LoRA adapters. Defaults to config.LORA_OUTPUT_DIR.

    Returns:
        tuple: (model, tokenizer) ready for inference
    """
    if adapter_path is None:
        adapter_path = LORA_OUTPUT_DIR

    print(f"Loading fine-tuned model from {adapter_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Set to inference mode
    FastLanguageModel.for_inference(model)

    # Fix pad_token if needed
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = "<|endoftext|>"

    print(f"✅ Model loaded and set to inference mode")
    return model, tokenizer


# =========================================================================
# INFERENCE
# =========================================================================

def generate_response(
    model,
    tokenizer,
    query: str,
    tools: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """
    Generate a response for a user query with available tools.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        query: User's natural language query
        tools: List of tool definitions (OpenAI schema format)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (low = more deterministic)

    Returns:
        str: The model's response (may contain tool calls)
    """
    # Build tool definitions string
    tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
    system_content = f"{SYSTEM_PROMPT}\n\n{tools_json}"

    # Build messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    # Apply chat template with generation prompt
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (not the input)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    # Clean up trailing special tokens
    for token in [tokenizer.eos_token, "<|im_end|>"]:
        if token and response.endswith(token):
            response = response[:-len(token)].strip()

    return response


def run_tool_calling_loop(
    model,
    tokenizer,
    query: str,
    tools: list[dict],
    tool_executor=None,
) -> dict:
    """
    Run a complete tool-calling loop:
    1. Generate response
    2. If tool call detected, parse it
    3. If executor provided, run the tool and feed result back
    4. Generate final response

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        query: User query
        tools: Available tools
        tool_executor: Optional callable(name, args) -> result

    Returns:
        dict with keys: response, tool_calls, tool_results, final_response
    """
    # Step 1: Generate initial response
    response = generate_response(model, tokenizer, query, tools)

    result = {
        "response": response,
        "tool_calls": [],
        "tool_results": [],
        "final_response": response,
    }

    # Step 2: Check for tool calls
    tool_calls = extract_tool_calls(response)
    result["tool_calls"] = tool_calls

    if not tool_calls:
        return result  # No tool call — response is the final answer

    # Step 3: Execute tools (if executor provided)
    if tool_executor:
        for call in tool_calls:
            try:
                tool_result = tool_executor(call["name"], call["arguments"])
                result["tool_results"].append({
                    "name": call["name"],
                    "result": tool_result,
                })
            except Exception as e:
                result["tool_results"].append({
                    "name": call["name"],
                    "error": str(e),
                })

        # Step 4: Generate final response with tool results
        # (For now, just return the tool calls — full multi-turn
        # inference requires building the full message chain)

    return result


# =========================================================================
# INTERACTIVE MODE
# =========================================================================

# Default tools for interactive testing
DEFAULT_TOOLS = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
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
    },
]


def interactive_mode(model, tokenizer):
    """Run an interactive REPL for testing tool calling."""
    print()
    print("=" * 60)
    print("INTERACTIVE TOOL-CALLING TEST")
    print("=" * 60)
    print()
    print("Available tools: get_weather, search_web, send_email")
    print("Type 'quit' to exit, 'tools' to see tool definitions")
    print()

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "tools":
            print(json.dumps(DEFAULT_TOOLS, indent=2))
            continue

        print()
        response = generate_response(model, tokenizer, query, DEFAULT_TOOLS)

        # Check for tool calls
        tool_calls = extract_tool_calls(response)

        if tool_calls:
            print(f"Assistant: [TOOL CALL]")
            for call in tool_calls:
                print(f"  → {call['name']}({json.dumps(call.get('arguments', {}), indent=4)})")
        else:
            print(f"Assistant: {response}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Tool-calling inference")
    parser.add_argument("--query", type=str, help="Single query (non-interactive)")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapters")
    args = parser.parse_args()

    model, tokenizer = load_finetuned_model(args.adapter_path)

    if args.query:
        response = generate_response(model, tokenizer, args.query, DEFAULT_TOOLS)
        tool_calls = extract_tool_calls(response)
        if tool_calls:
            print("Tool calls:")
            for call in tool_calls:
                print(f"  {call['name']}({json.dumps(call.get('arguments', {}))})")
        else:
            print(f"Response: {response}")
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
```

---

## Step 2: Verify the script parses tool calls correctly (unit test)

Run this standalone test to verify the parser works before training completes:

```bash
cd slm-tool-calling-finetune
python -c "
from inference import extract_tool_calls, has_tool_call

# Test 1: Single tool call
response1 = '''<|function_calls|>
[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}]
<|/function_calls|>'''
calls1 = extract_tool_calls(response1)
assert len(calls1) == 1
assert calls1[0]['name'] == 'get_weather'
print('✅ Test 1: Single tool call parsed')

# Test 2: Parallel tool calls
response2 = '''<|function_calls|>
[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}, {\"name\": \"search_web\", \"arguments\": {\"query\": \"NYC restaurants\"}}]
<|/function_calls|>'''
calls2 = extract_tool_calls(response2)
assert len(calls2) == 2
print('✅ Test 2: Parallel tool calls parsed')

# Test 3: No tool call
response3 = 'The capital of France is Paris.'
calls3 = extract_tool_calls(response3)
assert len(calls3) == 0
assert not has_tool_call(response3)
print('✅ Test 3: No tool call correctly identified')

# Test 4: Malformed JSON
response4 = '<|function_calls|>not valid json<|/function_calls|>'
calls4 = extract_tool_calls(response4)
assert len(calls4) == 0
print('✅ Test 4: Malformed JSON handled gracefully')

print()
print('All parser tests passed!')
"
```

---

## Acceptance Criteria

- [ ] `inference.py` exists with all functions from the spec
- [ ] `extract_tool_calls()` correctly parses single, parallel, and empty tool calls
- [ ] `has_tool_call()` correctly detects presence/absence of tool calls
- [ ] `generate_response()` builds proper ChatML input with tools in system message
- [ ] Interactive mode works with `python inference.py`
- [ ] Single query mode works with `python inference.py --query "..."`
- [ ] Parser handles malformed JSON gracefully (returns empty list, no crash)
- [ ] All 4 parser unit tests pass

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model loading fails | Verify LoRA adapters exist at `outputs/lora_adapters/`. Training (Ticket 09) must complete first for full testing. |
| Generation produces garbage | Check temperature setting (should be low, ~0.1). Verify model was trained correctly. |
| Tool calls aren't being detected | Print the raw response to see exact format. The model may use different token names — check the vocab. |
| Infinite generation | pad_token bug — verify `tokenizer.pad_token != tokenizer.eos_token`. |
