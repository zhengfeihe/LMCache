# SPDX-License-Identifier: Apache-2.0
"""Send one HMA probe request and save the completion for comparison.

The prompt is a numbered list of rules, each embedding a unique codeword,
followed by a question answerable only by reading codewords back out of the
prefix -- so a faithful answer requires attention over the whole cached
prefix, and corrupted KV shows up as a wrong or missing codeword. Requests go
to /v1/completions at temperature 0, so a given prompt has one correct
continuation.

run-hma-probe.sh owns the phase protocol: which prompts to send, when to
reset vLLM's local prefix cache, and which completions to compare.
"""

# Standard
import argparse
import json
import sys
import urllib.request

RULE_TEMPLATE = (
    "Rule %d: The secret codeword of rule %d is kiwi-%d. Always be precise, "
    "cite the rule number you applied, and keep the answer short. "
)
QUESTION = (
    "\n\nQuestion: List the secret codewords of rules 3, 17, and 41, "
    "one per line."
)
# Qwen3.5 opens a <think> reasoning block by default, which would fill the
# completion with a preamble instead of the codewords the probe compares.
# /v1/completions applies no chat template, so close the block in the prompt
# ourselves -- byte-identical to what the model's own chat template emits for
# enable_thinking=false (chat_template.jinja: '<think>\n\n</think>\n\n').
THINK_OFF = "\n\n<think>\n\n</think>\n\n"


def build_prompt(num_rules: int) -> str:
    """Build the deterministic probe prompt.

    Args:
        num_rules: Number of codeword-bearing rules in the prefix. Must be
            large enough that the prefix spans several LMCache chunks and
            covers every rule number named in the question.

    Returns:
        The full prompt string (prefix plus question).
    """
    prefix = "You are a careful assistant.\n" + "\n".join(
        RULE_TEMPLATE % (i, i, i) for i in range(num_rules)
    )
    return prefix + QUESTION + THINK_OFF


def send_probe(port: int, model: str, prompt: str, max_tokens: int) -> str:
    """Send the probe completion request and return the generated text.

    Args:
        port: TCP port of the vLLM /v1/completions endpoint.
        model: Served model name to put in the request body.
        prompt: The exact prompt string to complete.
        max_tokens: Completion length cap; large enough for three codewords.

    Returns:
        The completion text.

    Raises:
        urllib.error.URLError: If the server is unreachable or rejects the
            request.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    usage = payload.get("usage", {})
    print(f"prompt_tokens={usage.get('prompt_tokens')}")
    return payload["choices"][0]["text"]


def main() -> int:
    """Run one probe phase and write the completion to the output file.

    Returns:
        Process exit code: 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="File to write the text to")
    parser.add_argument("--num-rules", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--cut-at-rule",
        type=int,
        default=-1,
        help=(
            "When >= 0, send only the strict character prefix of the full "
            "prompt that ends right before this rule (no question). Used to "
            "warm vLLM's local prefix cache with a shorter prefix than "
            "LMCache holds, so the next full request takes the mixed "
            "local-plus-external path."
        ),
    )
    args = parser.parse_args()

    prompt = build_prompt(args.num_rules)
    if args.cut_at_rule >= 0:
        prompt = prompt[: prompt.index("Rule %d:" % args.cut_at_rule)]

    text = send_probe(args.port, args.model, prompt, args.max_tokens)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"completion: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
