"""Execute every offline Python example from the README as a pytest test module."""

import ast
import re
from pathlib import Path


def test_readme_examples(pytester):
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8-sig")
    examples = []
    count = 0
    for source in re.findall(r"```python\n(.*?)\n```", readme, re.S):
        tree = ast.parse(source)
        functions = [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        # The live recorder example is covered by SDK/CLI transport tests instead.
        if any(any(arg.arg == "llm_recorder" for arg in node.args.args) for node in functions):
            continue
        count += sum(node.name.startswith("test_") for node in functions)
        examples.append(source)
    assert count >= 9
    pytester.makepyfile("\n\n".join(examples))
    pytester.runpytest("-q").assert_outcomes(passed=count)
