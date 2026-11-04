#!/usr/bin/env python
"""Expand \\input{...} directives for pandoc (single-file LaTeX)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


def expand_file(path: Path, base: Path, stack: set[Path]) -> str:
    resolved = (base / path).resolve()
    if resolved in stack:
        raise RuntimeError(f"Circular \\input: {resolved}")
    stack.add(resolved)
    text = resolved.read_text(encoding="utf-8")
    out_parts: list[str] = []

    def repl(match: re.Match[str]) -> str:
        rel = match.group(1).strip()
        if not rel.endswith(".tex"):
            rel = f"{rel}.tex"
        child = (base / rel).resolve()
        if not child.exists():
            return f"% missing input: {rel}\n"
        return expand_file(child.relative_to(base), base, stack)

    pos = 0
    for match in INPUT_RE.finditer(text):
        out_parts.append(text[pos : match.start()])
        out_parts.append(repl(match))
        pos = match.end()
    out_parts.append(text[pos:])
    stack.remove(resolved)
    return "".join(out_parts)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    latex_dir = root / "reports" / "latex"
    main_tex = latex_dir / "main.tex"
    out_dir = latex_dir / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "main_expanded.tex"
    expanded = expand_file(Path("main.tex"), latex_dir, set())
    out_path.write_text(expanded, encoding="utf-8")
    print(f"Expanded: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
