# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""d2m_jit user-facing error type.

`D2mJitError` wraps any exception raised while compiling or invoking a
`@d2m.kernel` function and pins it to a (file, line, col) in the user's
Python source. The formatted message includes a short code excerpt and an
optional `did you mean?` hint so that the offending line is visible at the
top of the traceback rather than buried in `_src/ast.py`.

The class is intentionally a plain `Exception` subclass: tests can match
on `D2mJitError`, but `__cause__` is still set to the originating
`ValueError`/`KeyError`/`TypeError`/... so `pytest.raises(...)` against the
underlying type also works via `from`.
"""

import difflib
from typing import Iterable, Optional


class D2mJitError(Exception):
    """Error raised by `d2m_jit` with attached Python-source location.

    Construction is normally indirect -- `D2MCompiler._fail()` /
    `_format_error()` build this object from an AST node. Tests are
    expected to match on the formatted `str(exc)` (which includes
    `file:line`) and/or on `exc.cause` for the original exception type.
    """

    def __init__(
        self,
        msg: str,
        *,
        file: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        source_lines: Optional[Iterable[str]] = None,
        snippet_line: Optional[int] = None,
        hint: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ):
        self.msg = msg
        self.file = file
        self.line = line
        self.col = col
        self.source_lines = list(source_lines) if source_lines is not None else []
        self.snippet_line = snippet_line
        self.hint = hint
        self.cause = cause
        super().__init__(self._format())

    # ------------------------------------------------------------------
    def _format(self) -> str:
        parts = []
        loc = self.file or "<unknown>"
        if self.line is not None:
            loc = f"{loc}:{self.line}"
            if self.col is not None:
                loc = f"{loc}:{self.col}"
        parts.append(f"d2m_jit error at {loc}")

        if self.source_lines and self.snippet_line is not None:
            sl = self.snippet_line
            total = len(self.source_lines)
            start = max(sl - 1, 1)
            end = min(sl + 1, total)
            width = max(len(str(end)), 1)
            for n in range(start, end + 1):
                marker = "-->" if n == sl else "   "
                # Translate snippet line (1-indexed within the kernel
                # source) to the absolute file line for display.
                file_n = (self.line - sl + n) if self.line is not None else n
                text = self.source_lines[n - 1].rstrip("\n")
                parts.append(f"  {marker} {file_n:>{width}} | {text}")
                if n == sl and self.col is not None:
                    prefix = f"  {marker} {file_n:>{width}} | "
                    parts.append(" " * (len(prefix) + self.col) + "^")

        cause_name = type(self.cause).__name__ if self.cause else "Error"
        parts.append(f"{cause_name}: {self.msg}")
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return "\n".join(parts)


def closest_match(name: str, candidates: Iterable[str]) -> Optional[str]:
    """Return the single closest match for `name` in `candidates`, or None.

    Used to power `did you mean?` hints on unknown-function /
    unknown-attribute / unknown-variable errors.
    """
    candidates = [c for c in candidates if c and not c.startswith("_")]
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None
