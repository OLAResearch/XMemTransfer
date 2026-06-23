"""Load optional repo-root HF credentials into os.environ.

Priority:
1. Existing environment variables
2. ``.hf_env`` with KEY=VALUE lines
3. ``hf.txt`` containing a raw token
"""

import os
import re
from pathlib import Path
from typing import Optional

_LOADED = False
_ASSIGN_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _expand_env_refs(val: str) -> str:
    def braced(m):
        return os.environ.get(m.group(1), "")

    def plain(m):
        return os.environ.get(m.group(1), "")

    out = re.sub(r"\$\{([^}]+)\}", braced, val)
    out = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", plain, out)
    return out


def _apply_dot_hf_env(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGN_RE.match(line)
        if not match:
            continue
        key, raw_val = match.group(1), match.group(2)
        if key in os.environ:
            continue
        os.environ[key] = _expand_env_refs(_strip_quotes(raw_val))


def _apply_hf_txt(path: Path) -> None:
    if "HF_TOKEN" in os.environ or "HUGGING_FACE_HUB_TOKEN" in os.environ:
        return
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def load_hf_env(project_root: Optional[Path] = None) -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    root = project_root or Path(__file__).resolve().parent.parent
    dot_env = root / ".hf_env"
    hf_txt = root / "hf.txt"

    if dot_env.is_file():
        _apply_dot_hf_env(dot_env)
    elif hf_txt.is_file():
        _apply_hf_txt(hf_txt)
