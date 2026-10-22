"""Check the actual Git staging area before committing or publishing.

Print filenames and issue categories only, never secret contents. This is a
best-effort guard, not a replacement for reviewing new documents and Git history.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "data/samples/attention_basics.md",
    "data/samples/evaluation_notes.md",
    "data/samples/rag_notes.md",
}
SECRET_PATTERN = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def local_secrets() -> list[str]:
    """Compare local credential values without sending them to tools or output."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return []
    values = []
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        name, separator, raw = line.partition("=")
        if separator and re.search(r"key|token|password|secret", name, re.IGNORECASE):
            value = raw.strip().strip("\"'")
            if len(value) >= 12:
                values.append(value)
    return values


def main() -> int:
    entries = [entry for entry in git("ls-files", "--stage", "-z").split(b"\0") if entry]
    if not entries:
        print("Nothing staged. Add reviewed source files before running this check.")
        return 1
    secrets = local_secrets()
    home = str(Path.home())
    home_variants = {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}
    findings = []
    size = 0
    for entry in entries:
        metadata, raw_name = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.split()
        name = raw_name.decode("utf-8")
        path = Path(name)
        if (
            stage != b"0"
            or mode == b"120000"
            or (name.startswith("data/") and name not in SAMPLES)
            or any(
                part in {".venv", "work", "uploads", "attachments", "secrets", "credentials"}
                for part in path.parts
            )
            or (path.name.startswith(".env") and path.name != ".env.example")
            or path.name.startswith("secrets.")
            or path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx"}
        ):
            findings.append((name, "private file, symlink, or unresolved index entry"))
            continue
        content = git("cat-file", "blob", object_id.decode("ascii"))
        size += len(content)
        if len(content) > 1024 * 1024:
            findings.append((name, "file exceeds the source-only 1 MB limit"))
            continue
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            findings.append((name, "binary file requires manual review"))
            continue
        if SECRET_PATTERN.search(text) or any(secret in text for secret in secrets):
            findings.append((name, "possible credential"))
        if any(value.lower() in text.lower() for value in home_variants):
            findings.append((name, "local home directory"))
    for name, reason in findings:
        print(f"BLOCKED: {name}: {reason}")
    if findings:
        return 1
    print(
        f"PASS: {len(entries)} staged text files, {size:,} bytes; no detected private files or secrets."
    )
    print("Only the three reviewed synthetic samples are allowed from data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
