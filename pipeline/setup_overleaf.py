"""Clone the legacy + new Overleaf projects and configure git credentials.

Reads `.thesis-overleaf.json` at the repo root (gitignored) which contains
project IDs and the Overleaf personal git token. Clones both projects to
the configured `localClone` paths if they do not already exist.

The Overleaf Git bridge URL is:
    https://git.overleaf.com/<projectId>

Authentication uses HTTP Basic with username `git` and password = the
personal token. To avoid storing the token in `.git/config` (which would
make it harder to rotate), we configure git's credential store globally
for the `git.overleaf.com` host. The token never lands in the chat
transcript or any committed file.

Idempotent: running twice does not re-clone.

Usage:
    python pipeline/setup_overleaf.py
    python pipeline/setup_overleaf.py --legacy-only   # only the template
    python pipeline/setup_overleaf.py --new-only      # only the push target
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / ".thesis-overleaf.json"
OVERLEAF_HOST = "git.overleaf.com"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(
            f"ERROR: {CONFIG_PATH.relative_to(REPO_ROOT)} not found.\n"
            f"  Copy {CONFIG_PATH.relative_to(REPO_ROOT)}.example "
            "and fill in your Overleaf project IDs + git token.",
            file=sys.stderr,
        )
        sys.exit(2)
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


_NO_GIT_ACCESS_HINT = """
HINT: Overleaf returned "no git access". This is a per-project setting --
each project must opt-in to git integration. To enable:

  1. Open the project at https://www.overleaf.com/project/<projectId>
  2. Click "Menu" (top-left hamburger icon)
  3. Look for "Sync" or "Git" section, then "Set up git integration"
  4. Enable git access. Re-run `make thesis-setup`.

The personal git token itself is fine; it's the project that's gated.
"""


def _redact(text: str, token: str) -> str:
    """Replace the literal token with [REDACTED] so logs don't leak."""
    return text.replace(token, "[REDACTED]") if token else text


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    redact_token: str | None = None,
) -> str:
    """Run a command, capturing output."""
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        err = res.stderr or ""
        if redact_token:
            err = _redact(err, redact_token)
        safe_cmd = " ".join(_redact(a, redact_token or "") for a in cmd)
        print(f"FAIL: {safe_cmd}", file=sys.stderr)
        print(f"  stderr: {err}", file=sys.stderr)
        if "no git access" in err.lower() or (
            "403" in err and "overleaf.com" in err.lower()
        ):
            print(_NO_GIT_ACCESS_HINT, file=sys.stderr)
        sys.exit(res.returncode)
    return res.stdout


def _configure_credential_helper(token: str) -> None:
    """Tell git to use the token for git.overleaf.com without storing in .git/config.

    Uses git's `credential.helper=store` mechanism with a per-URL credential
    line written to ~/.git-credentials. This file is read by git for any
    git.overleaf.com URL.

    The line written has the form:
        https://git:<token>@git.overleaf.com

    This is rotatable -- the user can replace the token by editing the line
    or running this script again with a new token.
    """
    home = Path.home()
    cred_file = home / ".git-credentials"
    target = f"https://git:{token}@{OVERLEAF_HOST}\n"

    existing: list[str] = []
    if cred_file.exists():
        existing = [
            ln
            for ln in cred_file.read_text(encoding="utf-8").splitlines(keepends=True)
            if OVERLEAF_HOST not in ln
        ]
    cred_file.write_text("".join(existing) + target, encoding="utf-8")
    # Restrict permissions on POSIX. Windows has its own ACL story; the file
    # is in the user's home dir which is already user-scoped.
    try:
        cred_file.chmod(0o600)
    except OSError:
        pass

    # Make sure credential.helper is configured globally.
    _run(["git", "config", "--global", "credential.helper", "store"])
    print(f"  configured ~/.git-credentials for {OVERLEAF_HOST}")


def _clone_project(project: dict, token: str) -> Path:
    """Clone the project if its localClone path does not exist yet.

    We use a clone URL with the token embedded (`https://git:TOKEN@host/id`)
    because Overleaf's git server requires authentication on the initial
    request and git only consults the credential helper if it sees a 401
    response for a URL that has a username but no password. After the clone
    succeeds we strip the token from the remote URL so it doesn't sit in
    `.git/config`; subsequent fetch/push pulls the token from the helper.
    """
    project_id = project["projectId"]
    local = Path(project["localClone"])
    if local.exists():
        if (local / ".git").exists():
            print(f"  already cloned: {local}")
            return local
        print(
            f"ERROR: {local} exists but is not a git repo. "
            "Refusing to clone over it.",
            file=sys.stderr,
        )
        sys.exit(2)

    clone_url = f"https://git:{token}@{OVERLEAF_HOST}/{project_id}"
    public_url = f"https://{OVERLEAF_HOST}/{project_id}"
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {project['name']} ({project_id}) -> {local}")
    _run(["git", "clone", clone_url, str(local)], redact_token=token)
    # Replace the token-bearing remote URL with the clean form. The credential
    # helper supplies the token for subsequent operations.
    _run(["git", "remote", "set-url", "origin", public_url], cwd=local)
    _run(["git", "config", "user.name", "Yurii Murha"], cwd=local)
    _run(["git", "config", "user.email", "yurii.murha@gymbeam.com"], cwd=local)
    return local


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Clone only the legacy template project (skip the push target).",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help=(
            "Also clone the legacy template project. Default skips it because "
            "the template files (`tukethesis.cls`, etc.) already live in LaTeX/."
        ),
    )
    args = parser.parse_args()

    cfg = _load_config()
    projects = cfg.get("projects", {})
    new = projects.get("new")
    legacy = projects.get("legacy")  # optional
    if not new:
        print("ERROR: config missing `projects.new`.", file=sys.stderr)
        return 2

    token = new.get("gitToken") or (legacy or {}).get("gitToken")
    if not token or token.startswith("REPLACE_"):
        print("ERROR: gitToken is missing or not yet filled in.", file=sys.stderr)
        return 2
    _configure_credential_helper(token)

    # Legacy clone is optional -- the LaTeX template is already in the repo
    # under LaTeX/, so the push-target clone is all we strictly need. Only
    # clone the legacy project if explicitly requested with --include-legacy.
    if args.legacy_only or (legacy and getattr(args, "include_legacy", False)):
        print("\nLegacy template project:")
        _clone_project(legacy, token)
    if not args.legacy_only:
        print("\nNew push-target project:")
        _clone_project(new, token)

    print("\nDone. Both project paths are ready for `make thesis-overleaf`.")
    print(
        "Tip: rotate the token by replacing the line in ~/.git-credentials, "
        "then editing .thesis-overleaf.json."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
