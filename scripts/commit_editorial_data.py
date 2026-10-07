#!/usr/bin/env python3
"""Commit and push the editorial data files an editor changes through the app.

Four files under `data/` are tracked because they are hand-made and no
script can rebuild them: Varya's presenters, the match verdicts recorded in
the review tab, the notes topic list, and her Programs tab. Tracking gives a
baseline. It does not help with what happens *after*: the app appends a
verdict, the file changes on one machine, and nothing carries it to the
repository. This does.

Run by a timer. It is deliberately timid:

* only those four paths are ever staged, by name;
* a file that has lost rows since the last commit is refused, because the
  failure that costs editorial work is truncation, not addition;
* a diverged branch is left alone rather than merged or forced.

    scripts/commit_editorial_data.py            # commit and push
    scripts/commit_editorial_data.py --dry-run  # say what it would do
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Only these. A wildcard here would eventually sweep up a generated file.
EDITORIAL = (
    "data/library_presenters.csv",
    "data/youtube_map_decisions.csv",
    "data/library_note_topics.csv",
    "data/library_programs.csv",
)


class GitError(RuntimeError):
    pass


def git(*args: str, repo: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {(result.stderr or '').strip()}")
    return result.stdout.strip()


def line_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def committed(path: str, repo: Path = REPO_ROOT) -> str | None:
    """The file as HEAD has it, or None when HEAD does not have it yet."""
    try:
        return git("show", f"HEAD:{path}", repo=repo)
    except GitError:
        return None


def refuse_reason(path: str, repo: Path = REPO_ROOT) -> str | None:
    """Why this file must not be committed, or None when it is fine.

    Rows only ever get added to these files: a verdict is appended, a
    presenter is filled in. A file with fewer than it had is a truncation,
    and committing it would propagate the damage to the one copy that was
    supposed to survive it.
    """
    full = repo / path
    if not full.exists():
        return "missing"
    now = full.read_text(encoding="utf-8")
    if not now.strip():
        return "empty"
    before = committed(path, repo=repo)
    if before is None:
        return None
    if line_count(now) < line_count(before):
        return f"lost rows: {line_count(before)} -> {line_count(now)}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    args = parser.parse_args(argv)
    repo = Path(args.repo)

    if git("rev-parse", "--abbrev-ref", "HEAD", repo=repo) != args.branch:
        print(f"not on {args.branch}; leaving it alone")
        return 0

    changed = [p for p in EDITORIAL if git("status", "--porcelain", "--", p, repo=repo)]
    if not changed:
        print("no editorial changes")
        return 0

    refused = {p: why for p in changed if (why := refuse_reason(p, repo=repo))}
    for path, why in refused.items():
        print(f"REFUSING {path}: {why}")
    staging = [p for p in changed if p not in refused]
    if not staging:
        print("nothing safe to commit")
        return 1

    print("committing: " + ", ".join(staging))
    if args.dry_run:
        return 0

    git("fetch", "--quiet", args.remote, args.branch, repo=repo)
    behind = git("rev-list", "--count", f"HEAD..{args.remote}/{args.branch}", repo=repo)
    ahead = git("rev-list", "--count", f"{args.remote}/{args.branch}..HEAD", repo=repo)
    if behind != "0" and ahead != "0":
        # Somebody pushed while this box had a commit of its own. Merging
        # here would be this script making a judgement about code it did
        # not write, so it stops and says so instead.
        print(
            f"diverged from {args.remote}/{args.branch} ({ahead} ahead, {behind} behind); skipping"
        )
        return 1
    if behind != "0":
        git("merge", "--ff-only", f"{args.remote}/{args.branch}", repo=repo)

    git("add", "--", *staging, repo=repo)
    summary = ", ".join(Path(p).name for p in staging)
    git(
        "commit",
        "--quiet",
        "-m",
        f"chore(data): editorial changes from the app ({summary})\n\n"
        "Recorded on the deployment by an editor working in the Library, "
        "committed by scripts/commit_editorial_data.py so the work survives "
        "a rebuild.",
        repo=repo,
    )
    git("push", "--quiet", args.remote, args.branch, repo=repo)
    print(f"pushed {git('rev-parse', '--short', 'HEAD', repo=repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
