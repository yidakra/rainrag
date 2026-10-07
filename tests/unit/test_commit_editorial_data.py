"""Tests for the timer that carries editorial changes into the repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from commit_editorial_data import EDITORIAL, main, refuse_reason  # noqa: E402


DECISIONS = "data/youtube_map_decisions.csv"


def _repo(tmp_path: Path) -> Path:
    """A throwaway repo with an origin, shaped like the deployment."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "--initial-branch=main", str(origin)], check=True
    )
    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", str(origin), str(repo)], check=True)

    def run(*a):
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)

    run("config", "user.email", "t@example.test")
    run("config", "user.name", "t")
    (repo / "data").mkdir()
    for path in EDITORIAL:
        (repo / path).write_text("header\nrow1\n", encoding="utf-8")
    (repo / "code.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "init")
    run("branch", "-M", "main")
    run("push", "-q", "-u", "origin", "main")
    return repo


def _append(repo: Path, path: str, line: str) -> None:
    full = repo / path
    full.write_text(full.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")


def _clone_and_push(tmp_path: Path, name: str) -> None:
    """Somebody else pushing to the same origin."""
    other = tmp_path / name
    subprocess.run(["git", "clone", "-q", str(tmp_path / "origin.git"), str(other)], check=True)
    for cfg in (("user.email", "o@example.test"), ("user.name", "o")):
        subprocess.run(["git", "-C", str(other), "config", *cfg], check=True)
    (other / "code.py").write_text("x = 99\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(other), "commit", "-qam", "elsewhere"], check=True)
    subprocess.run(["git", "-C", str(other), "push", "-q"], check=True)


def test_a_recorded_verdict_reaches_the_repository(tmp_path, capsys):
    """The whole point: an editor's work on one machine must not stay there."""
    repo = _repo(tmp_path)
    _append(repo, DECISIONS, "abc,412707,match,2026-10-07T13:56:01+00:00")

    assert main(["--repo", str(repo)]) == 0
    assert "pushed" in capsys.readouterr().out

    pushed = subprocess.run(
        ["git", "-C", str(repo), "show", f"origin/main:{DECISIONS}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "412707" in pushed


def test_nothing_to_do_is_not_a_failure(tmp_path, capsys):
    repo = _repo(tmp_path)
    assert main(["--repo", str(repo)]) == 0
    assert "no editorial changes" in capsys.readouterr().out


def test_a_truncated_file_is_refused(tmp_path, capsys):
    """Rows only ever get added to these. Fewer than before is a truncation,
    and committing it would propagate the damage to the copy that was
    supposed to survive it."""
    repo = _repo(tmp_path)
    (repo / DECISIONS).write_text("header\n", encoding="utf-8")

    assert main(["--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out and "lost rows" in out


def test_an_emptied_file_is_refused(tmp_path):
    repo = _repo(tmp_path)
    (repo / DECISIONS).write_text("", encoding="utf-8")
    assert refuse_reason(DECISIONS, repo=repo) == "empty"


def test_only_the_four_editorial_paths_are_ever_staged(tmp_path, capsys):
    """A generated file changing must not be swept along with them."""
    repo = _repo(tmp_path)
    (repo / "data" / "library_tags.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")

    assert main(["--repo", str(repo)]) == 0
    assert "no editorial changes" in capsys.readouterr().out
    # The unrelated code edit is still sitting there, uncommitted.
    assert subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", "code.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_a_diverged_branch_is_left_alone(tmp_path, capsys):
    """Merging here would be this script judging code it did not write."""
    repo = _repo(tmp_path)
    _clone_and_push(tmp_path, "other")

    _append(repo, DECISIONS, "abc,412707,match,2026-10-07T13:56:01+00:00")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "local"], check=True)
    _append(repo, DECISIONS, "def,999,match,2026-10-07T14:00:00+00:00")

    assert main(["--repo", str(repo)]) == 1
    assert "diverged" in capsys.readouterr().out


def test_a_branch_that_is_merely_behind_fast_forwards(tmp_path, capsys):
    repo = _repo(tmp_path)
    _clone_and_push(tmp_path, "other2")

    _append(repo, DECISIONS, "abc,412707,match,2026-10-07T13:56:01+00:00")
    assert main(["--repo", str(repo)]) == 0
    assert "pushed" in capsys.readouterr().out
