"""Tests for scripts/deploy_on_merge.py.

The git side is real: two clones of a bare repository, one playing GitHub and
one playing the box. Only the side effects nobody wants in a test run are
faked: restarting units, polling health, syncing the venv, posting to Slack.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import deploy_on_merge as dom


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, rel: str, text: str, message: str) -> str:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path]:
    """(github, box): the box is a clone one commit behind nothing yet."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    github = tmp_path / "github"
    _git(tmp_path, "clone", "-q", str(origin), str(github))
    for repo in (github,):
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "t")
    _commit(github, "app.py", "print(1)\n", "initial")
    _git(github, "push", "-q", "-u", "origin", "main")
    box = tmp_path / "box"
    _git(tmp_path, "clone", "-q", str(origin), str(box))
    _git(box, "config", "user.email", "t@example.com")
    _git(box, "config", "user.name", "t")
    return github, box


class Fakes:
    def __init__(self, healthy: bool | list[bool] = True):
        self.restarted: list[str] = []
        self.synced = 0
        self.notices: list[str] = []
        self._healthy = healthy if isinstance(healthy, list) else [healthy] * 10

    def restart(self, unit: str) -> None:
        self.restarted.append(unit)

    def health(self, port: int, timeout: float) -> bool:
        return self._healthy.pop(0)

    def sync(self, repo: Path) -> None:
        self.synced += 1

    def notify(self, text: str) -> None:
        self.notices.append(text)


def _deployer(box: Path, fakes: Fakes, tmp_path: Path, **kw) -> dom.Deployer:
    return dom.Deployer(
        repo=box,
        restart=fakes.restart,
        health=fakes.health,
        sync_dependencies=fakes.sync,
        notify=fakes.notify,
        state_path=tmp_path / "state.json",
        **kw,
    )


# --------------------------------------------------------------------------- #
# Pure decisions
# --------------------------------------------------------------------------- #


def test_library_code_restarts_both_streamlit_units_canary_first():
    assert dom.units_to_restart(["src/rainrag/library_blend.py"]) == [
        "rainrag-streamlit",
        "rainrag-streamlit-ip",
    ]
    assert dom.units_to_restart(["ui_library.py"]) == ["rainrag-streamlit", "rainrag-streamlit-ip"]


def test_app_py_alone_needs_no_restart_because_streamlit_rereads_it():
    assert dom.units_to_restart(["app.py"]) == []


def test_docs_and_tests_need_no_restart():
    assert dom.units_to_restart(["README.md", "tests/unit/test_x.py", "data/exp/notes.md"]) == []


def test_a_lockfile_change_means_sync_then_restart():
    changed = ["uv.lock", "pyproject.toml"]
    assert dom.needs_dependency_sync(changed)
    assert dom.units_to_restart(changed)
    assert not dom.needs_dependency_sync(["src/rainrag/x.py"])


def test_manual_attention_flags_unit_files_and_service_code():
    notes = dom.manual_attention(["deploy/systemd/rainrag-api.service", "src/rainrag/api.py"])
    assert any("reinstall" in n for n in notes)
    assert any("rainrag-api" in n for n in notes)
    assert dom.manual_attention(["ui_library.py"]) == []


# --------------------------------------------------------------------------- #
# Planning against real repositories
# --------------------------------------------------------------------------- #


def test_nothing_to_do_when_the_box_is_current(repos, tmp_path):
    _, box = repos
    fakes = Fakes()
    assert _deployer(box, fakes, tmp_path).plan() is None
    assert fakes.notices == []


def test_plan_lists_the_changed_files_and_the_units_they_stale(repos, tmp_path):
    github, box = repos
    old = _git(box, "rev-parse", "HEAD")
    new = _commit(github, "src/rainrag/x.py", "x = 1\n", "feat")
    _git(github, "push", "-q")
    plan = _deployer(box, Fakes(), tmp_path).plan()
    assert plan is not None
    assert (plan.old, plan.new) == (old, new)
    assert plan.changed == ["src/rainrag/x.py"]
    assert plan.units == ["rainrag-streamlit", "rainrag-streamlit-ip"]
    assert "rainrag-api" in " ".join(plan.notes)


def test_a_dirty_tree_is_refused_and_reported_once(repos, tmp_path):
    github, box = repos
    _commit(github, "src/rainrag/x.py", "x = 1\n", "feat")
    _git(github, "push", "-q")
    (box / "app.py").write_text("print(2)\n", encoding="utf-8")
    fakes = Fakes()
    d = _deployer(box, fakes, tmp_path)
    assert d.plan() is None
    assert d.plan() is None
    assert len(fakes.notices) == 1
    assert "uncommitted" in fakes.notices[0]
    assert _git(box, "rev-parse", "HEAD") != _git(github, "rev-parse", "HEAD")


def test_a_tree_on_another_branch_is_refused(repos, tmp_path):
    github, box = repos
    _commit(github, "src/rainrag/x.py", "x = 1\n", "feat")
    _git(github, "push", "-q")
    _git(box, "checkout", "-q", "-b", "scratch")
    fakes = Fakes()
    assert _deployer(box, fakes, tmp_path).plan() is None
    assert "scratch" in fakes.notices[0]


def test_local_commits_on_main_are_never_overwritten(repos, tmp_path):
    github, box = repos
    _commit(github, "src/rainrag/x.py", "x = 1\n", "remote feat")
    _git(github, "push", "-q")
    local = _commit(box, "notes.md", "local\n", "local work")
    fakes = Fakes()
    assert _deployer(box, fakes, tmp_path).plan() is None
    assert "fast-forward" in fakes.notices[0]
    assert _git(box, "rev-parse", "HEAD") == local


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #


def test_apply_fast_forwards_restarts_in_order_and_reports(repos, tmp_path):
    github, box = repos
    new = _commit(github, "ui_library.py", "ui = 1\n", "feat")
    _git(github, "push", "-q")
    fakes = Fakes()
    d = _deployer(box, fakes, tmp_path)
    assert d.run() == 0
    assert _git(box, "rev-parse", "HEAD") == new
    assert fakes.restarted == ["rainrag-streamlit", "rainrag-streamlit-ip"]
    assert fakes.synced == 0
    assert any(n.startswith("deployed") for n in fakes.notices)
    state = dom.load_state(tmp_path / "state.json")
    assert state["last_deployed_sha"] == new


def test_a_lockfile_change_syncs_before_restarting(repos, tmp_path):
    github, box = repos
    _commit(github, "uv.lock", "lock\n", "deps")
    _git(github, "push", "-q")
    fakes = Fakes()
    assert _deployer(box, fakes, tmp_path).run() == 0
    assert fakes.synced == 1
    assert fakes.restarted


def test_a_failed_health_check_stops_before_the_public_unit(repos, tmp_path):
    """The canary fails; the public unit must be left on the previous code."""
    github, box = repos
    new = _commit(github, "src/rainrag/x.py", "x = 1\n", "feat")
    _git(github, "push", "-q")
    fakes = Fakes(healthy=[False])
    d = _deployer(box, fakes, tmp_path, health_timeout=1)
    assert d.run() == 1
    assert fakes.restarted == ["rainrag-streamlit"]
    failure = [n for n in fakes.notices if "DEPLOY FAILED" in n]
    assert failure and "rainrag-streamlit-ip" in failure[0]
    # The pull itself is not undone: the code is on disk, one unit is still up.
    assert _git(box, "rev-parse", "HEAD") == new
    assert dom.load_state(tmp_path / "state.json")["failed_unit"] == "rainrag-streamlit"


def test_docs_only_merges_pull_quietly(repos, tmp_path, capsys):
    github, box = repos
    _commit(github, "README.md", "docs\n", "docs")
    _git(github, "push", "-q")
    fakes = Fakes()
    assert _deployer(box, fakes, tmp_path).run() == 0
    assert fakes.restarted == []
    assert fakes.notices == []
    assert "nothing to restart" in capsys.readouterr().out


def test_dry_run_changes_nothing(repos, tmp_path):
    github, box = repos
    old = _git(box, "rev-parse", "HEAD")
    _commit(github, "src/rainrag/x.py", "x = 1\n", "feat")
    _git(github, "push", "-q")
    fakes = Fakes()
    assert _deployer(box, fakes, tmp_path, dry_run=True).run() == 0
    assert _git(box, "rev-parse", "HEAD") == old
    assert fakes.restarted == []
    assert fakes.notices and fakes.notices[0].startswith("dry run")


def test_a_successful_deploy_clears_an_old_skip_reason(repos, tmp_path):
    github, box = repos
    state = tmp_path / "state.json"
    dom.save_state({"last_skip_reason": "working tree has uncommitted changes"}, state)
    _commit(github, "ui_library.py", "ui = 1\n", "feat")
    _git(github, "push", "-q")
    assert _deployer(box, Fakes(), tmp_path).run() == 0
    assert "last_skip_reason" not in dom.load_state(state)


def test_state_file_survives_garbage(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json", encoding="utf-8")
    assert dom.load_state(path) == {}
    path.write_text("[1, 2]", encoding="utf-8")
    assert dom.load_state(path) == {}
