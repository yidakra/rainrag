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
    state = dom.load_state(tmp_path / "state.json")
    assert state["failed_step"] == "health rainrag-streamlit"
    assert state["last_failed_sha"] == new


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


# --------------------------------------------------------------------------- #
# Failures after the pull must be recorded, retried, then held
# --------------------------------------------------------------------------- #


class Flaky:
    """A sync that fails a given number of times, then works."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def __call__(self, repo: Path) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise subprocess.CalledProcessError(1, ["uv", "sync"], stderr="network down")


def test_a_failed_sync_is_reported_and_retried_on_the_next_tick(repos, tmp_path):
    """Before: the pull had moved HEAD, so the next tick saw nothing to do, forever."""
    github, box = repos
    _commit(github, "uv.lock", "lock\n", "deps")
    _git(github, "push", "-q")
    fakes = Fakes()
    flaky = Flaky(failures=1)
    d = _deployer(box, fakes, tmp_path)
    d.sync_dependencies = flaky
    assert d.run() == 1
    assert any("DEPLOY FAILED at uv sync" in n and "retry" in n for n in fakes.notices)
    assert fakes.restarted == []
    # Next tick: HEAD already equals origin, and the deploy still happens.
    assert d.run() == 0
    assert flaky.calls == 2
    assert fakes.restarted == ["rainrag-streamlit", "rainrag-streamlit-ip"]
    state = dom.load_state(tmp_path / "state.json")
    assert "last_failed_sha" not in state
    assert state["last_deployed_sha"] == _git(box, "rev-parse", "HEAD")


def test_repeated_failure_holds_with_one_notice_instead_of_looping(repos, tmp_path):
    github, box = repos
    _commit(github, "uv.lock", "lock\n", "deps")
    _git(github, "push", "-q")
    fakes = Fakes()
    d = _deployer(box, fakes, tmp_path)
    d.sync_dependencies = Flaky(failures=99)
    assert d.run() == 1
    assert d.run() == 1
    before = len(fakes.notices)
    assert d.run() == 0  # held, not attempted
    assert d.run() == 0
    held = fakes.notices[before:]
    assert len(held) == 1 and "holding" in held[0]
    assert fakes.restarted == []


def test_a_new_commit_lifts_the_hold(repos, tmp_path):
    github, box = repos
    _commit(github, "uv.lock", "lock\n", "deps")
    _git(github, "push", "-q")
    fakes = Fakes()
    d = _deployer(box, fakes, tmp_path)
    d.sync_dependencies = Flaky(failures=2)
    assert d.run() == 1 and d.run() == 1 and d.run() == 0
    _commit(github, "src/rainrag/y.py", "y = 1\n", "another")
    _git(github, "push", "-q")
    assert d.run() == 0
    assert fakes.restarted == ["rainrag-streamlit", "rainrag-streamlit-ip"]


def test_a_restart_that_raises_goes_through_the_failure_path(repos, tmp_path):
    github, box = repos
    _commit(github, "ui_library.py", "ui = 1\n", "feat")
    _git(github, "push", "-q")
    fakes = Fakes()

    def broken(unit: str) -> None:
        raise subprocess.CalledProcessError(1, ["systemctl"], stderr="sudo: a password is required")

    d = _deployer(box, fakes, tmp_path)
    d.restart = broken
    assert d.run() == 1
    assert any("DEPLOY FAILED at restart rainrag-streamlit" in n for n in fakes.notices)
    assert dom.load_state(tmp_path / "state.json")["failed_step"] == "restart rainrag-streamlit"


def test_a_manual_pull_without_a_restart_is_finished_on_the_next_tick(repos, tmp_path):
    """The 12 to 14 September failure mode, caught automatically this time."""
    github, box = repos
    fakes = Fakes()
    d = _deployer(box, fakes, tmp_path)
    assert d.run() == 0  # baseline recorded
    _commit(github, "ui_library.py", "ui = 1\n", "feat")
    _git(github, "push", "-q")
    _git(box, "pull", "-q", "--ff-only")  # a person pulled and walked away
    assert d.run() == 0
    assert fakes.restarted == ["rainrag-streamlit", "rainrag-streamlit-ip"]


def test_first_run_on_a_current_box_records_a_baseline_silently(repos, tmp_path):
    _, box = repos
    fakes = Fakes()
    assert _deployer(box, fakes, tmp_path).run() == 0
    assert fakes.restarted == [] and fakes.notices == []
    assert dom.load_state(tmp_path / "state.json")["last_deployed_sha"] == _git(
        box, "rev-parse", "HEAD"
    )


# --------------------------------------------------------------------------- #
# Git itself failing must not crash the tick
# --------------------------------------------------------------------------- #


def test_a_fetch_failure_is_noticed_once_and_never_crashes(repos, tmp_path, monkeypatch):
    """A network blip used to raise out of the oneshot every two minutes."""
    _, box = repos
    fakes = Fakes()
    d = _deployer(box, fakes, tmp_path)

    def offline(self, remote, branch):
        raise dom.GitError("fetch", "could not resolve host: github.com")

    monkeypatch.setattr(dom.Git, "fetch", offline)
    assert d.run() == 0
    assert d.run() == 0
    assert len(fakes.notices) == 1
    assert "git fetch failed" in fakes.notices[0]
    assert fakes.restarted == []


def test_when_git_recovers_the_pending_commit_deploys_and_the_outage_is_forgotten(
    repos, tmp_path, monkeypatch
):
    github, box = repos
    fakes = Fakes()
    d = _deployer(box, fakes, tmp_path)
    assert d.run() == 0  # baseline
    _commit(github, "ui_library.py", "ui = 1\n", "feat")
    _git(github, "push", "-q")
    real_fetch = dom.Git.fetch

    def offline(self, remote, branch):
        raise dom.GitError("fetch", "timeout")

    monkeypatch.setattr(dom.Git, "fetch", offline)
    assert d.run() == 0
    assert dom.load_state(tmp_path / "state.json")["last_skip_reason"].startswith("git ")
    monkeypatch.setattr(dom.Git, "fetch", real_fetch)
    assert d.run() == 0
    assert fakes.restarted == ["rainrag-streamlit", "rainrag-streamlit-ip"]
    assert "last_skip_reason" not in dom.load_state(tmp_path / "state.json")
    # A second outage later is announced again, not swallowed by the first.
    monkeypatch.setattr(dom.Git, "fetch", offline)
    before = len(fakes.notices)
    assert d.run() == 0
    assert len(fakes.notices) == before + 1


def test_git_errors_carry_the_subcommand_and_stderr(tmp_path):
    _git(tmp_path, "init", "-q", str(tmp_path / "r"))
    with pytest.raises(dom.GitError) as info:
        dom.Git(tmp_path / "r").run("rev-parse", "definitely-not-a-ref")
    assert info.value.step == "rev-parse"
    assert info.value.detail
