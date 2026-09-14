#!/usr/bin/env python3
"""Pull merged code onto the box and restart what needs restarting.

Production runs straight from this working tree, and the two Streamlit units
cache everything `app.py` imports for the life of the process. So a merge on
GitHub changes nothing here until somebody pulls, and a pull changes nothing
the editor sees until the units restart. Between 2026-09-12 and 2026-09-14 six
merged PRs sat unloaded for two days because that second step was skipped, and
a seventh was restarted fifteen seconds *before* the pull and loaded the old
tree. This script does the whole sequence, in order, and says what it did.

Run from a systemd timer every two minutes (see deploy/systemd/rainrag-deploy.*).
The repository is public, so `git fetch` needs no credentials. When there is
nothing new it exits quietly; the journal stays readable.

What it refuses to do, on purpose:
- deploy while the tree is dirty or checked out on anything but `main`, since
  that is somebody's work in progress (it notifies once, then stays quiet until
  the situation changes);
- anything but a fast-forward, since local commits on main mean a human has to
  look;
- roll back on a failed health check. It stops restarting further units, leaves
  the healthy one serving, and posts the failure. `Restart=on-failure` on the
  units already handles a crash loop; deciding what to do next is a person's job.

The two Streamlit units are restarted one at a time, the non-public one first
as a canary, each waited on until its health endpoint answers. The API and Slack
units are not restarted automatically: they import the same package, but they
hold uploads and transcriptions in flight, and nobody has yet asked for them to
bounce on every merge. When their code changes the notice says so.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Non-public unit first: if the new code fails to start, the public one is
# still serving the previous version while somebody reads the notice.
STREAMLIT_UNITS: dict[str, int] = {"rainrag-streamlit": 7860, "rainrag-streamlit-ip": 7861}

# A change to any of these is invisible to a running Streamlit process.
# app.py is deliberately absent: Streamlit re-executes it on every rerun.
RESTART_TRIGGERS: tuple[str, ...] = ("src/", "ui_library.py", "pyproject.toml", "uv.lock")
SYNC_TRIGGERS: tuple[str, ...] = ("pyproject.toml", "uv.lock")
UNIT_FILES_PREFIX = "deploy/systemd/"
# Modules the API and Slack connector run on. Changes here are reported, not
# acted on; see the module docstring.
SERVICE_CODE_PREFIX = "src/"

STATE_FILE = REPO_ROOT / "data" / "deploy_on_merge.state.json"
LOCK_FILE = REPO_ROOT / "data" / "deploy_on_merge.lock"


def _matches(path: str, triggers: Iterable[str]) -> bool:
    return any(path == t or (t.endswith("/") and path.startswith(t)) for t in triggers)


def units_to_restart(changed: Sequence[str]) -> list[str]:
    """Which Streamlit units a change set makes stale. Order is restart order."""
    if any(_matches(p, RESTART_TRIGGERS) for p in changed):
        return list(STREAMLIT_UNITS)
    return []


def needs_dependency_sync(changed: Sequence[str]) -> bool:
    """A lockfile change means the venv is behind before the restart even starts."""
    return any(_matches(p, SYNC_TRIGGERS) for p in changed)


def manual_attention(changed: Sequence[str]) -> list[str]:
    """Things a merge changed that this script will not act on by itself."""
    notes: list[str] = []
    if any(p.startswith(UNIT_FILES_PREFIX) for p in changed):
        notes.append("unit files under deploy/systemd/ changed: reinstall them by hand")
    if any(p.startswith(SERVICE_CODE_PREFIX) for p in changed):
        notes.append("src/ changed: rainrag-api and rainrag-slack were not restarted")
    return notes


@dataclass
class Plan:
    old: str
    new: str
    changed: list[str]
    units: list[str]
    sync: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = f"{self.old[:7]} → {self.new[:7]}, {len(self.changed)} file(s)"
        if self.units:
            head += f", restart {', '.join(self.units)}"
        if self.sync:
            head += ", uv sync"
        return head


class Git:
    def __init__(self, repo: Path):
        self.repo = repo

    def run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.repo, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    def branch(self) -> str:
        return self.run("rev-parse", "--abbrev-ref", "HEAD")

    def dirty(self) -> bool:
        return bool(self.run("status", "--porcelain", "--untracked-files=no"))

    def fetch(self, remote: str, branch: str) -> None:
        self.run("fetch", "--quiet", remote, branch)

    def sha(self, ref: str) -> str:
        return self.run("rev-parse", ref)

    def count(self, a: str, b: str) -> int:
        return int(self.run("rev-list", "--count", f"{a}..{b}") or 0)

    def changed(self, a: str, b: str) -> list[str]:
        out = self.run("diff", "--name-only", f"{a}..{b}")
        return [line for line in out.splitlines() if line]

    def fast_forward(self, ref: str) -> None:
        self.run("merge", "--ff-only", "--quiet", ref)


def default_restart(unit: str) -> None:
    subprocess.run(["sudo", "-n", "systemctl", "restart", f"{unit}.service"], check=True)


def default_health(port: int, timeout: float) -> bool:
    """True once Streamlit's health endpoint answers "ok" on this port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as r:  # noqa: S310
                if r.read().strip() == b"ok":
                    return True
        except Exception:  # noqa: BLE001 - the unit is still coming up
            pass
        time.sleep(2)
    return False


def default_sync_dependencies(repo: Path) -> None:
    uv = Path.home() / ".local" / "bin" / "uv"
    subprocess.run(
        [str(uv), "sync", "--frozen", "--extra", "analytics", "--extra", "sheets"],
        cwd=repo,
        check=True,
    )


def load_state(path: Path = STATE_FILE) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: dict, path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


MAX_ATTEMPTS = 2


@dataclass
class Deployer:
    repo: Path
    remote: str = "origin"
    branch: str = "main"
    health_timeout: float = 90.0
    restart: Callable[[str], None] = default_restart
    health: Callable[[int, float], bool] = default_health
    sync_dependencies: Callable[[Path], None] = default_sync_dependencies
    notify: Callable[[str], None] = lambda text: print(text)
    state_path: Path = STATE_FILE
    dry_run: bool = False

    def _skip(self, reason: str) -> int:
        """Say why nothing happened, but only when the reason is new.

        A dirty tree can stay dirty for an afternoon while somebody works; one
        notice per state is useful, one every two minutes is noise.
        """
        state = load_state(self.state_path)
        if state.get("last_skip_reason") != reason:
            self.notify(f"deploy skipped: {reason}")
            state["last_skip_reason"] = reason
            save_state(state, self.state_path)
        return 0

    def plan(self) -> Plan | None:
        """What to deploy, if anything.

        Two sources of work, not one. The obvious one is origin being ahead of
        the tree. The other is the tree being ahead of what the units actually
        run: a previous tick pulled and then failed on sync or restart, or a
        person pulled by hand and forgot the restart, which is how six merges
        sat unloaded for two days. Comparing origin to HEAD alone would call
        both situations "nothing to do" forever.
        """
        git = Git(self.repo)
        if git.branch() != self.branch:
            self._skip(f"tree is on '{git.branch()}', not '{self.branch}'")
            return None
        if git.dirty():
            self._skip("working tree has uncommitted changes")
            return None
        git.fetch(self.remote, self.branch)
        target = f"{self.remote}/{self.branch}"
        head, new = git.sha("HEAD"), git.sha(target)
        if git.count(target, "HEAD"):
            self._skip(f"local {self.branch} has commits not on {target}; not a fast-forward")
            return None

        state = load_state(self.state_path)
        deployed = state.get("last_deployed_sha")
        if deployed is None:
            # First run on a box that is current: take what is running as the
            # baseline rather than restarting everything for no reason.
            if head == new:
                state["last_deployed_sha"] = head
                save_state(state, self.state_path)
                return None
            deployed = head
        if deployed == new:
            return None
        if state.get("last_failed_sha") == new and state.get("failed_attempts", 0) >= MAX_ATTEMPTS:
            self._skip(
                f"deploy of {new[:7]} failed {MAX_ATTEMPTS} times at "
                f"'{state.get('failed_step')}'; holding until a new commit lands "
                f"or {self.state_path.name} is cleared"
            )
            return None
        changed = git.changed(deployed, new)
        return Plan(
            old=deployed,
            new=new,
            changed=changed,
            units=units_to_restart(changed),
            sync=needs_dependency_sync(changed),
            notes=manual_attention(changed),
        )

    def _fail(self, plan: Plan, step: str, detail: str) -> int:
        state = load_state(self.state_path)
        attempts = (
            state.get("failed_attempts", 0) + 1 if state.get("last_failed_sha") == plan.new else 1
        )
        # What the units are still running is the sha we set out from. Without
        # this, a first-ever run that failed left no baseline, the next tick
        # took the now-current tree as "deployed", and the retry never came.
        state.setdefault("last_deployed_sha", plan.old)
        state.update(
            {"last_failed_sha": plan.new, "failed_step": step, "failed_attempts": attempts}
        )
        state.pop("last_skip_reason", None)
        save_state(state, self.state_path)
        again = (
            "Will retry on the next tick."
            if attempts < MAX_ATTEMPTS
            else f"Attempt {attempts} of {MAX_ATTEMPTS}; not retrying until a new commit lands."
        )
        self.notify(
            f"DEPLOY FAILED at {step}: {plan.summary()}. {detail} Nothing was rolled back. {again}"
        )
        return 1

    def apply(self, plan: Plan) -> int:
        git = Git(self.repo)
        if self.dry_run:
            self.notify(f"dry run: would deploy {plan.summary()}")
            return 0
        try:
            if git.sha("HEAD") != plan.new:
                git.fast_forward(f"{self.remote}/{self.branch}")
        except subprocess.CalledProcessError as exc:
            return self._fail(plan, "fast-forward", f"git said: {exc.stderr or exc}".strip())
        if plan.sync:
            try:
                self.sync_dependencies(self.repo)
            except Exception as exc:  # noqa: BLE001 - reported through the failure path
                return self._fail(plan, "uv sync", f"{type(exc).__name__}: {exc}")
        restarted: list[str] = []
        for unit in plan.units:
            try:
                self.restart(unit)
            except Exception as exc:  # noqa: BLE001 - reported through the failure path
                return self._fail(plan, f"restart {unit}", f"{type(exc).__name__}: {exc}")
            if not self.health(STREAMLIT_UNITS[unit], self.health_timeout):
                still_old = [u for u in plan.units if u not in restarted and u != unit]
                left = f" Left on the previous code: {', '.join(still_old)}." if still_old else ""
                return self._fail(
                    plan,
                    f"health {unit}",
                    f"the unit did not report healthy within {self.health_timeout:.0f}s.{left} "
                    f"Check journalctl -u {unit}.",
                )
            restarted.append(unit)
        state = load_state(self.state_path)
        for key in ("last_skip_reason", "last_failed_sha", "failed_step", "failed_attempts"):
            state.pop(key, None)
        state.update(
            {"last_deployed_sha": plan.new, "deployed_at": time.strftime("%FT%TZ", time.gmtime())}
        )
        save_state(state, self.state_path)
        if plan.units or plan.notes:
            lines = [f"deployed {plan.summary()}"]
            lines += [f"  note: {n}" for n in plan.notes]
            self.notify("\n".join(lines))
        else:
            print(f"pulled {plan.summary()}; nothing to restart")
        return 0

    def run(self) -> int:
        plan = self.plan()
        if plan is None:
            return 0
        return self.apply(plan)


def slack_notifier(channel: str) -> Callable[[str], None]:
    """Post to Slack when configured, and always echo to stdout for the journal."""
    token = os.getenv("SLACK_BOT_TOKEN", "").strip()

    def notify(text: str) -> None:
        print(text)
        if channel and token:
            from scripts.health_check import post_slack_alert

            post_slack_alert(f":rocket: rainrag deploy: {text}", token=token, channel=channel)

    return notify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--health-timeout", type=float, default=90.0)
    parser.add_argument("--slack-channel", default="", help="channel id; empty means stdout only")
    parser.add_argument("--dry-run", action="store_true", help="fetch and report, change nothing")
    args = parser.parse_args(argv)

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("another deploy run holds the lock; exiting")
            return 0
        deployer = Deployer(
            repo=Path(args.repo),
            remote=args.remote,
            branch=args.branch,
            health_timeout=args.health_timeout,
            notify=slack_notifier(args.slack_channel),
            dry_run=args.dry_run,
        )
        return deployer.run()


if __name__ == "__main__":
    raise SystemExit(main())
