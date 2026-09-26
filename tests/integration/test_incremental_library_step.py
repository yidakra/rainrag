"""Step 4/4 of the hourly updater, exercised against the real shell script.

The step's whole point is that it is *tolerant*: a generator that fails must
log a warning and let the run finish, because the index is the job and a stale
link map is a degraded card, not a broken index. That tolerance is easy to
break silently -- the script runs under `set -euo pipefail` with an `ERR` trap,
and only the `if ! cmd` form is exempt from both -- so it is asserted here
rather than trusted.

The generators are stubbed: this covers the wiring, not their internals, which
`tests/unit/test_media_links.py` covers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_incremental_update.sh"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash"),
    pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock"),
]

STUB = """#!/usr/bin/env python3
import os, sys
print("STUB {name} " + " ".join(sys.argv[1:]))
sys.exit(int(os.environ.get("{env}", "0")))
"""

# Real python for `uv run python` (the script reads its config that way), a
# no-op for the rainrag pipeline steps we are not exercising.
UV_STUB = """#!/usr/bin/env bash
if [[ "${1:-}" == "run" && "${2:-}" == "python" ]]; then
  shift 2
  exec python3 "$@"
fi
exit 0
"""


def _deployment(tmp_path: Path, *, with_tags: bool = True) -> Path:
    root = tmp_path / "deploy"
    for sub in ("data", "embeddings", "archive", "scripts", "logs", "bin"):
        (root / sub).mkdir(parents=True)

    shutil.copy(SCRIPT, root / "scripts" / SCRIPT.name)
    for name, env in (
        ("library_untitled_titles", "TITLES_EXIT"),
        ("library_untitled_media", "MEDIA_EXIT"),
    ):
        (root / "scripts" / f"{name}.py").write_text(
            STUB.format(name=name, env=env), encoding="utf-8"
        )

    uv = root / "bin" / "uv"
    uv.write_text(UV_STUB, encoding="utf-8")
    uv.chmod(0o755)

    (root / "config.yaml").write_text(
        "incremental:\n"
        "  enabled: true\n"
        "  manifest_path: ./data/manifest.json\n"
        "paths:\n"
        f"  archive_root: {root / 'archive'}\n"
        "  docs_output: ./data/docs.jsonl\n"
        "  embeddings_cache: ./embeddings\n",
        encoding="utf-8",
    )
    (root / "data" / "manifest.json").write_text('{"a": 1}', encoding="utf-8")
    (root / "data" / "docs.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "embeddings" / "embeddings.npy").touch()
    (root / "embeddings" / "metadata.jsonl").touch()
    if with_tags:
        (root / "data" / "library_tags.jsonl").write_text('{"video_hash": "a"}\n', encoding="utf-8")
    return root


def _run(root: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{root / 'bin'}{os.pathsep}{env['PATH']}",
            "REPO_DIR": str(root),
            "RAINRAG_CONFIG": str(root / "config.yaml"),
            "LOCK_FILE": str(root / "lock"),
            "LOG_DIR": str(root / "logs"),
        }
    )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(root / "scripts" / SCRIPT.name)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _succeeded(root: Path, result: subprocess.CompletedProcess[str]) -> bool:
    return result.returncode == 0 and (root / "logs" / "incremental.last_success").exists()


def test_a_failing_generator_warns_without_failing_the_run(tmp_path: Path):
    """The one that matters: `if ! cmd` is exempt from `set -e` and the ERR trap.

    If that ever regresses, the trap fires, the run dies before `finalize`,
    and a stale link map starts withholding `incremental.last_success` --
    paging somebody about the wrong thing.
    """
    root = _deployment(tmp_path)
    result = _run(root, MEDIA_EXIT="3")

    assert "WARNING" in result.stdout
    assert "keeps the previous file" in result.stdout
    assert _succeeded(root, result), result.stdout


def test_both_generators_run_with_the_configured_archive_root(tmp_path: Path):
    root = _deployment(tmp_path)
    result = _run(root)

    assert f"STUB library_untitled_titles --tags {root / 'data' / 'library_tags.jsonl'}" in (
        result.stdout
    )
    assert f"--archive-root {root / 'archive'}" in result.stdout
    assert f"STUB library_untitled_media --tags {root / 'data' / 'library_tags.jsonl'}" in (
        result.stdout
    )
    assert _succeeded(root, result), result.stdout


def test_the_step_is_skipped_when_asked(tmp_path: Path):
    root = _deployment(tmp_path)
    result = _run(root, SKIP_LIBRARY="1")

    assert "skipped via SKIP_LIBRARY=1" in result.stdout
    assert "STUB library_untitled" not in result.stdout
    assert _succeeded(root, result), result.stdout


def test_no_tag_file_is_a_normal_state_not_a_warning(tmp_path: Path):
    """A deployment with no tagging run has nothing for the Library to show."""
    root = _deployment(tmp_path, with_tags=False)
    result = _run(root)

    assert "skipped: no tag file" in result.stdout
    assert "WARNING" not in result.stdout
    assert _succeeded(root, result), result.stdout


def test_an_unmounted_archive_root_skips_rather_than_regenerating(tmp_path: Path):
    """Nothing should run against a tree that is not there."""
    root = _deployment(tmp_path)
    (root / "archive").rmdir()
    result = _run(root)

    assert "is not a directory" in result.stdout
    assert "STUB library_untitled" not in result.stdout
    assert _succeeded(root, result), result.stdout


def test_an_unset_archive_root_skips_rather_than_using_the_repo(tmp_path: Path):
    """`Path("")` is relative, so an absent key must not normalise to the repo."""
    root = _deployment(tmp_path)
    config = root / "config.yaml"
    config.write_text(
        "\n".join(
            line
            for line in config.read_text(encoding="utf-8").splitlines()
            if "archive_root" not in line
        )
        + "\n",
        encoding="utf-8",
    )
    result = _run(root)

    assert "paths.archive_root is not set" in result.stdout
    assert "STUB library_untitled" not in result.stdout
    assert _succeeded(root, result), result.stdout
