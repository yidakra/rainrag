# Deploy on merge

Production runs from the working tree at `/home/ubuntu/rainrag`, and the two
Streamlit units cache everything `app.py` imports for as long as the process
lives. A merge on GitHub therefore changes nothing on the box until someone
pulls, and a pull changes nothing the editor sees until the units restart.

Between 12 and 14 September 2026 six merged pull requests sat unloaded for two
days because the restart was skipped. The fix for the bug that finally exposed
them was then restarted fifteen seconds *before* the pull, and loaded the old
tree. This timer exists so that sequence is never done by hand again.

## What runs

`rainrag-deploy.timer` fires `rainrag-deploy.service` every two minutes. The
service runs `scripts/deploy_on_merge.py` as `ubuntu`, which:

1. Refuses to act if the tree is dirty, is not on `main`, or has local commits
   that `origin/main` lacks. It posts one notice per such state and then stays
   quiet until the state changes.
2. Fetches `origin/main`. If the box is current, exits silently.
3. Fast-forwards the tree.
4. If `pyproject.toml` or `uv.lock` changed, runs `uv sync --frozen` with the
   `analytics` and `sheets` extras, so the venv is current before anything
   restarts.
5. If anything under `src/`, or `ui_library.py`, or the dependency files
   changed, restarts `rainrag-streamlit` (port 7860, the non-public canary)
   and waits for its health endpoint, then `rainrag-streamlit-ip` (port 7861,
   rag.tvrain.tv). If the canary fails to come up healthy within 90 seconds it
   stops there, leaves the public unit on the previous code, and posts a
   failure. Nothing is rolled back; a person decides.
6. Treats "tree ahead of what the units run" as work too. The script keeps
   the last fully deployed sha in `data/deploy_on_merge.state.json`. If a tick
   pulled and then failed on sync or restart, or if a person pulled by hand
   and forgot the restart, the next tick finishes the job. A failed step is
   retried once; after two failures on the same commit it holds, posts one
   notice, and waits for a new commit or for the state file to be cleared.
7. Posts a one-line summary to the bot channel (`#rainrag-test`, C0BSBNC8AN7)
   when units were restarted or something needs a hand. Docs-only merges pull
   silently and show up only in the journal.

`app.py` alone does not trigger a restart: Streamlit re-executes it on every
rerun. `rainrag-api` and `rainrag-slack` are never restarted automatically.
They import the same package but hold uploads and transcriptions in flight;
when `src/` changes the notice says they were left alone.

## Install

```
sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-deploy.service /etc/systemd/system/
sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-deploy.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rainrag-deploy.timer
```

`ubuntu` needs to run `systemctl restart` for the two Streamlit units without
a password. It already has unrestricted passwordless sudo on this box; a
narrower rule, if that ever changes:

```
ubuntu ALL=(root) NOPASSWD: /usr/bin/systemctl restart rainrag-streamlit.service, /usr/bin/systemctl restart rainrag-streamlit-ip.service
```

## Check it

```
systemctl list-timers rainrag-deploy.timer
journalctl -u rainrag-deploy.service -n 50 --no-pager
cat /home/ubuntu/rainrag/data/deploy_on_merge.state.json
```

A dry run from a shell, changing nothing:

```
.venv/bin/python3 scripts/deploy_on_merge.py --dry-run
```

## Working on the box

The refusals are the point. While you are on a feature branch or have
uncommitted edits, nothing deploys, and one notice says so. Switch back to a
clean `main` and the next tick catches up. Unit files under `deploy/systemd/`
are not reinstalled automatically; the notice tells you when they changed.
