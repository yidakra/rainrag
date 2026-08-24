# RainRAG health check

`scripts/health_check.py` probes the deployed stack. **It is not installed.** The
unit files here are checked in; enabling them is a deliberate human step, below.

(Older units live in [`deploy/systemd/`](systemd/README.md). The health units sit
at the top of `deploy/` because that is where the task that added them asked for
them; move them into `systemd/` if you prefer one directory.)

## Quiet unless broken

Healthy run: **prints nothing, exits 0.**

That is the entire point. This runs 48 times a day. If it printed a paragraph
every time, the journal would fill with a status report nobody reads, and the one
run that mattered would be scrolled past with the rest. Output means "a human
should look at this".

How the signal reaches you, mechanically:

- `rainrag-health.service` is `Type=oneshot`. A healthy run exits 0, writes
  nothing to stdout, and the unit goes back to `inactive (dead)` leaving no
  journal entry.
- A failed check exits 1. systemd marks the unit **failed**, so it shows up in
  `systemctl list-units --failed`, and the report — the only output the script
  ever produces — is in `journalctl -u rainrag-health`.
- Nothing emails or pages you by itself. If you want that, add a drop-in with
  `OnFailure=` pointing at your own notifier:
  `sudo systemctl edit rainrag-health.service`.

## Install

```bash
sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-health.service /etc/systemd/system/
sudo cp /home/ubuntu/rainrag/deploy/systemd/rainrag-health.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rainrag-health.timer
```

Check it:

```bash
systemctl list-timers rainrag-health.timer          # when it next runs
sudo systemctl start rainrag-health.service         # run it now
systemctl is-failed rainrag-health.service          # "failed" == something is wrong
journalctl -u rainrag-health --no-pager             # the report, if there was one
```

Uninstall: `sudo systemctl disable --now rainrag-health.timer`, then remove the
two files from `/etc/systemd/system/` and `daemon-reload`.

The timer fires on `OnCalendar=*:0/30` with `RandomizedDelaySec=3m`, so roughly
every 30 minutes, jittered by up to three minutes. There is no `Persistent=`:
a missed check has no backlog worth replaying, since the next run reports the
state of the world as it is then.

## Running it by hand

```bash
/home/ubuntu/rainrag/.venv/bin/python3 scripts/health_check.py --verbose
```

`--verbose` prints every check including the passing ones, which is what you want
when you are asking "is anything wrong?" rather than being told. Exit code is the
same either way.

## What it checks, and why those thresholds

| Check | Fails when | Protects against |
| --- | --- | --- |
| `api` | `http://localhost:8001/health` is not 200 within `--timeout` (5s) | the FastAPI process being down — every search in the UI errors |
| `streamlit` | either port 7860 or 7861 is not 200 | the two UIs are separate units, so one can die while the other lives |
| `imports` | over `--window-hours` (24), with at least `--min-attempts` (3) attempts, the failed share exceeds `--failure-threshold` (50%) | silent breakage of video import: yt-dlp getting blocked, Telegram credentials expiring, a platform changing its player |
| `disk` | the filesystem behind `/tmp/rainrag_hls_cache` or `./data` is over `--disk-threshold` (90%) used | HLS segments and downloaded video accumulate; a full disk breaks imports in a confusing way |
| `session` | `data/telegram.session` is missing while `video_upload.telegram_enabled` is true, or its mode is looser than 0600 | a broken Telegram import path, and a live Telegram credential left readable by other users |

**Why a single failure does not alarm.** A failed import is often the system
working correctly: a journalist pastes a link to a deleted post or a
region-locked video and gets a clear error. One failure out of one attempt is a
100% failure rate and means nothing. So the check needs at least three attempts
in the window before it will speak, and the failure share must *strictly exceed*
50% — a run of bad luck sitting exactly on the line is not an incident. What this
is built to catch is the different shape: most or all attempts failing, which is
what a blocked downloader or expired credential looks like.

**Why the HTTP probes retry once.** Restarting `rainrag-api` leaves a few seconds
where nothing answers on 8001, and a deploy is not an outage. Each probe is
retried once, three seconds later, and only a second failure is reported (the
report then says `2 attempts`). A healthy service still costs the check nothing:
the retry only happens after a failure.

Two more deliberate choices:

- Reasons are listed in the report (`3x blocked (platform refused us)`), because
  `blocked` and `geo` mean "the platform is refusing us" while `no_media` mostly
  means "users pasted bad links" — different responses.
- Disk usage is `(total - free) / total`, which reads a percent or two higher
  than `df`'s `Use%`. `df` excludes the root-reserved blocks from its
  denominator; those blocks are not available to the `ubuntu` user either, so the
  pessimistic number is the honest one. Do not be surprised when it disagrees
  with `df` by a hair.

`--window-hours`, `--min-attempts`, `--failure-threshold` and `--disk-threshold`
are all flags; thresholds accept `50`, `50%` or `0.5`. Change them in
`ExecStart=` via a drop-in rather than editing the checked-in unit.

## Where the import data comes from

The API logs one machine-readable line per import
(`[usage] event=video_import outcome=http_503 ... reason=blocked`) and one per
question asked
(`[usage] event=query outcome=ok seconds=30.6 mode=corpus provider=mistral docs=5 tokens_in=1913 tokens_out=1346`).
Neither line ever carries the question text or the pasted URL.
`scripts/health_check.py` reads those through the same parser
`scripts/usage_report.py` uses, so the two tools can never disagree about what a
failure is. For the human-readable breakdown, run
`scripts/usage_report.py --days 7`.

The failure-rate check counts imports only. Questions are asked far more often
than videos are imported, so mixing them would swamp the import signal.

`--journal-file FILE` reads `[usage]` lines from a file instead of `journalctl`,
which is how the failure-rate path gets tested without waiting for real
breakage. On a host with no `journalctl` at all (a laptop), the import check is
skipped rather than failed — there is nothing to read and nothing is wrong.
