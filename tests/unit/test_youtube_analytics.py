"""Tests for the Analytics transform and CSV writing (no Google calls)."""

from __future__ import annotations

import stat
from pathlib import Path


def test_rows_to_dicts_zips_each_batch_with_its_own_headers():
    """A metric rejected on a later batch must not shift earlier rows."""
    from rainrag.youtube_analytics import rows_to_dicts

    wide = rows_to_dicts([{"name": "video"}, {"name": "views"}, {"name": "cpm"}], [["a", 10, 1.5]])
    narrow = rows_to_dicts([{"name": "video"}, {"name": "views"}], [["b", 20]])
    assert wide == [{"video": "a", "views": 10, "cpm": 1.5}]
    assert narrow == [{"video": "b", "views": 20}]


def test_rows_to_dicts_requires_the_video_dimension_only_when_rows_exist():
    import pytest

    from rainrag.youtube_analytics import rows_to_dicts

    assert rows_to_dicts([{"name": "views"}], []) == []
    with pytest.raises(ValueError):
        rows_to_dicts([{"name": "views"}], [[1]])


def test_rows_to_snapshot_maps_records_into_sheet_schema_with_blanks():
    from rainrag.youtube_analytics import CSV_COLUMNS, rows_to_snapshot

    out = rows_to_snapshot(
        [{"video": "a", "views": 1200, "playbackBasedCpm": 2.5}, {"video": "b", "views": 5}],
        "2026-09-09",
    )
    assert out[0]["views"] == 1200 and out[0]["playbackBasedCpm"] == 2.5
    assert out[0]["snapshot_date"] == "2026-09-09"
    assert out[1]["playbackBasedCpm"] == "" and out[1]["cpm"] == ""
    assert set(out[0]) == set(CSV_COLUMNS)


def test_append_snapshot_header_once_and_readable_by_load_metrics(tmp_path: Path):
    from rainrag.library_performance import load_metrics
    from rainrag.youtube_analytics import append_snapshot, rows_to_snapshot

    p = tmp_path / "metrics.csv"
    append_snapshot(
        p, rows_to_snapshot([{"video": "a", "views": 10, "playbackBasedCpm": 1.0}], "2026-09-01")
    )
    append_snapshot(
        p, rows_to_snapshot([{"video": "a", "views": 20, "playbackBasedCpm": 1.5}], "2026-09-09")
    )
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("youtube_id,snapshot_date,views") and len(lines) == 3
    assert load_metrics(p)["a"] == {"views": 20.0, "playbackBasedCpm": 1.5}


def test_token_file_is_owner_only(tmp_path: Path):
    from rainrag.youtube_analytics import _write_token

    t = tmp_path / "token.json"
    t.write_text("old", encoding="utf-8")
    t.chmod(0o644)
    _write_token(t, "{}")
    assert stat.S_IMODE(t.stat().st_mode) == 0o600
    assert t.read_text(encoding="utf-8") == "{}"


def test_chunked_and_rejected_metric_exact():
    from rainrag.youtube_analytics import _rejected_metric, chunked

    assert [len(c) for c in chunked([str(i) for i in range(1203)])] == [500, 500, 203]
    msg = "Unknown identifier (engagedViews) given in field parameters.metrics"
    assert _rejected_metric(msg, ["views", "engagedViews"]) == "engagedViews"
    # a shorter metric name must not match inside a longer identifier
    assert _rejected_metric("bad metric 'playbackBasedCpm'", ["cpm", "playbackBasedCpm"]) == (
        "playbackBasedCpm"
    )
    assert _rejected_metric("quota exceeded", ["views"]) is None


class _FakeCreds:
    def to_json(self) -> str:
        return '{"refresh_token": "r"}'


class _FakeFlow:
    """Stands in for InstalledAppFlow: CI installs no Google packages."""

    made: list[_FakeFlow] = []

    def __init__(self) -> None:
        self.code_verifier: str | None = None
        self.autogenerate_code_verifier = True
        self.redirect_uri: str | None = None
        self.fetched: tuple[str | None, str | None] | None = None
        self.credentials = _FakeCreds()

    @classmethod
    def from_client_secrets_file(cls, path: str, scopes: list[str]) -> _FakeFlow:
        flow = cls()
        cls.made.append(flow)
        return flow

    def authorization_url(self, **kwargs: object) -> tuple[str, str]:
        self.code_verifier = "verifier-minted-by-authorization-url"
        return ("https://accounts.google.com/o/oauth2/auth?code_challenge=abc", "state")

    def fetch_token(self, code: str | None = None) -> None:
        self.fetched = (code, self.code_verifier)


def _stub_google(monkeypatch) -> None:
    import sys
    import types

    _FakeFlow.made = []
    flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    flow_mod.InstalledAppFlow = _FakeFlow  # type: ignore[attr-defined]
    req_mod = types.ModuleType("google.auth.transport.requests")
    req_mod.Request = object  # type: ignore[attr-defined]
    cred_mod = types.ModuleType("google.oauth2.credentials")
    cred_mod.Credentials = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_mod)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", req_mod)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", cred_mod)


def test_consent_url_saves_the_pkce_verifier_owner_only(tmp_path: Path, monkeypatch):
    """--auth and --auth-code are separate processes; the verifier must outlive the first."""
    import pytest

    from rainrag.youtube_analytics import _verifier_path, load_credentials

    _stub_google(monkeypatch)
    client = tmp_path / "client.json"
    client.write_text("{}", encoding="utf-8")
    token = tmp_path / "token.json"

    with pytest.raises(SystemExit):
        load_credentials(client, token, auth_code=None)

    saved = _verifier_path(token)
    assert saved.read_text(encoding="utf-8") == "verifier-minted-by-authorization-url"
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert not token.exists()


def test_auth_code_exchange_reuses_the_saved_verifier_then_clears_it(tmp_path: Path, monkeypatch):
    from rainrag.youtube_analytics import _verifier_path, load_credentials

    _stub_google(monkeypatch)
    client = tmp_path / "client.json"
    client.write_text("{}", encoding="utf-8")
    token = tmp_path / "token.json"
    _verifier_path(token).write_text("verifier-from-the-earlier-run\n", encoding="utf-8")

    load_credentials(client, token, auth_code="4/code")

    assert _FakeFlow.made[-1].fetched == ("4/code", "verifier-from-the-earlier-run")
    assert token.read_text(encoding="utf-8") == '{"refresh_token": "r"}'
    assert not _verifier_path(token).exists()


def test_auth_code_without_a_saved_verifier_refuses_before_calling_google(
    tmp_path: Path, monkeypatch
):
    """Better a clear message than an invalid_grant from Google the user cannot read."""
    import pytest

    from rainrag.youtube_analytics import load_credentials

    _stub_google(monkeypatch)
    client = tmp_path / "client.json"
    client.write_text("{}", encoding="utf-8")
    token = tmp_path / "token.json"

    with pytest.raises(SystemExit, match="--auth"):
        load_credentials(client, token, auth_code="4/code")

    assert _FakeFlow.made[-1].fetched is None
    assert not token.exists()


# --------------------------------------------------------------------------- #
# Demographics: one request per video, two sheet columns, one blend vector
# --------------------------------------------------------------------------- #


def test_marginals_sum_the_joint_table_both_ways():
    from rainrag.youtube_analytics import marginals

    rows = [
        {"ageGroup": "age25-34", "gender": "female", "viewerPercentage": 20.0},
        {"ageGroup": "age25-34", "gender": "male", "viewerPercentage": 30.0},
        {"ageGroup": "age35-44", "gender": "female", "viewerPercentage": 10.0},
    ]
    age, gender = marginals(rows)
    assert age == {"age25-34": 50.0, "age35-44": 10.0}
    assert gender == {"female": 30.0, "male": 30.0}


def test_distribution_cell_round_trips_and_tolerates_junk():
    from rainrag.library_performance import format_distribution, parse_distribution

    cell = format_distribution({"age35-44": 10.04, "age25-34": 50.0})
    assert cell == "age25-34:50.0;age35-44:10.0"
    assert parse_distribution(cell) == {"age25-34": 50.0, "age35-44": 10.0}
    assert parse_distribution("") == {}
    assert parse_distribution(None) == {}
    assert parse_distribution("female:41,2;garbage;:9;male:58.8") == {"female": 41.2, "male": 58.8}


def test_rows_to_snapshot_carries_demographic_cells_and_blanks_them_when_absent():
    from rainrag.youtube_analytics import rows_to_snapshot

    with_demo = {"video": "a", "views": 5, "viewerPercentage: ageGroup": "age25-34:100.0"}
    without = {"video": "b", "views": 7}
    rows = rows_to_snapshot([with_demo, without], "2026-09-14")
    assert rows[0]["viewerPercentage: ageGroup"] == "age25-34:100.0"
    assert rows[0]["viewerPercentage: gender"] == ""
    assert rows[1]["viewerPercentage: ageGroup"] == ""


class _FakeHttpError(Exception):
    def __init__(self, status: int = 400):
        super().__init__(f"HTTP {status}")
        self.resp = type("Resp", (), {"status": status})()


class _FakeReports:
    def __init__(self, answers: dict[str, object]):
        self.answers = answers
        self.calls: list[dict] = []

    def query(self, **kw):
        self.calls.append(kw)
        vid = kw["filters"].split("==", 1)[1]
        answer = self.answers.get(vid)
        # A list means "answer these in turn", so a retry can be scripted.
        if isinstance(answer, list):
            answer = answer.pop(0) if answer else {}

        class _Exec:
            def execute(self):
                if isinstance(answer, Exception):
                    raise answer
                return answer or {}

        return _Exec()


def _stub_discovery(monkeypatch, reports: _FakeReports) -> None:
    import sys
    import types

    disc = types.ModuleType("googleapiclient.discovery")

    class _Svc:
        def reports(self):
            return reports

    disc.build = lambda *a, **k: _Svc()  # type: ignore[attr-defined]
    errs = types.ModuleType("googleapiclient.errors")
    errs.HttpError = _FakeHttpError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", disc)
    monkeypatch.setitem(sys.modules, "googleapiclient.errors", errs)


def test_fetch_video_demographics_is_one_request_per_video_and_zips_by_header(monkeypatch):
    from rainrag.youtube_analytics import fetch_video_demographics

    # Headers deliberately not in the order the code might assume.
    headers = [{"name": "gender"}, {"name": "viewerPercentage"}, {"name": "ageGroup"}]
    reports = _FakeReports(
        {
            "a": {
                "columnHeaders": headers,
                "rows": [["female", 60.0, "age25-34"], ["male", 40.0, "age25-34"]],
            },
            "b": {"columnHeaders": headers, "rows": []},  # too few views: withheld
            "c": _FakeHttpError(403),
        }
    )
    _stub_discovery(monkeypatch, reports)
    out, failed = fetch_video_demographics(
        object(), ["a", "b", "c"], "2015-01-01", "2026-09-14", workers=2
    )
    assert len(reports.calls) == 3
    assert all(c["dimensions"] == "ageGroup,gender" for c in reports.calls)
    assert out == {
        "a": {
            "viewerPercentage: ageGroup": "age25-34:100.0",
            "viewerPercentage: gender": "female:60.0;male:40.0",
        }
    }
    assert failed == ["c"]


def test_merge_demographics_attaches_cells_to_the_right_record():
    from rainrag.youtube_analytics import merge_demographics

    records = [{"video": "a", "views": 1}, {"video": "b", "views": 2}]
    merged = merge_demographics(records, {"b": {"viewerPercentage: gender": "male:100.0"}})
    assert "viewerPercentage: gender" not in merged[0]
    assert merged[1]["viewerPercentage: gender"] == "male:100.0"


def test_load_audience_merges_age_and_gender_from_the_newest_snapshot(tmp_path: Path):
    from rainrag.library_performance import load_audience

    p = tmp_path / "m.csv"
    p.write_text(
        "youtube_id,snapshot_date,viewerPercentage: ageGroup,viewerPercentage: gender,views\n"
        "a,2026-09-10,age25-34:100.0,female:100.0,5\n"
        "a,2026-09-14,age35-44:100.0,,9\n"
        "b,2026-09-14,,,3\n",
        encoding="utf-8",
    )
    audience = load_audience(p)
    # Newest age cell wins; the gender cell survives from the older snapshot.
    assert audience["a"] == {"age35-44": 100.0, "female": 100.0}
    assert "b" not in audience


def test_an_expired_refresh_token_is_explained_not_dumped(tmp_path: Path, monkeypatch):
    """A nightly timer hitting Testing-mode expiry must say what to do."""
    import sys
    import types

    import pytest

    from rainrag.youtube_analytics import load_credentials

    class _RefreshError(Exception):
        pass

    class _Creds:
        valid = False
        expired = True
        refresh_token = "r"

        @classmethod
        def from_authorized_user_file(cls, path, scopes):
            return cls()

        def refresh(self, request):
            raise _RefreshError("invalid_grant: Token has been expired or revoked.")

    exc_mod = types.ModuleType("google.auth.exceptions")
    exc_mod.RefreshError = _RefreshError  # type: ignore[attr-defined]
    cred_mod = types.ModuleType("google.oauth2.credentials")
    cred_mod.Credentials = _Creds  # type: ignore[attr-defined]
    req_mod = types.ModuleType("google.auth.transport.requests")
    req_mod.Request = object  # type: ignore[attr-defined]
    flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    flow_mod.InstalledAppFlow = object  # type: ignore[attr-defined]
    for name, mod in (
        ("google.auth.exceptions", exc_mod),
        ("google.oauth2.credentials", cred_mod),
        ("google.auth.transport.requests", req_mod),
        ("google_auth_oauthlib.flow", flow_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="consent has expired"):
        load_credentials(tmp_path / "client.json", token)


def test_demographics_retries_once_on_a_rate_limit_then_gives_up(monkeypatch):
    from rainrag.youtube_analytics import fetch_video_demographics

    monkeypatch.setattr("rainrag.youtube_analytics.time.sleep", lambda s: None)
    headers = [{"name": "ageGroup"}, {"name": "gender"}, {"name": "viewerPercentage"}]
    ok = {"columnHeaders": headers, "rows": [["age18-24", "male", 100.0]]}
    reports = _FakeReports(
        {
            "recovers": [_FakeHttpError(429), ok],
            "keeps_failing": [_FakeHttpError(503), _FakeHttpError(503)],
        }
    )
    _stub_discovery(monkeypatch, reports)
    out, failed = fetch_video_demographics(
        object(), ["recovers", "keeps_failing"], "2015-01-01", "2026-09-14", workers=1
    )
    assert "recovers" in out and failed == ["keeps_failing"]
    assert len([c for c in reports.calls if "recovers" in c["filters"]]) == 2
    assert len([c for c in reports.calls if "keeps_failing" in c["filters"]]) == 2


def test_a_socket_timeout_marks_one_video_failed_instead_of_aborting_the_pull(monkeypatch):
    """The script sets a socket timeout; a timeout is not an HttpError."""
    from rainrag.youtube_analytics import fetch_video_demographics

    monkeypatch.setattr("rainrag.youtube_analytics.time.sleep", lambda s: None)
    headers = [{"name": "ageGroup"}, {"name": "gender"}, {"name": "viewerPercentage"}]
    ok = {"columnHeaders": headers, "rows": [["age18-24", "male", 100.0]]}
    reports = _FakeReports(
        {
            "hangs": [TimeoutError("timed out"), TimeoutError("timed out")],
            "flaky_then_ok": [TimeoutError("timed out"), ok],
            "fine": ok,
        }
    )
    _stub_discovery(monkeypatch, reports)
    out, failed = fetch_video_demographics(
        object(), ["hangs", "flaky_then_ok", "fine"], "2015-01-01", "2026-09-14", workers=3
    )
    assert failed == ["hangs"]
    assert set(out) == {"flaky_then_ok", "fine"}


def test_metrics_batch_retries_a_timeout_then_succeeds(monkeypatch):
    """The process-wide socket timeout must not turn one slow batch into a lost night."""
    from rainrag.youtube_analytics import fetch_video_metrics

    monkeypatch.setattr("rainrag.youtube_analytics.time.sleep", lambda s: None)
    headers = [{"name": "video"}, {"name": "views"}]
    ok = {"columnHeaders": headers, "rows": [["a", 10], ["b", 20]]}
    reports = _FakeReports({"a,b": [TimeoutError("timed out"), TimeoutError("timed out"), ok]})
    _stub_discovery(monkeypatch, reports)
    records = fetch_video_metrics(object(), ["a", "b"], "2015-01-01", "2026-09-14")
    assert [r["video"] for r in records] == ["a", "b"]
    assert len(reports.calls) == 3


def test_metrics_batch_gives_up_after_three_transport_failures(monkeypatch):
    import pytest

    from rainrag.youtube_analytics import fetch_video_metrics

    monkeypatch.setattr("rainrag.youtube_analytics.time.sleep", lambda s: None)
    reports = _FakeReports(
        {"a": [TimeoutError("t"), TimeoutError("t"), TimeoutError("t"), TimeoutError("t")]}
    )
    _stub_discovery(monkeypatch, reports)
    with pytest.raises(TimeoutError):
        fetch_video_metrics(object(), ["a"], "2015-01-01", "2026-09-14")
    assert len(reports.calls) == 3


def test_the_consent_url_exit_is_distinguishable_from_a_failure(tmp_path: Path, monkeypatch):
    """The pull script must not alert on the --auth run stopping as designed."""
    import pytest

    from rainrag.youtube_analytics import ConsentRequired, load_credentials

    _stub_google(monkeypatch)
    client = tmp_path / "client.json"
    client.write_text("{}", encoding="utf-8")
    with pytest.raises(ConsentRequired) as info:
        load_credentials(client, tmp_path / "token.json", auth_code=None)
    assert isinstance(info.value, SystemExit)
    assert info.value.code not in (0, None)
    assert "accounts.google.com" in str(info.value)


def test_metrics_batch_retries_a_rate_limit_and_still_drops_a_rejected_metric(monkeypatch):
    """A 429 on the batch is transient and gets the bounded retry; a rejected metric is not."""
    from rainrag.youtube_analytics import fetch_video_metrics

    monkeypatch.setattr("rainrag.youtube_analytics.time.sleep", lambda s: None)
    headers = [{"name": "video"}, {"name": "views"}]
    ok = {"columnHeaders": headers, "rows": [["a", 10]]}
    reports = _FakeReports({"a": [_FakeHttpError(429), _FakeHttpError(503), ok]})
    _stub_discovery(monkeypatch, reports)
    records = fetch_video_metrics(object(), ["a"], "2015-01-01", "2026-09-14")
    assert [r["video"] for r in records] == ["a"]
    assert len(reports.calls) == 3


def test_metrics_batch_gives_up_after_three_rate_limits(monkeypatch):
    import pytest

    from rainrag.youtube_analytics import fetch_video_metrics

    monkeypatch.setattr("rainrag.youtube_analytics.time.sleep", lambda s: None)
    reports = _FakeReports({"a": [_FakeHttpError(429)] * 4})
    _stub_discovery(monkeypatch, reports)
    with pytest.raises(_FakeHttpError):
        fetch_video_metrics(object(), ["a"], "2015-01-01", "2026-09-14")
    assert len(reports.calls) == 3


def test_metrics_batch_still_raises_immediately_on_a_non_transient_http_error(monkeypatch):
    """403 forbidden is the wrong channel, not a blip; retrying it hides the cause."""
    import pytest

    from rainrag.youtube_analytics import fetch_video_metrics

    reports = _FakeReports({"a": [_FakeHttpError(403)] * 3})
    _stub_discovery(monkeypatch, reports)
    with pytest.raises(_FakeHttpError):
        fetch_video_metrics(object(), ["a"], "2015-01-01", "2026-09-14")
    assert len(reports.calls) == 1


def test_marginals_skip_an_unavailable_cell_rather_than_raising():
    """YouTube renders an unavailable value as "-"."""
    from rainrag.youtube_analytics import marginals

    rows = [
        {"ageGroup": "age25-34", "gender": "female", "viewerPercentage": "-"},
        {"ageGroup": "age25-34", "gender": "male", "viewerPercentage": 40.0},
    ]
    assert marginals(rows) == ({"age25-34": 40.0}, {"male": 40.0})


def test_a_malformed_response_marks_that_video_failed_and_the_pull_continues(monkeypatch):
    """Shaping used to run outside the worker's guard and abort the pool."""
    from rainrag.youtube_analytics import fetch_video_demographics

    monkeypatch.setattr("rainrag.youtube_analytics.time.sleep", lambda s: None)
    headers = [{"name": "ageGroup"}, {"name": "gender"}, {"name": "viewerPercentage"}]
    ok = {"columnHeaders": headers, "rows": [["age18-24", "male", 100.0]]}
    reports = _FakeReports(
        {
            "garbled": {"columnHeaders": "not-a-list", "rows": [["x"]]},
            "dashes": {"columnHeaders": headers, "rows": [["age18-24", "male", "-"]]},
            "fine": ok,
        }
    )
    _stub_discovery(monkeypatch, reports)
    out, failed = fetch_video_demographics(
        object(), ["garbled", "dashes", "fine"], "2015-01-01", "2026-09-14", workers=3
    )
    assert "fine" in out
    assert "dashes" not in out  # every cell unavailable reads as withheld, not failed
    assert failed == ["garbled"]
