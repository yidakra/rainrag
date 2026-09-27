def test_browser_media_urls_are_same_origin_when_no_asset_host_is_set(monkeypatch):
    """Tenki on #88: they used to inherit the server-side API base.

    Once the Streamlit units pointed RAINRAG_API_URL at 127.0.0.1:8001, every
    <video> src became the reader's own loopback and playback broke with no
    error anywhere. A relative path is correct behind any proxy.
    """
    import importlib

    # Set empty rather than delete: app.py calls load_dotenv() at import, and
    # a deleted key is repopulated from the deployment's .env, while a present
    # one is left alone.
    monkeypatch.setenv("RAINRAG_ASSET_URL", "")
    monkeypatch.setenv("RAINRAG_API_URL", "http://127.0.0.1:8001")
    app = importlib.reload(importlib.import_module("app"))
    try:
        assert app.ASSET_BASE_URL == ""
        assert app.build_asset_url("/video/x.mp4") == "/video/x.mp4"
        assert app.build_asset_url("video/x.mp4") == "/video/x.mp4"
        # An absolute URL is still passed through untouched.
        assert app.build_asset_url("https://cdn.example/x.mp4") == "https://cdn.example/x.mp4"
    finally:
        # Undo before the reload: monkeypatch teardown runs after the test
        # returns, so reloading first bakes the test's env into the cached
        # module and the next importer inherits it (Tenki on #88).
        monkeypatch.undo()
        importlib.reload(app)


def test_an_explicit_asset_host_is_still_honoured(monkeypatch):
    import importlib

    monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.tvrain.io/")
    app = importlib.reload(importlib.import_module("app"))
    try:
        assert app.build_asset_url("/video/x.mp4") == "https://rag.tvrain.io/video/x.mp4"
    finally:
        # Undo before the reload: monkeypatch teardown runs after the test
        # returns, so reloading first bakes the test's env into the cached
        # module and the next importer inherits it (Tenki on #88).
        monkeypatch.undo()
        importlib.reload(app)


def test_server_side_fetches_get_an_absolute_url(monkeypatch):
    """Tenki on #88: requests rejects a relative URL with MissingSchema, and a
    broad except turned that into a silent None, so subtitles stopped loading."""
    import importlib

    monkeypatch.setenv("RAINRAG_ASSET_URL", "")
    monkeypatch.setenv("RAINRAG_API_URL", "http://127.0.0.1:8001")
    app = importlib.reload(importlib.import_module("app"))
    try:
        assert app.build_api_url("/vtt/ab/x.ru.vtt") == "http://127.0.0.1:8001/vtt/ab/x.ru.vtt"
        assert app.build_api_url("vtt/ab/x.ru.vtt") == "http://127.0.0.1:8001/vtt/ab/x.ru.vtt"
        assert app.build_api_url("https://cdn.example/x.vtt") == "https://cdn.example/x.vtt"
        # The browser-facing builder stays relative; the two must not converge.
        assert app.build_asset_url("/vtt/ab/x.ru.vtt") == "/vtt/ab/x.ru.vtt"
    finally:
        # Undo before the reload: monkeypatch teardown runs after the test
        # returns, so reloading first bakes the test's env into the cached
        # module and the next importer inherits it (Tenki on #88).
        monkeypatch.undo()
        importlib.reload(app)


def test_the_vtt_fetch_uses_the_api_builder_not_the_asset_one():
    import inspect

    import app

    assert "build_api_url(vtt_url)" in inspect.getsource(app.fetch_vtt_content)
