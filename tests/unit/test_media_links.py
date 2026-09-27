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
        importlib.reload(app)


def test_an_explicit_asset_host_is_still_honoured(monkeypatch):
    import importlib

    monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.tvrain.io/")
    app = importlib.reload(importlib.import_module("app"))
    try:
        assert app.build_asset_url("/video/x.mp4") == "https://rag.tvrain.io/video/x.mp4"
    finally:
        importlib.reload(app)
