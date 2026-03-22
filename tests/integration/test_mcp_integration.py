"""Integration tests for MCP server.

These tests verify the MCP server works end-to-end with actual components.
They require Qdrant to be running and are marked as integration tests.
"""

import os
import re
import subprocess
import time
from pathlib import Path
from shutil import which

import pytest
import requests


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = str(_REPO_ROOT / "src")


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC_PATH if not existing else f"{_SRC_PATH}{os.pathsep}{existing}"
    return env


def _rainrag_mcp_cmd(*args: str) -> list[str]:
    return ["uv", "run", "python", "-m", "rainrag.cli", "mcp", *args]


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


@pytest.mark.integration
class TestMCPServerHTTPIntegration:
    """Integration tests for MCP server with HTTP transport."""

    @pytest.fixture
    def config_file(self, tmp_path: Path) -> Path:
        """Create a minimal config file for testing."""
        config_content = """
paths:
  archive_root: "./tests/fixtures/vtt_files"
  docs_output: "./data/test_docs.jsonl"
  embeddings_cache: "./embeddings"

embedding:
  provider: "mistral"
  model_name: "intfloat/multilingual-e5-large"

qdrant:
  host: "localhost"
  port: 6333
  collection_name: "test_mcp_collection"
  vector_size: 1024

llm:
  provider: "mistral"

mistral:
  api_key: "test-key-placeholder"
  model_name: "mistral-small-latest"
  max_tokens: 512
  temperature: 0.3
  top_k: 3

openai:
  api_key: "test-key-placeholder"

processing:
  num_workers: 2

logging:
  level: "INFO"
  log_file: "./logs/test_mcp.log"

mcp:
  transport: "streamable-http"
  host: "localhost"
  port: 8888
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)
        return config_path

    @pytest.mark.skipif(
        subprocess.run(
            ["curl", "-f", "http://localhost:6333/readyz"],
            capture_output=True,
            timeout=2,
        ).returncode
        != 0,
        reason="Qdrant not running on localhost:6333",
    )
    def test_mcp_server_startup_http(self, config_file: Path) -> None:
        """Test that MCP server starts successfully with HTTP transport.

        This test verifies:
        1. Server can be started with a config file
        2. Server becomes accessible via HTTP
        3. Server can be gracefully shutdown

        Note: Requires Qdrant to be running locally.
        """
        # Start the MCP server as a subprocess
        process = subprocess.Popen(
            _rainrag_mcp_cmd(
                "--config",
                str(config_file),
                "--transport",
                "streamable-http",
                "--port",
                "8888",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_subprocess_env(),
        )

        try:
            # Wait for server to start (max 60 seconds)
            server_started = False
            for _ in range(120):
                # Check if server process has terminated
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1)
                    raise AssertionError(
                        "MCP server exited before becoming ready.\n"
                        + f"stdout:\n{_strip_ansi(stdout)}\n"
                        + f"stderr:\n{_strip_ansi(stderr)}"
                    )
                try:
                    # Try to connect to the MCP endpoint
                    response = requests.get("http://localhost:8888/mcp", timeout=1)
                    # Any response means server is up. Streamable HTTP commonly returns 406 for plain GET
                    # requests without `Accept: text/event-stream`.
                    if response.status_code in [200, 405, 404, 406]:
                        server_started = True
                        break
                except requests.exceptions.RequestException:
                    time.sleep(0.5)

            assert server_started, "MCP server failed to start within 60 seconds"

            # Give it a moment to fully initialize
            time.sleep(2)

            # Verify server is still running
            assert process.poll() is None, "MCP server terminated unexpectedly"

        finally:
            # Clean up: terminate the server
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    @pytest.mark.skipif(
        subprocess.run(
            ["curl", "-f", "http://localhost:6333/readyz"],
            capture_output=True,
            timeout=2,
        ).returncode
        != 0,
        reason="Qdrant not running on localhost:6333",
    )
    def test_mcp_server_health_check(self, config_file: Path) -> None:
        """Test MCP server health and basic connectivity.

        This test verifies the server responds to HTTP requests.
        """
        process = subprocess.Popen(
            _rainrag_mcp_cmd(
                "--config",
                str(config_file),
                "--transport",
                "streamable-http",
                "--port",
                "8889",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_subprocess_env(),
        )

        try:
            # Wait for server startup
            server_ready = False
            for _ in range(120):
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1)
                    raise AssertionError(
                        "MCP server exited before becoming ready.\n"
                        + f"stdout:\n{_strip_ansi(stdout)}\n"
                        + f"stderr:\n{_strip_ansi(stderr)}"
                    )
                try:
                    response = requests.get("http://localhost:8889/mcp", timeout=1)
                    if response.status_code in [200, 405, 404, 406]:
                        server_ready = True
                        break
                except requests.exceptions.RequestException:
                    time.sleep(0.5)

            assert server_ready, "Server did not become ready within 60 seconds"

            # Test that we can consistently connect
            for _ in range(3):
                try:
                    response = requests.get("http://localhost:8889/mcp", timeout=2)
                    # Server should respond (even if method not allowed)
                    assert response.status_code in [
                        200,
                        405,
                        404,
                        406,
                    ], f"Unexpected status code: {response.status_code}"
                except requests.exceptions.RequestException as e:
                    pytest.fail(f"Failed to connect to server: {e}")

        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.integration
class TestMCPServerCommandLine:
    """Integration tests for MCP server CLI commands."""

    def test_mcp_help_command(self) -> None:
        """Test that mcp command help works."""
        result = subprocess.run(
            _rainrag_mcp_cmd("--help"),
            capture_output=True,
            text=True,
            timeout=10,
            env=_subprocess_env(),
        )

        assert result.returncode == 0, "Help command should succeed"
        out = _strip_ansi(result.stdout)
        assert "MCP (Model Context Protocol)" in out
        assert "--transport" in out
        assert "--port" in out
        assert "--host" in out

    def test_mcp_invalid_transport(self, tmp_path: Path) -> None:
        """Test that invalid transport option is handled."""
        # Create minimal config
        config = tmp_path / "config.yaml"
        config.write_text(
            """
paths:
  archive_root: "."
  docs_output: "."
  embeddings_cache: "."
embedding:
  provider: "local"
qdrant:
  host: "localhost"
llm:
  provider: "mistral"
mistral:
  api_key: "test"
openai:
  api_key: "test"
processing: {}
logging: {}
"""
        )

        # Try to run with invalid transport (should fail during initialization)
        # Note: The actual validation might happen at different points
        result = subprocess.run(
            _rainrag_mcp_cmd(
                "--config",
                str(config),
                "--transport",
                "invalid-transport",
            ),
            capture_output=True,
            text=True,
            timeout=10,
            env=_subprocess_env(),
        )

        # Server should either fail immediately or after trying to start
        # We mainly verify it doesn't hang indefinitely
        assert result.returncode != 0 or "invalid" in result.stderr.lower()


@pytest.mark.integration
@pytest.mark.slow
class TestMCPServerWithInspector:
    """Tests for MCP server using the MCP Inspector tool.

    These tests require the MCP Inspector to be installed:
    npm install -g @modelcontextprotocol/inspector
    """

    def test_mcp_inspector_connection(self, tmp_path: Path) -> None:
        """Smoke test that MCP Inspector CLI is available.

        If the inspector is installed, verify the CLI responds to --help.
        """
        if which("mcp-inspector") is None:
            pytest.skip("MCP Inspector CLI not found in PATH")

        result = subprocess.run(
            ["mcp-inspector", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, "MCP Inspector CLI did not run successfully"
