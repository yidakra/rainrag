"""Integration tests for MCP server.

These tests verify the MCP server works end-to-end with actual components.
They require Qdrant to be running and are marked as integration tests.
"""

import subprocess
import time
from pathlib import Path

import pytest
import requests


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
            [
                "poetry",
                "run",
                "rainrag",
                "mcp",
                "--config",
                str(config_file),
                "--transport",
                "streamable-http",
                "--port",
                "8888",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for server to start (max 15 seconds)
            server_started = False
            for _ in range(30):
                try:
                    # Try to connect to the MCP endpoint
                    response = requests.get("http://localhost:8888/mcp", timeout=1)
                    # Any response (even 405 Method Not Allowed) means server is up
                    if response.status_code in [200, 405, 404]:
                        server_started = True
                        break
                except requests.exceptions.RequestException:
                    time.sleep(0.5)

            assert server_started, "MCP server failed to start within 15 seconds"

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
            [
                "poetry",
                "run",
                "rainrag",
                "mcp",
                "--config",
                str(config_file),
                "--transport",
                "streamable-http",
                "--port",
                "8889",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for server startup
            server_ready = False
            for _ in range(30):
                try:
                    response = requests.get("http://localhost:8889/mcp", timeout=1)
                    if response.status_code in [200, 405, 404]:
                        server_ready = True
                        break
                except requests.exceptions.RequestException:
                    time.sleep(0.5)

            assert server_ready, "Server did not become ready"

            # Test that we can consistently connect
            for _ in range(3):
                try:
                    response = requests.get("http://localhost:8889/mcp", timeout=2)
                    # Server should respond (even if method not allowed)
                    assert response.status_code in [
                        200,
                        405,
                        404,
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
            ["poetry", "run", "rainrag", "mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, "Help command should succeed"
        assert "MCP (Model Context Protocol)" in result.stdout
        assert "--transport" in result.stdout
        assert "--port" in result.stdout
        assert "--host" in result.stdout

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
            [
                "poetry",
                "run",
                "rainrag",
                "mcp",
                "--config",
                str(config),
                "--transport",
                "invalid-transport",
            ],
            capture_output=True,
            text=True,
            timeout=10,
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

    @pytest.mark.skip(reason="Requires MCP Inspector npm package - run manually if needed")
    def test_mcp_inspector_connection(self, tmp_path: Path) -> None:
        """Test MCP server can be inspected with MCP Inspector.

        This is a manual test - run it only if you have the inspector installed.
        """
        # This test is intentionally skipped by default
        # To run it manually:
        # 1. Install inspector: npm install -g @modelcontextprotocol/inspector
        # 2. pytest -v tests/integration/test_mcp_integration.py::TestMCPServerWithInspector::test_mcp_inspector_connection -s
        pass
