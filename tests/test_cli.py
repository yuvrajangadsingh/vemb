from click.testing import CliRunner
from vemb.cli import cli


def test_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.3.0" in result.output


def test_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "httpie for embeddings" in result.output


def test_text_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["text", "hello"])
    assert result.exit_code != 0


def test_image_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["image", "/nonexistent/file.jpg"])
    assert result.exit_code != 0
