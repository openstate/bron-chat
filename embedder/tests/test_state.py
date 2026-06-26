import os

from app.state import read_last_run, write_last_run, EPOCH


def test_read_last_run_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.state.settings.STATE_FILE", str(tmp_path / "nonexistent.txt"))
    assert read_last_run() == EPOCH


def test_read_last_run_empty_file(tmp_path, monkeypatch):
    f = tmp_path / "last_run.txt"
    f.write_text("")
    monkeypatch.setattr("app.state.settings.STATE_FILE", str(f))
    assert read_last_run() == EPOCH


def test_write_and_read_roundtrip(tmp_path, monkeypatch):
    state_file = str(tmp_path / "state" / "last_run.txt")
    monkeypatch.setattr("app.state.settings.STATE_FILE", state_file)
    ts = "2024-06-01T12:00:00+00:00"
    write_last_run(ts)
    assert read_last_run() == ts


def test_write_creates_directories(tmp_path, monkeypatch):
    nested = str(tmp_path / "a" / "b" / "c" / "last_run.txt")
    monkeypatch.setattr("app.state.settings.STATE_FILE", nested)
    write_last_run("2024-01-01T00:00:00")
    assert os.path.exists(nested)


def test_write_never_regresses_to_older_timestamp(tmp_path, monkeypatch):
    state_file = str(tmp_path / "last_run.txt")
    monkeypatch.setattr("app.state.settings.STATE_FILE", state_file)
    write_last_run("2024-06-01T00:00:00")
    write_last_run("2024-01-01T00:00:00")
    assert read_last_run() == "2024-06-01T00:00:00"


def test_write_advances_to_newer_timestamp(tmp_path, monkeypatch):
    state_file = str(tmp_path / "last_run.txt")
    monkeypatch.setattr("app.state.settings.STATE_FILE", state_file)
    write_last_run("2024-01-01T00:00:00")
    write_last_run("2024-06-01T00:00:00")
    assert read_last_run() == "2024-06-01T00:00:00"
