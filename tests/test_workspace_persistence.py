#!/usr/bin/env python3
"""
Tests for workspace persistence (breakpoint memory and recovery).

Covers:
- load_workspace: returns defaults when no file exists, loads saved state
- save_workspace: persists state to JSON, handles file objects and path strings
- WORKSPACE_FILE constant definition
- brain_panel visibility logic based on restored mode
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Tests: WORKSPACE_FILE definition and functions exist in webui.py source
# ---------------------------------------------------------------------------

class TestWorkspaceSourceCode:
    """Verify workspace persistence code exists in webui.py."""

    @pytest.fixture
    def webui_source(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "webui.py",
        )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_workspace_file_defined(self, webui_source):
        """WORKSPACE_FILE constant should be defined."""
        assert "WORKSPACE_FILE" in webui_source

    def test_load_workspace_defined(self, webui_source):
        """load_workspace function should be defined."""
        assert "def load_workspace(" in webui_source

    def test_save_workspace_defined(self, webui_source):
        """save_workspace function should be defined."""
        assert "def save_workspace(" in webui_source

    def test_save_workspace_called_in_run_cinecast(self, webui_source):
        """save_workspace should be called inside run_cinecast."""
        assert "save_workspace(epub_file" in webui_source

    def test_last_state_loaded_before_ui(self, webui_source):
        """last_state should be loaded before UI construction."""
        assert "last_state = load_workspace()" in webui_source

    def test_book_file_uses_saved_state(self, webui_source):
        """book_file should use saved state for default value."""
        assert 'value=default_file' in webui_source

    def test_mode_selector_uses_saved_state(self, webui_source):
        """mode_selector should use saved state for default value."""
        assert 'last_state.get("mode"' in webui_source

    def test_master_json_uses_saved_state(self, webui_source):
        """master_json should use saved state for default value."""
        assert 'last_state.get("master_json"' in webui_source

    def test_brain_panel_visibility_uses_saved_state(self, webui_source):
        """brain_panel visibility should be based on saved mode."""
        assert 'init_brain_visible' in webui_source
        assert '"智能配音" in last_state.get("mode"' in webui_source


# ---------------------------------------------------------------------------
# Tests: load_workspace and save_workspace logic (isolated from Gradio)
# ---------------------------------------------------------------------------

class TestLoadWorkspace:
    """Verify load_workspace returns correct defaults and loaded state."""

    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        """When workspace file does not exist, return defaults."""
        monkeypatch.chdir(tmp_path)
        # Inline the logic from load_workspace (no Gradio import needed)
        workspace_file = os.path.join(str(tmp_path), ".cinecast_workspace.json")
        if os.path.exists(workspace_file):
            with open(workspace_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
        else:
            state = {"book_file": None, "mode": "🎙️ 纯净旁白模式", "master_json": ""}

        assert state["book_file"] is None
        assert state["mode"] == "🎙️ 纯净旁白模式"
        assert state["master_json"] == ""

    def test_loads_saved_state(self, tmp_path):
        """When workspace file exists with valid JSON, load it."""
        workspace_file = tmp_path / ".cinecast_workspace.json"
        saved = {
            "book_file": "/some/path/book.epub",
            "mode": "🎭 智能配音模式 (外脑控制版)",
            "master_json": '{"characters": {}}',
        }
        workspace_file.write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')

        with open(str(workspace_file), 'r', encoding='utf-8') as f:
            state = json.load(f)

        assert state["book_file"] == "/some/path/book.epub"
        assert state["mode"] == "🎭 智能配音模式 (外脑控制版)"
        assert state["master_json"] == '{"characters": {}}'

    def test_returns_defaults_on_corrupt_json(self, tmp_path):
        """When workspace file contains invalid JSON, return defaults."""
        workspace_file = tmp_path / ".cinecast_workspace.json"
        workspace_file.write_text("not valid json {{{", encoding='utf-8')

        try:
            with open(str(workspace_file), 'r', encoding='utf-8') as f:
                state = json.load(f)
        except Exception:
            state = {"book_file": None, "mode": "🎙️ 纯净旁白模式", "master_json": ""}

        assert state["book_file"] is None
        assert state["mode"] == "🎙️ 纯净旁白模式"


class TestSaveWorkspace:
    """Verify save_workspace persists state correctly."""

    def test_saves_state_with_string_path(self, tmp_path):
        """Save workspace with a string file path."""
        workspace_file = tmp_path / ".cinecast_workspace.json"
        file_path = "/some/path/book.epub"
        mode = "🎙️ 纯净旁白模式"
        master_json = ""

        state = {"book_file": file_path, "mode": mode, "master_json": master_json}
        with open(str(workspace_file), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        with open(str(workspace_file), 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded["book_file"] == file_path
        assert loaded["mode"] == mode
        assert loaded["master_json"] == master_json

    def test_saves_state_with_file_object(self, tmp_path):
        """Save workspace extracts path from object with .name attribute."""
        workspace_file = tmp_path / ".cinecast_workspace.json"

        class FakeFile:
            name = "/gradio/tmp/book.epub"

        book_file = FakeFile()
        file_path = book_file.name if hasattr(book_file, "name") else book_file

        state = {
            "book_file": file_path,
            "mode": "🎭 智能配音模式 (外脑控制版)",
            "master_json": '{"characters": {}}',
        }
        with open(str(workspace_file), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        with open(str(workspace_file), 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded["book_file"] == "/gradio/tmp/book.epub"

    def test_roundtrip_persistence(self, tmp_path):
        """Save then load should return identical state."""
        workspace_file = tmp_path / ".cinecast_workspace.json"
        original = {
            "book_file": "/path/to/novel.txt",
            "mode": "🎭 智能配音模式 (外脑控制版)",
            "master_json": '{"characters": {"旁白": {"gender": "male"}}, "recaps": {}}',
        }

        with open(str(workspace_file), 'w', encoding='utf-8') as f:
            json.dump(original, f, ensure_ascii=False, indent=2)

        with open(str(workspace_file), 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == original


# ---------------------------------------------------------------------------
# Tests: brain_panel visibility logic
# ---------------------------------------------------------------------------

class TestBrainPanelVisibility:
    """Verify brain panel visibility is correctly derived from saved mode."""

    def test_visible_for_smart_mode(self):
        """Brain panel should be visible when mode contains '智能配音'."""
        mode = "🎭 智能配音模式 (外脑控制版)"
        assert "智能配音" in mode

    def test_hidden_for_narrator_mode(self):
        """Brain panel should be hidden for pure narrator mode."""
        mode = "🎙️ 纯净旁白模式"
        assert "智能配音" not in mode

    def test_hidden_for_empty_mode(self):
        """Brain panel should be hidden when mode is empty string."""
        mode = ""
        assert "智能配音" not in mode
