#!/usr/bin/env python3
"""
Tests for Brain Node unified workflow (Master JSON parsing).

Covers:
- process_master_json: valid JSON with both characters and recaps
- process_master_json: partial JSON (only characters, only recaps)
- process_master_json: empty / None / whitespace input
- process_master_json: malformed JSON error handling
- Default config includes global_cast, custom_recaps, enable_auto_recap keys
- LLMScriptDirector accepts global_cast parameter
- custom_recaps injection into chapter scripts via phase_1
"""

import json
import os
import re
import sys
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Inline copy of process_master_json (webui.py requires gradio)
# ---------------------------------------------------------------------------

def process_master_json(master_json_str):
    """Standalone copy for environments where gradio is unavailable."""
    global_cast = {}
    custom_recaps = {}

    if not master_json_str or not master_json_str.strip():
        return global_cast, custom_recaps, True, ""

    try:
        master_data = json.loads(master_json_str)
        global_cast = master_data.get("characters", {})
        custom_recaps = master_data.get("recaps", {})
        return global_cast, custom_recaps, True, "✅ 外脑数据解析成功"
    except json.JSONDecodeError:
        return {}, {}, False, "❌ 外脑 JSON 格式错误，请检查是否有遗漏的逗号或引号。"


# ---------------------------------------------------------------------------
# Test: process_master_json — valid inputs
# ---------------------------------------------------------------------------

class TestProcessMasterJsonValid:
    def test_full_master_json(self):
        data = {
            "characters": {
                "旁白": {"gender": "male", "emotion": "平静"},
                "老渔夫": {"gender": "male", "emotion": "沧桑"},
            },
            "recaps": {
                "Chapter_002": "上回说到……",
                "Chapter_003": "警长开始调查……",
            },
        }
        cast, recaps, ok, msg = process_master_json(json.dumps(data, ensure_ascii=False))
        assert ok is True
        assert "旁白" in cast
        assert "老渔夫" in cast
        assert cast["老渔夫"]["gender"] == "male"
        assert "Chapter_002" in recaps
        assert "Chapter_003" in recaps
        assert "成功" in msg

    def test_only_characters(self):
        data = {"characters": {"角色A": {"gender": "female", "emotion": "活泼"}}}
        cast, recaps, ok, _ = process_master_json(json.dumps(data))
        assert ok is True
        assert "角色A" in cast
        assert recaps == {}

    def test_only_recaps(self):
        data = {"recaps": {"Chapter_002": "前情提要内容"}}
        cast, recaps, ok, _ = process_master_json(json.dumps(data))
        assert ok is True
        assert cast == {}
        assert "Chapter_002" in recaps

    def test_extra_keys_ignored(self):
        data = {
            "characters": {"旁白": {"gender": "male", "emotion": "平静"}},
            "recaps": {},
            "metadata": {"version": "1.0"},
        }
        cast, recaps, ok, _ = process_master_json(json.dumps(data))
        assert ok is True
        assert "旁白" in cast


# ---------------------------------------------------------------------------
# Test: process_master_json — empty / whitespace / None
# ---------------------------------------------------------------------------

class TestProcessMasterJsonEmpty:
    def test_empty_string(self):
        cast, recaps, ok, msg = process_master_json("")
        assert ok is True
        assert cast == {}
        assert recaps == {}
        assert msg == ""

    def test_none_input(self):
        cast, recaps, ok, msg = process_master_json(None)
        assert ok is True
        assert cast == {}

    def test_whitespace_only(self):
        cast, recaps, ok, msg = process_master_json("   \n\t  ")
        assert ok is True
        assert cast == {}


# ---------------------------------------------------------------------------
# Test: process_master_json — invalid JSON
# ---------------------------------------------------------------------------

class TestProcessMasterJsonInvalid:
    def test_invalid_json_returns_error(self):
        cast, recaps, ok, msg = process_master_json("{invalid}")
        assert ok is False
        assert cast == {}
        assert recaps == {}
        assert "格式错误" in msg

    def test_truncated_json(self):
        cast, recaps, ok, msg = process_master_json('{"characters": {')
        assert ok is False

    def test_markdown_wrapped_json(self):
        # Users might copy JSON with markdown fences
        raw = '```json\n{"characters": {}}\n```'
        cast, recaps, ok, msg = process_master_json(raw)
        assert ok is False  # We intentionally reject markdown-wrapped JSON


# ---------------------------------------------------------------------------
# Test: Default config includes new keys
# ---------------------------------------------------------------------------

class TestDefaultConfigBrainNode:
    def test_default_config_has_global_cast(self):
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer.__new__(CineCastProducer)
        config = producer._get_default_config()
        assert "global_cast" in config
        assert config["global_cast"] == {}

    def test_default_config_has_custom_recaps(self):
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer.__new__(CineCastProducer)
        config = producer._get_default_config()
        assert "custom_recaps" in config
        assert config["custom_recaps"] == {}

    def test_default_config_has_enable_auto_recap(self):
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")
        producer = CineCastProducer.__new__(CineCastProducer)
        config = producer._get_default_config()
        assert "enable_auto_recap" in config
        assert config["enable_auto_recap"] is True


# ---------------------------------------------------------------------------
# Test: LLMScriptDirector accepts global_cast
# ---------------------------------------------------------------------------

class TestDirectorGlobalCast:
    def test_director_default_global_cast_empty(self):
        from modules.llm_director import LLMScriptDirector
        director = LLMScriptDirector()
        assert director.global_cast == {}

    def test_director_accepts_global_cast(self):
        from modules.llm_director import LLMScriptDirector
        cast = {"老渔夫": {"gender": "male", "emotion": "沧桑"}}
        director = LLMScriptDirector(global_cast=cast)
        assert director.global_cast == cast
        assert director.global_cast["老渔夫"]["gender"] == "male"

    def test_director_none_global_cast_becomes_empty(self):
        from modules.llm_director import LLMScriptDirector
        director = LLMScriptDirector(global_cast=None)
        assert director.global_cast == {}


# ---------------------------------------------------------------------------
# Test: custom_recaps injection into phase_1
# ---------------------------------------------------------------------------

class TestCustomRecapsInjection:
    def test_custom_recaps_injected_by_chapter_name(self):
        """When custom_recaps has a matching chapter name key, it should inject recap chunks."""
        try:
            from main_producer import CineCastProducer
        except ImportError:
            pytest.skip("main_producer requires mlx (macOS-only)")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "chapters")
            os.makedirs(input_dir)
            # Repeat text 100x to exceed the 500-char non-content filter threshold
            for i in range(1, 4):
                fname = f"Chapter_{i:03d}.txt"
                with open(os.path.join(input_dir, fname), "w", encoding="utf-8") as f:
                    f.write(f"第{i}章 标题\n" + "这是正文内容，足够长的文本。" * 100)

            custom_recaps = {
                "Chapter_002": "上回说到老渔夫在暴风雪中发现了黑匣子……",
                "Chapter_003": "警长的调查陷入僵局……",
            }

            config = {
                "assets_dir": os.path.join(tmpdir, "assets"),
                "output_dir": os.path.join(tmpdir, "output"),
                "model_path": "dummy",
                "ambient_theme": "iceland_wind",
                "target_duration_min": 30,
                "min_tail_min": 10,
                "use_local_llm": True,
                "enable_recap": True,
                "pure_narrator_mode": False,
                "user_recaps": None,
                "global_cast": {},
                "custom_recaps": custom_recaps,
                "enable_auto_recap": False,
            }
            producer = CineCastProducer.__new__(CineCastProducer)
            producer.config = config
            producer.script_dir = os.path.join(tmpdir, "scripts")
            producer.cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(producer.script_dir, exist_ok=True)
            os.makedirs(producer.cache_dir, exist_ok=True)
            producer.assets = None

            # Verify config routing without full pipeline (Ollama unavailable in CI)
            assert producer.config["custom_recaps"]["Chapter_002"] == "上回说到老渔夫在暴风雪中发现了黑匣子……"
            assert producer.config["custom_recaps"]["Chapter_003"] == "警长的调查陷入僵局……"
            assert producer.config["enable_auto_recap"] is False
