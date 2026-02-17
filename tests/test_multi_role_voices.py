#!/usr/bin/env python3
"""
Tests for multi-role voice upload in movie dubbing mode.

Covers:
- Default female_pool now has two entries (f1, f2)
- set_custom_role_voices: overrides narrator, m1, m2, f1, f2
- set_custom_role_voices: skips None / missing file paths
- set_custom_role_voices: unknown role names are ignored
- Voice assignment consistency: when more characters than uploaded voices,
  hash-based fallback assigns a deterministic voice that stays fixed across
  the entire book (repeated calls return the same voice).
"""

import os
import sys
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.asset_manager import AssetManager


@pytest.fixture
def manager():
    """Create an AssetManager for testing (asset files may not exist)."""
    return AssetManager(asset_dir="./assets")


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a tiny temporary WAV file for upload simulation."""
    wav = tmp_path / "custom_voice.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)  # minimal stub
    return str(wav)


# ---------------------------------------------------------------------------
# Default pool sizes
# ---------------------------------------------------------------------------

class TestDefaultPoolSizes:
    def test_female_pool_has_two_entries(self, manager):
        """female_pool should now contain f1 and f2."""
        assert len(manager.voices["female_pool"]) == 2

    def test_male_pool_has_two_entries(self, manager):
        """male_pool should still contain m1 and m2."""
        assert len(manager.voices["male_pool"]) == 2

    def test_f2_audio_path(self, manager):
        """f2 entry should point to f2.wav."""
        assert "f2.wav" in manager.voices["female_pool"][1]["audio"]


# ---------------------------------------------------------------------------
# set_custom_role_voices: basic overrides
# ---------------------------------------------------------------------------

class TestSetCustomRoleVoices:
    def test_override_narrator(self, manager, tmp_wav):
        manager.set_custom_role_voices({"narrator": tmp_wav})
        assert manager.voices["narrator"]["audio"] == tmp_wav
        assert manager.voices["narration"]["audio"] == tmp_wav
        assert manager.voices["title"]["audio"] == tmp_wav
        assert manager.voices["subtitle"]["audio"] == tmp_wav

    def test_override_m1(self, manager, tmp_wav):
        manager.set_custom_role_voices({"m1": tmp_wav})
        assert manager.voices["male_pool"][0]["audio"] == tmp_wav

    def test_override_m2(self, manager, tmp_wav):
        manager.set_custom_role_voices({"m2": tmp_wav})
        assert manager.voices["male_pool"][1]["audio"] == tmp_wav

    def test_override_f1(self, manager, tmp_wav):
        manager.set_custom_role_voices({"f1": tmp_wav})
        assert manager.voices["female_pool"][0]["audio"] == tmp_wav

    def test_override_f2(self, manager, tmp_wav):
        manager.set_custom_role_voices({"f2": tmp_wav})
        assert manager.voices["female_pool"][1]["audio"] == tmp_wav

    def test_override_multiple(self, manager, tmp_path):
        """Overriding several roles at once should work."""
        files = {}
        for name in ("narrator", "m1", "f1"):
            p = tmp_path / f"{name}.wav"
            p.write_bytes(b"RIFF" + b"\x00" * 40)
            files[name] = str(p)

        manager.set_custom_role_voices(files)
        assert manager.voices["narrator"]["audio"] == files["narrator"]
        assert manager.voices["male_pool"][0]["audio"] == files["m1"]
        assert manager.voices["female_pool"][0]["audio"] == files["f1"]
        # m2 and f2 should remain at their defaults
        assert "m2.wav" in manager.voices["male_pool"][1]["audio"]
        assert "f2.wav" in manager.voices["female_pool"][1]["audio"]


# ---------------------------------------------------------------------------
# set_custom_role_voices: edge cases
# ---------------------------------------------------------------------------

class TestSetCustomRoleVoicesEdgeCases:
    def test_none_values_skipped(self, manager):
        """None file paths should be silently skipped."""
        original_narrator = manager.voices["narrator"]["audio"]
        manager.set_custom_role_voices({"narrator": None, "m1": None})
        assert manager.voices["narrator"]["audio"] == original_narrator

    def test_nonexistent_file_skipped(self, manager):
        """Non-existent paths should be silently skipped."""
        original_m1 = manager.voices["male_pool"][0]["audio"]
        manager.set_custom_role_voices({"m1": "/nonexistent/path.wav"})
        assert manager.voices["male_pool"][0]["audio"] == original_m1

    def test_unknown_role_ignored(self, manager, tmp_wav):
        """Unknown role names should be ignored without errors."""
        manager.set_custom_role_voices({"unknown_role": tmp_wav})
        # No crash, no changes

    def test_empty_dict(self, manager):
        """Empty dict should be a no-op."""
        original = manager.voices["narrator"]["audio"]
        manager.set_custom_role_voices({})
        assert manager.voices["narrator"]["audio"] == original

    def test_none_input(self, manager):
        """None input should be a no-op."""
        original = manager.voices["narrator"]["audio"]
        manager.set_custom_role_voices(None)
        assert manager.voices["narrator"]["audio"] == original


# ---------------------------------------------------------------------------
# Fallback voice assignment consistency
# ---------------------------------------------------------------------------

class TestFallbackVoiceConsistency:
    """When there are more characters than uploaded voices, the system uses
    hash-based assignment from the pool. This must be deterministic and
    consistent across the entire book (repeated calls return the same voice).
    """

    def test_extra_male_characters_get_consistent_voice(self, manager):
        """Multiple male characters should each get a deterministic voice
        from the male_pool, and the same voice on every call."""
        speakers = ["男角色A", "男角色B", "男角色C", "男角色D", "男角色E"]
        first_pass = {}
        for s in speakers:
            v = manager.get_voice_for_role("dialogue", s, "male")
            first_pass[s] = v["audio"]

        # Second pass: same results
        for s in speakers:
            v = manager.get_voice_for_role("dialogue", s, "male")
            assert v["audio"] == first_pass[s], (
                f"Voice changed for {s}: {first_pass[s]} vs {v['audio']}"
            )

    def test_extra_female_characters_get_consistent_voice(self, manager):
        """Multiple female characters should each get a deterministic voice
        from the female_pool, and the same voice on every call."""
        speakers = ["女角色A", "女角色B", "女角色C", "女角色D"]
        first_pass = {}
        for s in speakers:
            v = manager.get_voice_for_role("dialogue", s, "female")
            first_pass[s] = v["audio"]

        for s in speakers:
            v = manager.get_voice_for_role("dialogue", s, "female")
            assert v["audio"] == first_pass[s]

    def test_voice_from_pool(self, manager):
        """Each assigned voice should come from the correct gender pool."""
        v_m = manager.get_voice_for_role("dialogue", "某男", "male")
        pool_audios = [e["audio"] for e in manager.voices["male_pool"]]
        assert v_m["audio"] in pool_audios

        v_f = manager.get_voice_for_role("dialogue", "某女", "female")
        pool_audios_f = [e["audio"] for e in manager.voices["female_pool"]]
        assert v_f["audio"] in pool_audios_f

    def test_consistency_across_100_calls(self, manager):
        """100 calls for the same speaker should all return the same voice."""
        speaker = "反复测试角色"
        voices = [
            manager.get_voice_for_role("dialogue", speaker, "female")
            for _ in range(100)
        ]
        first = voices[0]
        for v in voices[1:]:
            assert v["audio"] == first["audio"]

    def test_partial_upload_still_consistent(self, manager, tmp_path):
        """Even if only f1 is uploaded (not f2), all female characters
        still get deterministic voices from the pool."""
        p = tmp_path / "f1_custom.wav"
        p.write_bytes(b"RIFF" + b"\x00" * 40)
        manager.set_custom_role_voices({"f1": str(p)})

        speakers = ["女A", "女B", "女C"]
        first_pass = {}
        for s in speakers:
            v = manager.get_voice_for_role("dialogue", s, "female")
            first_pass[s] = v["audio"]

        # Verify consistency
        for s in speakers:
            v = manager.get_voice_for_role("dialogue", s, "female")
            assert v["audio"] == first_pass[s]
