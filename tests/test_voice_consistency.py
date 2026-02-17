#!/usr/bin/env python3
"""
Tests for voice consistency fixes.

Covers:
- AssetManager voice config: "narration" key exists and maps to narrator voice
- get_voice_for_role: gender=None defaults to "male"
- get_voice_for_role: dialogue without speaker_name returns narrator (not random)
- get_voice_for_role: same speaker always returns the same voice (hash-based)
- get_voice_for_role: narration type always returns the same voice config
"""

import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.asset_manager import AssetManager


@pytest.fixture
def manager():
    """Create an AssetManager for testing (asset files may not exist)."""
    return AssetManager(asset_dir="./assets")


# ---------------------------------------------------------------------------
# Narration key exists in voices dict
# ---------------------------------------------------------------------------

class TestNarrationVoiceKey:
    def test_narration_key_exists(self, manager):
        """The voices dict should have an explicit 'narration' key."""
        assert "narration" in manager.voices

    def test_narration_uses_narrator_audio(self, manager):
        """'narration' voice should use the same audio file as 'narrator'."""
        assert manager.voices["narration"]["audio"] == manager.voices["narrator"]["audio"]

    def test_narration_voice_for_role(self, manager):
        """get_voice_for_role('narration') should return a voice with narrator audio."""
        voice = manager.get_voice_for_role("narration")
        assert voice["audio"] == manager.voices["narrator"]["audio"]


# ---------------------------------------------------------------------------
# Pure narrator mode: all narration chunks get the same voice
# ---------------------------------------------------------------------------

class TestPureNarratorVoiceConsistency:
    def test_all_narration_chunks_same_voice(self, manager):
        """Calling get_voice_for_role('narration') 100 times should always
        return the same voice config (same object or same values)."""
        voices = [
            manager.get_voice_for_role("narration", "narrator", "male")
            for _ in range(100)
        ]
        first = voices[0]
        for v in voices[1:]:
            assert v["audio"] == first["audio"]
            assert v["text"] == first["text"]

    def test_narration_returns_narrator_wav(self, manager):
        """In pure narrator mode, voice should point to narrator.wav."""
        voice = manager.get_voice_for_role("narration", "narrator", "male")
        assert "narrator.wav" in voice["audio"]


# ---------------------------------------------------------------------------
# Gender=None handling
# ---------------------------------------------------------------------------

class TestGenderNoneHandling:
    def test_gender_none_uses_male_pool(self, manager):
        """When gender=None, dialogue speaker should get a male voice."""
        voice = manager.get_voice_for_role("dialogue", "test_speaker_m", None)
        # Should have been assigned from male_pool
        assert voice in manager.voices["male_pool"]

    def test_gender_none_does_not_pick_female(self, manager):
        """When gender=None, dialogue speaker should NOT get a female voice."""
        voice = manager.get_voice_for_role("dialogue", "test_speaker_n", None)
        assert voice not in manager.voices["female_pool"]


# ---------------------------------------------------------------------------
# Dialogue speaker voice consistency (hash-based)
# ---------------------------------------------------------------------------

class TestDialogueVoiceConsistency:
    def test_same_speaker_same_voice(self, manager):
        """Same speaker name should always return the same voice config."""
        v1 = manager.get_voice_for_role("dialogue", "老渔夫", "male")
        v2 = manager.get_voice_for_role("dialogue", "老渔夫", "male")
        assert v1 is v2

    def test_same_speaker_100_calls(self, manager):
        """100 calls for the same speaker should all return the same voice."""
        voices = [
            manager.get_voice_for_role("dialogue", "张三", "male")
            for _ in range(100)
        ]
        first = voices[0]
        for v in voices[1:]:
            assert v["audio"] == first["audio"]

    def test_different_speakers_can_differ(self, manager):
        """Different speaker names may get different voices."""
        # Just verify they are both valid, not necessarily different
        v1 = manager.get_voice_for_role("dialogue", "speaker_a", "male")
        v2 = manager.get_voice_for_role("dialogue", "speaker_b", "male")
        assert "audio" in v1
        assert "audio" in v2


# ---------------------------------------------------------------------------
# Dialogue without speaker name returns narrator (not random)
# ---------------------------------------------------------------------------

class TestDialogueWithoutSpeaker:
    def test_no_speaker_returns_narrator(self, manager):
        """Dialogue without speaker_name should return narrator voice."""
        voice = manager.get_voice_for_role("dialogue", None, "male")
        assert voice["audio"] == manager.voices["narrator"]["audio"]

    def test_no_speaker_consistent(self, manager):
        """Multiple calls without speaker_name should return the same voice."""
        voices = [
            manager.get_voice_for_role("dialogue", None, "male")
            for _ in range(50)
        ]
        first = voices[0]
        for v in voices[1:]:
            assert v["audio"] == first["audio"]
            assert v["text"] == first["text"]


# ---------------------------------------------------------------------------
# Non-dialogue types consistency
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gender fallback: unknown gender uses male pool (Bug fix)
# ---------------------------------------------------------------------------

class TestGenderUnknownFallback:
    def test_unknown_gender_uses_male_pool(self, manager):
        """When gender='unknown', dialogue speaker should get a male voice (not female)."""
        voice = manager.get_voice_for_role("dialogue", "路人甲_unknown", "unknown")
        assert voice in manager.voices["male_pool"]

    def test_unknown_gender_does_not_pick_female(self, manager):
        """When gender='unknown', dialogue speaker should NOT get a female voice."""
        voice = manager.get_voice_for_role("dialogue", "路人乙_unknown", "unknown")
        assert voice not in manager.voices["female_pool"]

    def test_female_gender_still_uses_female_pool(self, manager):
        """When gender='female', dialogue speaker should get a female voice."""
        voice = manager.get_voice_for_role("dialogue", "艾米莉_test", "female")
        assert voice in manager.voices["female_pool"]

    def test_male_gender_uses_male_pool(self, manager):
        """When gender='male', dialogue speaker should get a male voice."""
        voice = manager.get_voice_for_role("dialogue", "老渔夫_test_m", "male")
        assert voice in manager.voices["male_pool"]


# ---------------------------------------------------------------------------
# Custom .wav voice binding for named characters
# ---------------------------------------------------------------------------

class TestCustomWavVoiceBinding:
    def test_custom_wav_binds_to_speaker(self, tmp_path):
        """If assets/voices/<speaker>.wav exists, it should be used."""
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        custom_wav = voices_dir / "老渔夫.wav"
        custom_wav.write_bytes(b"RIFF" + b"\x00" * 40)
        # Create narrator.wav stub for AssetManager init
        narrator_wav = voices_dir / "narrator.wav"
        narrator_wav.write_bytes(b"RIFF" + b"\x00" * 40)

        mgr = AssetManager(asset_dir=str(tmp_path))
        voice = mgr.get_voice_for_role("dialogue", "老渔夫", "male")
        assert voice["audio"] == str(custom_wav)

    def test_no_custom_wav_uses_pool(self, manager):
        """Without a custom .wav file, speaker uses pool-based assignment."""
        voice = manager.get_voice_for_role("dialogue", "张三_no_wav", "male")
        assert voice in manager.voices["male_pool"]


# ---------------------------------------------------------------------------
# Non-dialogue types consistency
# ---------------------------------------------------------------------------

class TestNonDialogueTypes:
    def test_title_consistent(self, manager):
        v1 = manager.get_voice_for_role("title")
        v2 = manager.get_voice_for_role("title")
        assert v1["audio"] == v2["audio"]

    def test_subtitle_consistent(self, manager):
        v1 = manager.get_voice_for_role("subtitle")
        v2 = manager.get_voice_for_role("subtitle")
        assert v1["audio"] == v2["audio"]

    def test_recap_consistent(self, manager):
        v1 = manager.get_voice_for_role("recap")
        v2 = manager.get_voice_for_role("recap")
        assert v1["audio"] == v2["audio"]
