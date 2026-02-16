#!/usr/bin/env python3
"""
CineCast 资产与选角管理器
负责处理音色、声场、音效的加载与智能分配
"""

import os
import json
import random
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

class AssetManager:
    def __init__(self, asset_dir="./assets"):
        self.asset_dir = asset_dir
        self.target_sr = 22050  # Qwen-TTS 标准采样率
        self._initialize_default_voices()
        self._load_voice_config()
        self.role_voice_map = {}  # 记忆已分配角色的音色
        
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """🌟 核心防御：将外部音频归一化为 22050Hz 单声道，杜绝混音时的内存爆炸"""
        return audio.set_frame_rate(self.target_sr).set_channels(1)
    
    def _initialize_default_voices(self):
        """初始化默认音色配置"""
        self.voices = {
            "narrator": {
                "audio": f"{self.asset_dir}/voices/narrator.wav", 
                "text": "沉稳旁白", 
                "speed": 1.0
            },
            # 1.4.1 章节题目：严肃一字一顿，速度调至 0.8
            "title": {
                "audio": f"{self.asset_dir}/voices/narrator.wav", 
                "text": "沉稳旁白", 
                "speed": 0.8
            },
            # 1.4.2 小标题：严肃但比正文慢，速度调至 0.9
            "subtitle": {
                "audio": f"{self.asset_dir}/voices/narrator.wav", 
                "text": "沉稳旁白", 
                "speed": 0.9
            },
            "male_pool": [
                {
                    "audio": f"{self.asset_dir}/voices/m1.wav", 
                    "text": "男声1", 
                    "speed": 1.0
                },
                {
                    "audio": f"{self.asset_dir}/voices/m2.wav", 
                    "text": "男声2", 
                    "speed": 1.05  # 年轻男声加快
                }
            ],
            "female_pool": [
                {
                    "audio": f"{self.asset_dir}/voices/f1.wav", 
                    "text": "女声1", 
                    "speed": 1.0
                }
            ],
            # 新增：前情摘要专属音色 (可稍微加速，带出回顾的紧凑感)
            "recap": {
                "audio": f"{self.asset_dir}/voices/talkover.wav", 
                "text": "前情提要专用声音", 
                "speed": 1.1
            },
        }

    def _load_voice_config(self):
        """从 audio_assets_config.json 加载音色配置，覆盖硬编码的默认值"""
        config_path = os.path.join(self.asset_dir, "..", "audio_assets_config.json")
        if not os.path.exists(config_path):
            # 也尝试项目根目录
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_assets_config.json")
        if not os.path.exists(config_path):
            logger.info("未找到 audio_assets_config.json，使用默认音色配置")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            voice_ref = config.get("voice_reference", {})
            if not voice_ref:
                return

            # 用配置文件中的 acoustic_description 覆盖默认 text 字段
            if "narrator" in voice_ref:
                self.voices["narrator"]["text"] = voice_ref["narrator"].get(
                    "acoustic_description", self.voices["narrator"]["text"]
                )
                self.voices["title"]["text"] = voice_ref["narrator"].get(
                    "acoustic_description", self.voices["title"]["text"]
                )
                self.voices["subtitle"]["text"] = voice_ref["narrator"].get(
                    "acoustic_description", self.voices["subtitle"]["text"]
                )

            if "male_default" in voice_ref and self.voices["male_pool"]:
                self.voices["male_pool"][0]["text"] = voice_ref["male_default"].get(
                    "acoustic_description", self.voices["male_pool"][0]["text"]
                )

            if "young_male" in voice_ref and len(self.voices["male_pool"]) > 1:
                self.voices["male_pool"][1]["text"] = voice_ref["young_male"].get(
                    "acoustic_description", self.voices["male_pool"][1]["text"]
                )

            if "female_default" in voice_ref and self.voices["female_pool"]:
                self.voices["female_pool"][0]["text"] = voice_ref["female_default"].get(
                    "acoustic_description", self.voices["female_pool"][0]["text"]
                )

            # 加载采样率配置
            audio_proc = config.get("audio_processing", {})
            if "target_sample_rate" in audio_proc:
                self.target_sr = audio_proc["target_sample_rate"]

            logger.info("✅ 已从 audio_assets_config.json 加载音色配置")
        except Exception as e:
            logger.warning(f"⚠️ 加载 audio_assets_config.json 失败，使用默认配置: {e}")
    
    def get_voice_for_role(self, role_type, speaker_name=None, gender="male"):
        """
        智能选角逻辑
        
        Args:
            role_type: 角色类型 (title, subtitle, narration, dialogue)
            speaker_name: 说话人姓名 (用于对话角色记忆)
            gender: 性别 (male, female)
        """
        # 处理非对话角色
        if role_type in ["title", "subtitle", "narration", "recap"]:
            return self.voices.get(role_type, self.voices["narrator"])
            
        # 对话角色音色记忆
        if speaker_name and speaker_name not in self.role_voice_map:
            pool = self.voices["male_pool"] if gender == "male" else self.voices["female_pool"]
            # 随机或哈希分配一个音色给新角色
            self.role_voice_map[speaker_name] = random.choice(pool)
            
        if speaker_name:
            return self.role_voice_map.get(speaker_name, self.voices["narrator"])
        else:
            # 如果没有说话人信息，根据性别选择
            pool = self.voices["male_pool"] if gender == "male" else self.voices["female_pool"]
            return random.choice(pool)
    
    def get_ambient_sound(self, theme="default") -> AudioSegment:
        """🌟 防采样率爆炸：支持用户动态上传环境音并强制归一化"""
        # 寻找 assets/ambient 下所有可用的音频
        ambient_dir = f"{self.asset_dir}/ambient"
        # 允许用户上传任意支持的格式
        for ext in ['.wav', '.mp3', '.m4a', '.flac']:
            path = f"{ambient_dir}/{theme}{ext}"
            if os.path.exists(path):
                try:
                    logger.info(f"✅ 加载环境音: {path}")
                    audio = AudioSegment.from_file(path)
                    return self._normalize_audio(audio)
                except Exception as e:
                    logger.warning(f"无法加载环境音 {path}: {e}")
                    continue
        logger.info(f"未找到环境音 {theme}，使用静音回退")
        return AudioSegment.silent(duration=100)
    
    def get_transition_chime(self) -> AudioSegment:
        """🌟 防采样率爆炸：获取防惊跳柔和过渡音并强制归一化"""
        transitions_dir = f"{self.asset_dir}/transitions"
        # 支持多种音频格式
        for filename in ['soft_chime.wav', 'soft_chime.mp3', 'chime.wav', 'transition.wav']:
            path = os.path.join(transitions_dir, filename)
            if os.path.exists(path):
                try:
                    logger.info(f"✅ 加载过渡音: {path}")
                    audio = AudioSegment.from_file(path)
                    return self._normalize_audio(audio)
                except Exception as e:
                    logger.warning(f"无法加载过渡音 {path}: {e}")
                    continue
        logger.info("未找到过渡音，使用默认静音")
        return AudioSegment.silent(duration=500)  # 默认半秒空白
    
    def scan_voice_assets(self):
        """扫描可用的音色文件"""
        voices_dir = f"{self.asset_dir}/voices"
        if not os.path.exists(voices_dir):
            logger.warning(f"音色目录不存在: {voices_dir}")
            return []
        
        voice_files = []
        for file in os.listdir(voices_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                voice_files.append(os.path.join(voices_dir, file))
        
        logger.info(f"发现 {len(voice_files)} 个音色文件")
        return voice_files
    
    def add_custom_voice(self, name, file_path, gender="male", speed=1.0):
        """添加自定义音色"""
        if not os.path.exists(file_path):
            logger.error(f"音色文件不存在: {file_path}")
            return False
        
        voice_config = {
            "audio": file_path,
            "text": f"自定义音色 {name}",
            "speed": speed
        }
        
        if gender == "male":
            self.voices["male_pool"].append(voice_config)
        else:
            self.voices["female_pool"].append(voice_config)
        
        logger.info(f"添加自定义音色: {name}")
        return True

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    manager = AssetManager()
    
    # 测试音色获取
    print("测试音色获取:")
    print("旁白音色:", manager.get_voice_for_role("narration"))
    print("标题音色:", manager.get_voice_for_role("title"))
    print("对话音色:", manager.get_voice_for_role("dialogue", "张三", "male"))
    
    # 测试环境音
    print("\n测试环境音:")
    ambient = manager.get_ambient_sound()
    print(f"环境音时长: {len(ambient)}ms")
    
    # 测试过渡音
    chime = manager.get_transition_chime()
    print(f"过渡音时长: {len(chime)}ms")