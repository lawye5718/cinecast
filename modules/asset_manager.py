#!/usr/bin/env python3
"""
CineCast 资产与选角管理器
负责处理音色、声场、音效的加载与智能分配
"""

import os
import random
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

class AssetManager:
    def __init__(self, asset_dir="./assets"):
        self.asset_dir = asset_dir
        self._initialize_default_voices()
        self.role_voice_map = {}  # 记忆已分配角色的音色
        
    def _initialize_default_voices(self):
        """初始化默认音色配置"""
        self.voices = {
            "narrator": {
                "audio": f"{self.asset_dir}/voices/narrator.wav", 
                "text": "沉稳的旁白", 
                "speed": 1.0
            },
            "title": {
                "audio": f"{self.asset_dir}/voices/narrator.wav", 
                "text": "沉稳的旁白", 
                "speed": 0.8
            },  # 减速一字一顿
            "subtitle": {
                "audio": f"{self.asset_dir}/voices/narrator.wav", 
                "text": "沉稳的旁白", 
                "speed": 0.9
            },
            "male_pool": [
                {
                    "audio": f"{self.asset_dir}/voices/m1.wav", 
                    "text": "男声一号", 
                    "speed": 1.05
                },
                {
                    "audio": f"{self.asset_dir}/voices/m2.wav", 
                    "text": "男声二号", 
                    "speed": 1.0
                }
            ],
            "female_pool": [
                {
                    "audio": f"{self.asset_dir}/voices/f1.wav", 
                    "text": "女声一号", 
                    "speed": 1.05
                }
            ]
        }
    
    def get_voice_for_role(self, role_type, speaker_name=None, gender="male"):
        """
        智能选角逻辑
        
        Args:
            role_type: 角色类型 (title, subtitle, narration, dialogue)
            speaker_name: 说话人姓名 (用于对话角色记忆)
            gender: 性别 (male, female)
        """
        # 处理非对话角色
        if role_type in ["title", "subtitle", "narration"]:
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
        """
        获取沉浸式声场
        如果用户未上传则返回静音
        """
        path = f"{self.asset_dir}/ambient/{theme}.wav"
        if os.path.exists(path):
            try:
                return AudioSegment.from_file(path)
            except Exception as e:
                logger.warning(f"无法加载环境音 {path}: {e}")
                return AudioSegment.silent(duration=100)  # Fallback
        return AudioSegment.silent(duration=100)  # Fallback
    
    def get_transition_chime(self) -> AudioSegment:
        """获取防惊跳柔和过渡音"""
        path = f"{self.asset_dir}/transitions/soft_chime.wav"
        if os.path.exists(path):
            try:
                return AudioSegment.from_file(path)
            except Exception as e:
                logger.warning(f"无法加载过渡音 {path}: {e}")
                return AudioSegment.silent(duration=500)  # 默认半秒空白
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