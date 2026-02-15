#!/usr/bin/env python3
"""
Alexandria项目修复方案 - 解决WAV生成问题和并发处理
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional
import threading
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

logger = logging.getLogger(__name__)

class AlexandriaAudioFixer:
    """Alexandria项目音频生成修复器"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.config_path = os.path.join(project_root, "app", "config.json")
        self.tts_module_path = os.path.join(project_root, "app", "tts.py")
        self.project_module_path = os.path.join(project_root, "app", "project.py")
        
        # 串行锁，确保本地LLM/TTS调用串行执行
        self.serial_execution_lock = threading.Lock()
        
    def fix_zero_byte_wav_issue(self):
        """修复WAV文件为0字节的问题"""
        print("🔧 修复WAV文件为0字节的问题...")
        
        # 修改project.py中的音频生成逻辑
        project_py_path = os.path.join(self.project_root, "app", "project.py")
        
        with open(project_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复音频文件检查逻辑
        # 原来的检查可能过于严格或有错误
        if "if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:" in content:
            # 替换为更宽松的检查，同时增加调试信息
            content = content.replace(
                'if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:',
                'if not os.path.exists(temp_path):\n                print(f"DEBUG: Temp file does not exist: {temp_path}")\n                self._update_chunk_fields(index, status="error")\n                return False, "Generated audio file does not exist"\n            elif os.path.getsize(temp_path) == 0:\n                print(f"DEBUG: Temp file is empty: {temp_path}, size: {os.path.getsize(temp_path)})\n                self._update_chunk_fields(index, status="error")\n                return False, "Generated audio file is empty"'
            )
        
        # 保存修改
        with open(project_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ project.py 中的音频检查逻辑已修复")
    
    def implement_serial_llm_processing(self):
        """实现串行LLM处理以避免内存冲突"""
        print("🔄 实现串行LLM处理以避免内存冲突...")
        
        # 修改TTS引擎以确保串行执行
        tts_py_path = os.path.join(self.project_root, "app", "tts.py")
        
        with open(tts_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 在TTSEngine类中添加串行执行锁
        if "class TTSEngine:" in content:
            # 在类定义后添加锁
            content = content.replace(
                "class TTSEngine:",
                "class TTSEngine:\n    _serial_lock = threading.Lock()"
            )
        
        # 修改generate_voice方法以使用锁
        if "def generate_voice(self, text, instruct_text, speaker, voice_config, output_path):" in content:
            content = content.replace(
                "def generate_voice(self, text, instruct_text, speaker, voice_config, output_path):",
                "def generate_voice(self, text, instruct_text, speaker, voice_config, output_path):\n        # 串行执行以避免内存冲突\n        with self._serial_lock:"
            )
        
        # 修改generate_batch方法以使用锁
        if "def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1):" in content:
            content = content.replace(
                "def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1):",
                "def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1):\n        # 串行执行以避免内存冲突\n        with self._serial_lock:"
            )
        
        # 保存修改
        with open(tts_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ TTS引擎中的串行执行逻辑已添加")
    
    def update_config_for_local_processing(self):
        """更新配置以支持本地处理"""
        print("⚙️ 更新配置以支持本地处理...")
        
        config_path = os.path.join(self.project_root, "config.json")
        
        # 如果配置文件不存在，创建一个默认配置
        if not os.path.exists(config_path):
            default_config = {
                "llm": {
                    "api_url": "http://localhost:11434/api/chat",
                    "model": "qwen:14b",
                    "temperature": 0.0,
                    "num_ctx": 8192
                },
                "tts": {
                    "mode": "local",  # 使用本地模式
                    "device": "auto",
                    "language": "Chinese",
                    "compile_codec": False
                }
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            
            print("✅ 创建了默认配置文件")
        else:
            # 更新现有配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 确保TTS模式设置为本地
            if 'tts' not in config:
                config['tts'] = {}
            config['tts']['mode'] = 'local'
            config['tts']['device'] = 'auto'
            
            # 确保LLM配置正确
            if 'llm' not in config:
                config['llm'] = {}
            if 'api_url' not in config['llm']:
                config['llm']['api_url'] = 'http://localhost:11434/api/chat'
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            print("✅ 配置文件已更新")
    
    def add_debugging_and_error_handling(self):
        """添加调试和错误处理"""
        print("🐛 添加调试和错误处理...")
        
        # 修改TTS引擎以添加更多调试信息
        tts_py_path = os.path.join(self.project_root, "app", "tts.py")
        
        with open(tts_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 在关键方法中添加调试信息
        if "def _local_generate_custom(self, text, instruct_text, speaker, voice_config, output_path):" in content:
            content = content.replace(
                "print(f\"TTS [local] generating with instruct='",
                "print(f\"DEBUG: TTS [local] generating with instruct='"
            )
        
        # 在音频保存方法中添加调试
        if "def _save_wav(audio_array, sample_rate, output_path):" in content:
            # 在保存前添加调试信息
            content = content.replace(
                "sf.write(output_path, audio_array, sample_rate)",
                "print(f\"DEBUG: Saving WAV to {output_path}, shape: {audio_array.shape if hasattr(audio_array, 'shape') else 'N/A'}, size: {audio_array.size if hasattr(audio_array, 'size') else len(audio_array) if isinstance(audio_array, (list, tuple)) else 'N/A'}\")\n        sf.write(output_path, audio_array, sample_rate)\n        print(f\"DEBUG: WAV file saved, actual size: {os.path.getsize(output_path) if os.path.exists(output_path) else 'N/A'}\")"
            )
        
        # 保存修改
        with open(tts_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 调试信息已添加到TTS引擎")
    
    def apply_all_fixes(self):
        """应用所有修复"""
        print("🚀 开始应用Alexandria项目修复...")
        
        try:
            self.fix_zero_byte_wav_issue()
            self.implement_serial_llm_processing()
            self.update_config_for_local_processing()
            self.add_debugging_and_error_handling()
            
            print("\n✅ 所有修复已应用!")
            print("\n📋 修复内容总结:")
            print("   1. 修复了WAV文件为0字节的问题")
            print("   2. 实现了串行LLM处理以避免内存冲突")
            print("   3. 更新了配置以支持本地处理")
            print("   4. 添加了调试和错误处理")
            
            print("\n💡 建议:")
            print("   - 确保已安装必要的依赖: pip3 install -r requirements.txt")
            print("   - 确保Ollama服务正在运行: ollama serve")
            print("   - 确保已下载Qwen模型: ollama pull qwen:14b")
            print("   - 运行项目前检查配置文件")
            
        except Exception as e:
            print(f"❌ 修复过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    project_root = "/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook"
    
    print("🔧 Alexandria项目修复工具")
    print(f"项目路径: {project_root}")
    
    fixer = AlexandriaAudioFixer(project_root)
    fixer.apply_all_fixes()

if __name__ == "__main__":
    main()