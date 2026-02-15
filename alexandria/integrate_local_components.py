#!/usr/bin/env python3
"""
Alexandria本地化集成脚本
将CineCast中测试通过的本地MLX Qwen模型集成到Alexandria项目中
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from local_llm_client import LocalLLMClient
from local_tts_engine import LocalTTSEngine

logger = logging.getLogger(__name__)

class AlexandriaLocalAdapter:
    """Alexandria本地化适配器"""
    
    def __init__(self, config_path: str = "local_config.json"):
        """初始化本地化适配器"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化组件
        self.llm_client = LocalLLMClient(self.config)
        self.tts_engine = LocalTTSEngine(self.config)
        
        logger.info("🎯 Alexandria本地化适配器初始化完成")
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            "llm": {
                "provider": "ollama",
                "model": "qwen14b-pro",
                "host": "http://localhost:11434",
                "api_url": "http://localhost:11434/api/chat",
                "temperature": 0.0,
                "num_ctx": 8192
            },
            "tts": {
                "mode": "local",
                "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",
                "device": "metal",
                "compile_codec": False,
                "language": "Chinese"
            },
            "processing": {
                "max_chars_per_chunk": 300,
                "context_window": 3,
                "smart_chunking": True
            }
        }
    
    def generate_local_script(self, text_chunk: str, context: str = "") -> list:
        """使用本地LLM生成剧本"""
        logger.info("🧠 使用本地Qwen14B-Pro生成剧本...")
        return self.llm_client.generate_script(text_chunk, context)
    
    def render_local_audio(self, text: str, voice_config: dict, save_path: str, emotion: str = "平静") -> bool:
        """使用本地TTS渲染音频"""
        logger.info(f"🎵 使用本地MLX Qwen-TTS渲染音频: {save_path}")
        return self.tts_engine.render_dry_chunk(text, voice_config, save_path, emotion)
    
    def process_book_chunk(self, text_chunk: str, chunk_id: str, output_dir: str, context: str = "") -> bool:
        """处理书籍片段的完整流程"""
        try:
            # 1. 生成剧本
            logger.info(f"챕터 {chunk_id} 开始处理...")
            script = self.generate_local_script(text_chunk, context)
            
            if not script:
                logger.error(f"챕터 {chunk_id} 剧本生成失败")
                return False
            
            # 2. 渲染音频
            chunk_dir = os.path.join(output_dir, f"chunk_{chunk_id}")
            os.makedirs(chunk_dir, exist_ok=True)
            
            success_count = 0
            for i, item in enumerate(script):
                wav_path = os.path.join(chunk_dir, f"{i:04d}_{item['type']}.wav")
                voice_config = {
                    "speaker": item["speaker"],
                    "gender": item["gender"]
                }
                
                if self.render_local_audio(item["content"], voice_config, wav_path, item.get("emotion", "平静")):
                    success_count += 1
            
            logger.info(f"챕터 {chunk_id} 处理完成: {success_count}/{len(script)} 片段成功")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"챕터 {chunk_id} 处理失败: {e}")
            return False
    
    def health_check(self) -> dict:
        """健康检查"""
        checks = {
            "ollama_connection": self.llm_client._check_connection(),
            "tts_engine_available": self.tts_engine.is_available(),
            "config_loaded": bool(self.config),
            "components_initialized": all([
                hasattr(self, 'llm_client'),
                hasattr(self, 'tts_engine')
            ])
        }
        
        overall_status = all(checks.values())
        checks["overall_status"] = "✅ 正常" if overall_status else "❌ 异常"
        
        return checks

def main():
    """主函数 - 演示本地化集成"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 初始化适配器
    adapter = AlexandriaLocalAdapter()
    
    # 健康检查
    print("\n🏥 健康检查结果:")
    health_status = adapter.health_check()
    for check, status in health_status.items():
        print(f"  {check}: {status}")
    
    if health_status["overall_status"].startswith("❌"):
        print("\n⚠️ 系统存在问题，请检查配置和依赖")
        return
    
    # 简单测试
    print("\n🧪 简单功能测试:")
    test_text = "第一章 测试\n这是一个简单的测试文本，用来验证本地化集成是否正常工作。"
    
    try:
        # 测试剧本生成
        script = adapter.generate_local_script(test_text)
        print(f"  ✅ 剧本生成成功: {len(script)} 个片段")
        
        # 显示生成的剧本片段
        for i, item in enumerate(script[:3]):  # 只显示前3个
            print(f"    片段 {i+1}: [{item['type']}] {item['speaker']}: {item['content'][:30]}...")
        
        print("  ✅ 本地化集成测试通过!")
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

if __name__ == "__main__":
    main()