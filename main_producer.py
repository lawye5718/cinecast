#!/usr/bin/env python3
"""
CineCast 主控程序
串联所有车间，实现全自动化跑通
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.asset_manager import AssetManager
from modules.llm_director import LLMScriptDirector
from modules.mlx_tts_engine import MLXRenderEngine
from modules.cinematic_packager import CinematicPackager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cinecast.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CineCastProducer:
    def __init__(self, config=None):
        """
        初始化CineCast生产线
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or self._get_default_config()
        self.assets = AssetManager(self.config["assets_dir"])
        self.script_dir = os.path.join(self.config["output_dir"], "scripts")
        os.makedirs(self.script_dir, exist_ok=True)
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            "assets_dir": "./assets",
            "output_dir": "./output/Fish_No_Feet",
            "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",  # 相对于cinecast目录
            "ambient_theme": "iceland_wind",  # 环境音主题
            "target_duration_min": 30,  # 目标时长（分钟）
            "min_tail_min": 10  # 最小尾部时长（分钟）
        }
    
    def _initialize_components(self):
        """初始化各个组件"""
        logger.info("🎬 初始化CineCast电影级有声书生产线...")
        
        try:
            # 1. 初始化资产管理系统
            self.assets = AssetManager(self.config["assets_dir"])
            logger.info("✅ 资产管理系统初始化完成")
            
            # 2. 初始化LLM剧本导演
            self.director = LLMScriptDirector(
                use_local_mlx_lm=self.config["use_local_llm"]
            )
            logger.info("✅ LLM剧本导演初始化完成")
            
            # 3. 初始化MLX渲染引擎
            model_path = self.config["model_path"]
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(model_path):
                model_path = os.path.join(project_root.parent, model_path)
            
            self.engine = MLXRenderEngine(model_path)
            logger.info("✅ MLX渲染引擎初始化完成")
            
            # 4. 初始化混音打包器
            self.packager = CinematicPackager(self.config["output_dir"])
            logger.info("✅ 混音打包器初始化完成")
            
            logger.info("🎉 所有组件初始化完成！")
            
        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            raise
    
    # ==========================================
    # 阶段一：编剧期 (调用 Ollama 14B)
    # ==========================================
    def phase_1_generate_scripts(self, text_files: list):
        """阶段一：启动编剧引擎 (Ollama 14B)"""
        logger.info("🎬 [阶段一] 启动编剧引擎 (Ollama 14B)...")
        director = LLMScriptDirector() # 内部使用 keep_alive=0 自动回收内存
        
        for file_path in text_files:
            chapter_name = os.path.splitext(os.path.basename(file_path))[0]
            script_path = os.path.join(self.script_dir, f"{chapter_name}.json")
            
            if os.path.exists(script_path):
                logger.info(f"⏭️ 剧本已存在，跳过: {chapter_name}")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            logger.info(f"✍️ 正在拆解剧本: {chapter_name}")
            script = director.parse_text_to_script(content)
            
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(script, f, ensure_ascii=False, indent=2)
                
        logger.info("✅ 阶段一完成！建议人工审阅 scripts/ 下的剧本文件。")

    # ==========================================
    # 阶段二：渲染期 (独占调用 MLX TTS)
    # ==========================================
    def phase_2_render_audio(self):
        """阶段二：启动录音棚 (MLX TTS 引擎)"""
        logger.info("🎬 [阶段二] 启动录音棚 (MLX TTS 引擎)...")
        # 此时 Ollama 已经释放内存，M4 的 24GB 全部归 MLX 所有！
        engine = MLXRenderEngine(self.config["model_path"])
        packager = CinematicPackager(self.config["output_dir"])
        
        ambient_bgm = self.assets.get_ambient_sound(self.config["ambient_theme"])
        chime_sound = self.assets.get_transition_chime()
        
        script_files = sorted([f for f in os.listdir(self.script_dir) if f.endswith('.json')])
        
        for file in script_files:
            with open(os.path.join(self.script_dir, file), 'r', encoding='utf-8') as f:
                script = json.load(f)
                
            logger.info(f"🎙️ 正在录制: {file}")
            for unit in script:
                # 获取音色并渲染
                voice_cfg = self.assets.get_voice_for_role(
                    unit["type"], unit.get("speaker"), unit.get("gender", "male")
                )
                unit_audio = engine.render_unit(unit["content"], voice_cfg)
                packager.add_audio(unit_audio, ambient=ambient_bgm, chime=chime_sound)
                
        # 最终封包尾部音频
        packager.finalize(ambient=ambient_bgm, chime=chime_sound)
        logger.info("🎉 阶段二完成！全书压制完毕！")
    
def main():
    """主函数"""
    producer = CineCastProducer()
    
    # 假设你的 txt 章节放在 ./input/chapters 目录下
    input_dir = "./input/chapters"
    os.makedirs(input_dir, exist_ok=True)
    
    # 获取需要处理的文件列表
    text_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')])
    
    if not text_files:
        logger.error("❌ input/chapters 目录下没有TXT文件。请放入章节文件后重试。")
        return

    # 完美的解耦流水线
    try:
        producer.phase_1_generate_scripts(text_files)
        # 你甚至可以在这里加一个 input("请人工审阅剧本后，按回车键开始录制...")
        producer.phase_2_render_audio()
    except Exception as e:
        logger.error(f"💥 生产线运行失败: {e}")

if __name__ == "__main__":
    main()