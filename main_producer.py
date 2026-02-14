#!/usr/bin/env python3
"""
CineCast 主控程序
串联所有车间，实现全自动化跑通
"""

import os
import sys
import json
import logging
import requests
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
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
            "output_dir": "./output/Audiobooks",
            "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",  # 相对于cinecast目录
            "ambient_theme": "iceland_wind",  # 环境音主题
            "target_duration_min": 30,  # 目标时长（分钟）
            "min_tail_min": 10,  # 最小尾部时长（分钟）
            "use_local_llm": True  # 是否使用本地LLM
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
    
    def _extract_epub_chapters(self, epub_path: str) -> dict:
        """🌟 从 EPUB 提取干净的章节文本字典 {章节名: 文本内容}"""
        logger.info(f"📖 正在解析 EPUB 文件: {epub_path}")
        book = epub.read_epub(epub_path)
        chapters = {}
        for idx, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator='\n')
            clean_text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
            if len(clean_text) > 100: # 过滤极短废页
                title = f"Chapter_{idx:03d}"
                chapters[title] = clean_text
        return chapters
    
    def _eject_ollama_memory(self):
        """🌟 核心绝招：强行弹射 Ollama 模型，清空 M4 显存"""
        logger.info("🧹 正在卸载 Ollama 模型释放显存...")
        try:
            requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": "qwen14b-pro", "prompt": "bye", "keep_alive": 0},
                timeout=10
            )
            logger.info("✅ 14B 大模型已成功从统一内存中弹射！")
        except Exception as e:
            logger.warning(f"⚠️ 弹射 Ollama 失败，可能已自动释放: {e}")
    
    # ==========================================
    # 🌟 阶段一：编剧期 (Ollama 14B 独占内存)
    # ==========================================
    def phase_1_generate_scripts(self, input_source):
        """🌟 阶段一：启动编剧引擎 (Ollama 14B 独占内存)"""
        logger.info("\n" + "="*50 + "\n🎬 [阶段一] 启动编剧引擎 (Ollama 14B)...\n" + "="*50)
        
        # 🌟 支持EPUB和TXT两种输入格式
        if input_source.endswith('.epub'):
            chapters = self._extract_epub_chapters(input_source)
            if not chapters:
                logger.error("❌ EPUB 解析失败或无有效文本！")
                return False
        else:
            # 处理TXT目录
            text_files = sorted([f for f in os.listdir(input_source) if f.endswith(('.txt', '.md'))])
            if not text_files:
                logger.error(f"❌ 目录 {input_source} 为空，无法生成剧本！")
                return False
            chapters = {}
            for file_name in text_files:
                with open(os.path.join(input_source, file_name), 'r', encoding='utf-8') as f:
                    chapters[os.path.splitext(file_name)[0]] = f.read()
    
        director = LLMScriptDirector()
        
        for chapter_name, content in chapters.items():
            script_path = os.path.join(self.script_dir, f"{chapter_name}.json")
            if os.path.exists(script_path):
                logger.info(f"⏭️ 剧本已存在，跳过: {chapter_name}")
                continue
                
            logger.info(f"✍️ 正在构思剧本: {chapter_name} (字数: {len(content)})")
            script = director.parse_text_to_script(content)
            
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(script, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ 生成剧本: {script_path}")
                
        # 🌟 阶段一结束，立即弹射内存
        self._eject_ollama_memory()
        return True
    
    # ==========================================
    # 🌟 阶段二：录音与混音期 (MLX 独占内存)
    # ==========================================
    def phase_2_render_audio(self):
        """🌟 阶段二：启动录音棚 (MLX TTS 引擎 独占内存)"""
        logger.info("\n" + "="*50 + "\n🎬 [阶段二] 启动录音棚 (MLX TTS 引擎)...\n" + "="*50)
        engine = MLXRenderEngine(self.config["model_path"])
        packager = CinematicPackager(self.config["output_dir"])
            
        ambient_bgm = self.assets.get_ambient_sound(self.config["ambient_theme"])
        chime_sound = self.assets.get_transition_chime()
            
        script_files = sorted([f for f in os.listdir(self.script_dir) if f.endswith('.json')])
            
        for file in script_files:
            with open(os.path.join(self.script_dir, file), 'r', encoding='utf-8') as f:
                script = json.load(f)
                    
            logger.info(f"🎙️ 正在录制剧本: {file}")
            for unit in script:
                try:
                    voice_cfg = self.assets.get_voice_for_role(
                        unit["type"], unit.get("speaker"), unit.get("gender", "male")
                    )
                    unit_audio = engine.render_unit(unit["content"], voice_cfg)
                    packager.add_audio(unit_audio, ambient=ambient_bgm, chime=chime_sound)
                except Exception as e:
                    logger.error(f"❌ 渲染单元失败跳过: {e}")
                        
        packager.finalize(ambient=ambient_bgm, chime=chime_sound)
        logger.info("🎉 阶段二完成！全书压制完毕，请前往 output 目录查收。")
    
def main():
    """主函数"""
    producer = CineCastProducer()
    # 🌟 支持EPUB文件输入
    epub_path = "../qwentts/tests/鱼没有脚 (约恩卡尔曼斯特凡松) (Z-Library)-2024-04-30-09-13-38.epub" 
    
    if os.path.exists(epub_path):
        input_source = epub_path
        logger.info(f"📚 检测到EPUB文件: {epub_path}")
    else:
        # 回退到TXT目录模式
        input_dir = "./input_chapters"
        os.makedirs(input_dir, exist_ok=True)
        if not os.listdir(input_dir):
            logger.warning(f"⚠️ 请先在 {input_dir} 文件夹中放入测试用的 .txt 章节！")
            with open(os.path.join(input_dir, "第一章_测试.txt"), 'w') as f:
                f.write("第一章 风雪\n1976年\n夜幕降临港口。\"你相信命运吗？\"老渔夫问。\n\"我不信。\"年轻人回答。")
        input_source = input_dir
        logger.info(f"📝 使用TXT目录模式: {input_dir}")
    
    try:
        if producer.phase_1_generate_scripts(input_source):
            producer.phase_2_render_audio()
    except Exception as e:
        logger.error(f"💥 生产线崩溃: {e}")

if __name__ == "__main__":
    main()