#!/usr/bin/env python3
"""
CineCast 主控程序
串联所有车间，实现全自动化跑通
"""

import os
import sys
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
        self._initialize_components()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            "assets_dir": "./assets",
            "output_dir": "./output/Fish_No_Feet",
            "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",  # 相对于cinecast目录
            "use_local_llm": True,
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
    
    def process_text(self, text: str, chapter_title: str = ""):
        """
        处理单段文本
        
        Args:
            text: 待处理的文本
            chapter_title: 章节标题（可选）
        """
        logger.info(f"📄 处理文本: {chapter_title or '无标题'} ({len(text)}字符)")
        
        # 获取全局配置的声场和过渡音
        ambient_bgm = self.assets.get_ambient_sound(self.config["ambient_theme"])
        chime_sound = self.assets.get_transition_chime()
        
        # A. 编剧处理 (大模型角色分配)
        script = self.director.parse_text_to_script(text)
        logger.info(f"🎭 剧本解析完成，共 {len(script)} 个单元")
        
        # B. 录音处理
        for i, unit in enumerate(script):
            try:
                logger.info(f"🎤 处理单元 {i+1}/{len(script)}: {unit['type']} - {unit.get('speaker', '未知')}")
                
                # 获取对应的音色配置
                voice_cfg = self.assets.get_voice_for_role(
                    unit["type"], 
                    unit.get("speaker"), 
                    unit.get("gender", "male")
                )
                
                # MLX 渲染音频片段
                unit_audio = self.engine.render_unit(unit["content"], voice_cfg)
                
                # C. 送入发行缓冲池
                self.packager.add_audio(unit_audio, ambient=ambient_bgm, chime=chime_sound)
                
            except Exception as e:
                logger.error(f"❌ 单元处理失败: {e}")
                continue
        
        # 显示当前缓冲区状态
        status = self.packager.get_buffer_status()
        logger.info(f"📊 缓冲区状态: {status['buffer_length_min']:.1f}/{status['target_duration_min']:.1f}分钟")
    
    def process_chapter_file(self, file_path: str):
        """
        处理章节文件
        
        Args:
            file_path: 章节文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取文件名作为章节标题
            chapter_title = os.path.splitext(os.path.basename(file_path))[0]
            self.process_text(content, chapter_title)
            
        except Exception as e:
            logger.error(f"❌ 处理文件失败 {file_path}: {e}")
    
    def process_epub_directory(self, epub_dir: str):
        """
        处理EPUB目录中的所有章节
        
        Args:
            epub_dir: EPUB文本目录路径
        """
        if not os.path.exists(epub_dir):
            logger.error(f"❌ 目录不存在: {epub_dir}")
            return
        
        # 获取所有文本文件
        text_files = []
        for file in os.listdir(epub_dir):
            if file.lower().endswith(('.txt', '.md')):
                text_files.append(os.path.join(epub_dir, file))
        
        # 按文件名排序
        text_files.sort()
        
        logger.info(f"📚 发现 {len(text_files)} 个章节文件")
        
        # 依次处理每个章节
        for i, file_path in enumerate(text_files, 1):
            logger.info(f"챕 开始处理第 {i}/{len(text_files)} 章")
            self.process_chapter_file(file_path)
    
    def finalize_production(self):
        """完成整个生产流程"""
        logger.info("🔚 开始最终化处理...")
        
        ambient_bgm = self.assets.get_ambient_sound(self.config["ambient_theme"])
        chime_sound = self.assets.get_transition_chime()
        
        # 处理书籍结尾的碎片
        self.packager.finalize(ambient=ambient_bgm, chime=chime_sound)
        
        logger.info("🎉 全书压制完成！")
        
        # 显示最终统计
        final_status = self.packager.get_buffer_status()
        logger.info(f"📊 最终统计: 生成 {final_status['current_file_index'] - 1} 个音频文件")

def main():
    """主函数"""
    logger.info("🎬 CineCast 电影级有声书生产线启动")
    
    try:
        # 创建生产线实例
        producer = CineCastProducer()
        
        # 示例：处理测试文本
        test_text = """
第一章 凯夫拉维克的风雪

夜幕降临，港口的灯火开始闪烁。

"你相信命运吗？"老渔夫说道。

年轻人摇摇头："我只相信海。"

远处传来汽笛声，划破了寂静的夜空。

海浪拍打着礁石，发出永恒的节奏。就像时间一样，永不停息地向前流淌。
"""
        
        # 处理测试文本
        producer.process_text(test_text, "第一章 测试")
        
        # 完成生产
        producer.finalize_production()
        
        logger.info("✅ 生产线运行完成")
        
    except Exception as e:
        logger.error(f"💥 生产线运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()