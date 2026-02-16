#!/usr/bin/env python3
"""
CineCast 主控程序
三段式物理隔离架构 (Three-Stage Isolated Pipeline)
实现100%防内存溢出和断点续传
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
from modules.llm_director import LLMScriptDirector, atomic_json_write
from modules.mlx_tts_engine import MLXRenderEngine, group_indices_by_voice_type
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
        初始化CineCast三段式生产线
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or self._get_default_config()
        self.assets = AssetManager(self.config["assets_dir"])
        self.script_dir = os.path.join(self.config["output_dir"], "scripts")
        self.cache_dir = os.path.join(self.config["output_dir"], "temp_wav_cache")
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
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
    
    def check_ollama_alive(self):
        """前置检查：验证 Ollama 服务是否可用"""
        try:
            response = requests.get(
                "http://127.0.0.1:11434/api/tags", timeout=10
            )
            if response.status_code == 200:
                logger.info("✅ Ollama 服务前置检查通过")
                return True
            else:
                logger.error(f"❌ Ollama 服务响应异常 (HTTP {response.status_code})")
                return False
        except Exception as e:
            logger.error(f"❌ Ollama 服务不可达: {e}")
            return False

    # ==========================================
    # 🎬 阶段一：剧本化与微切片 (Script & Micro-chunking)
    # ==========================================
    def phase_1_generate_scripts(self, input_source):
        """阶段一：编剧期 (Ollama) - 生成包含chunk_id和停顿时间的微切片剧本"""
        logger.info("\n" + "="*50 + "\n🎬 [阶段一] 编剧期 (Ollama)\n" + "="*50)
        
        # 🌟 前置检查：确认 Ollama 服务存活
        if not self.check_ollama_alive():
            logger.error("❌ Ollama 服务不可用，阶段一中止。请检查 Ollama 是否已启动。")
            return False

        # 支持EPUB和TXT两种输入格式
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
        prev_chapter_content = None  # 用于存储上一章内容
        failed_chapters = []
        
        for chapter_name, content in chapters.items():
            script_path = os.path.join(self.script_dir, f"{chapter_name}_micro.json")
            if os.path.exists(script_path):
                logger.info(f"⏭️ 微切片剧本已存在，跳过: {chapter_name}")
                # 保留已有章节的文本给下一章用
                prev_chapter_content = content
                continue
                
            logger.info(f"✍️ 正在生成微切片剧本: {chapter_name} (字数: {len(content)})")
            try:
                # 🌟 修复：传入 chapter_name 作为 ID 前缀，避免文件名冲突
                micro_script = director.parse_and_micro_chunk(content, chapter_prefix=chapter_name)
                
                # 验证生成的剧本数据结构
                if not micro_script:
                    logger.error(f"❌ {chapter_name} 生成的微切片剧本为空，跳过该章节")
                    failed_chapters.append(chapter_name)
                    continue
                
                # 🌟 核心逻辑：如果不是第一章，且有上一章的内容，则生成并插入前情提要
                if prev_chapter_content is not None and len(prev_chapter_content) >= 1000:
                    logger.info(f"🔄 正在为 {chapter_name} 生成前情摘要...")
                    recap_text = director.generate_chapter_recap(prev_chapter_content)
                    
                    if recap_text:
                        # 构建一个标准的前情提要引子单元
                        intro_unit = {
                            "chunk_id": f"{chapter_name}_recap_intro",
                            "type": "recap",
                            "speaker": "talkover",
                            "content": "前情提要：",
                            "pause_ms": 500
                        }
                        # 构建摘要主体单元
                        recap_unit = {
                            "chunk_id": f"{chapter_name}_recap_body",
                            "type": "recap",
                            "speaker": "talkover",
                            "content": recap_text,
                            "pause_ms": 1500
                        }
                        # 将提要插入到本章剧本的最开头（在标题之后，正文之前）
                        micro_script.insert(1, intro_unit)
                        micro_script.insert(2, recap_unit)
                
                # 保存当前章的原始文本，供下一章使用
                prev_chapter_content = content
                
                # 验证每个片段都有必需的字段
                valid = True
                for i, item in enumerate(micro_script):
                    required_fields = ['chunk_id', 'type', 'speaker', 'content']
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        logger.error(f"❌ {chapter_name} 第{i+1}个片段缺少字段: {missing_fields}")
                        logger.error(f"   片段内容: {item}")
                        valid = False
                        break

                if not valid:
                    logger.error(f"❌ 章节 {chapter_name} 数据校验失败，跳过该章")
                    failed_chapters.append(chapter_name)
                    continue
                
                # 🌟 原子化写入：防止中断导致 JSON 损坏
                atomic_json_write(script_path, micro_script)
                logger.info(f"✅ 生成微切片剧本: {script_path} ({len(micro_script)}个片段)")
            except Exception as e:
                logger.error(f"❌ 章节 {chapter_name} 解析严重失败，跳过该章: {e}")
                import traceback
                logger.error(f"详细错误信息:\n{traceback.format_exc()}")
                failed_chapters.append(chapter_name)
                continue
                
        # 强制弹射Ollama内存
        self._eject_ollama_memory()

        if failed_chapters:
            logger.warning(f"⚠️ 以下章节处理失败: {', '.join(failed_chapters)}")

        logger.info("✅ 阶段一完成，Ollama已从内存中安全撤离！")
        return True
    
    # ==========================================
    # 🎙️ 阶段二：纯净干音渲染 (Dry Voice Rendering)
    # ==========================================
    def phase_2_render_dry_audio(self):
        """阶段二：录音期 (MLX TTS) - 纯净干音渲染，只产生WAV文件
        
        Uses a "group-by-voice" strategy: chunks sharing the same voice type
        are rendered consecutively to minimise MLX embedding switches.
        """
        logger.info("\n" + "="*50 + "\n🎙️ [阶段二] 录音期 (MLX TTS)\n" + "="*50)
        engine = MLXRenderEngine(self.config["model_path"])
        
        script_files = sorted([f for f in os.listdir(self.script_dir) if f.endswith('_micro.json')])
        total_chunks = 0
        rendered_chunks = 0
        
        for file in script_files:
            with open(os.path.join(self.script_dir, file), 'r', encoding='utf-8') as f:
                micro_script = json.load(f)
            total_chunks += len(micro_script)
            
            logger.info(f"🎙️ 正在渲染干音: {file} ({len(micro_script)}个片段)")
            
            # 🌟 Group-by-voice 优化：按角色分组批量渲染，减少 MLX 音色切换开销
            voice_groups = group_indices_by_voice_type(micro_script)
            for voice_key, indices in voice_groups.items():
                logger.info(f"   🎤 渲染音色组: {voice_key} ({len(indices)}个片段)")
                for idx in indices:
                    item = micro_script[idx]
                    voice_cfg = self.assets.get_voice_for_role(
                        item["type"], 
                        item.get("speaker"), 
                        item.get("gender")
                    )
                    save_path = os.path.join(self.cache_dir, f"{item['chunk_id']}.wav")
                    if engine.render_dry_chunk(item["content"], voice_cfg, save_path):
                        rendered_chunks += 1
                    
                    if rendered_chunks > 0 and rendered_chunks % 50 == 0:
                        logger.info(f"   🎵 进度: {rendered_chunks}/{total_chunks} 片段已渲染")
        
        # 释放 MLX 模型显存
        del engine
        try:
            import mlx.core as mx
            mx.clear_cache()
        except ImportError:
            pass
        logger.info(f"✅ 阶段二完成 ({rendered_chunks}/{total_chunks} 片段)，MLX 已从内存中安全撤离！")
        
    # ==========================================
    # 🎛️ 阶段三：电影级混音发版 (Cinematic Post-Processing)
    # ==========================================
    def phase_3_cinematic_mix(self):
        """阶段三：混音发版期 (Pydub) - 从干音缓存组装成电影级有声书"""
        logger.info("\n" + "="*50 + "\n🎛️ [阶段三] 混音发版期 (Pydub)\n" + "="*50)

        # 🌟 前置检查：确认缓存目录存在有效音频片段
        if os.path.isdir(self.cache_dir):
            wav_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.wav')]
        else:
            wav_files = []
        if not wav_files:
            logger.warning("⚠️ 未发现有效音频片段，请检查剧本解析阶段（阶段一）和干音渲染阶段（阶段二）是否成功。跳过混音。")
            return

        packager = CinematicPackager(self.config["output_dir"])
        ambient_bgm = self.assets.get_ambient_sound(self.config["ambient_theme"])
        chime_sound = self.assets.get_transition_chime()
        
        script_files = sorted([f for f in os.listdir(self.script_dir) if f.endswith('_micro.json')])
        for file in script_files:
            with open(os.path.join(self.script_dir, file), 'r', encoding='utf-8') as f:
                micro_script = json.load(f)
            # 🌟 Pydub 开始组装，此时已经没有大模型在抢占内存了
            packager.process_from_cache(micro_script, self.cache_dir, self.assets, ambient_bgm, chime_sound)
        
        logger.info("🎉 三段式架构全流程完成！全书压制完毕，请前往 output 目录查收。")
    
def main():
    """主函数 - 严格的三段式串行处理，彻底切断内存重叠"""
    producer = CineCastProducer()
    
    # 支持EPUB文件输入（通过命令行参数或配置）
    epub_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if epub_path and os.path.exists(epub_path):
        input_source = epub_path
        logger.info(f"📚 检测到EPUB文件: {epub_path}")
    else:
        # 回退到TXT目录模式
        input_dir = "./input_chapters"
        os.makedirs(input_dir, exist_ok=True)
        if not os.listdir(input_dir):
            logger.warning(f"⚠️ 请先在 {input_dir} 文件夹中放入测试用的 .txt 章节！")
            with open(os.path.join(input_dir, "第一章_测试.txt"), 'w', encoding='utf-8') as f:
                f.write("第一章 风雪\n1976年\n夜幕降临港口。\"你相信命运吗？\"老渔夫问。\n\"我不信。\"年轻人回答。")
        input_source = input_dir
        logger.info(f"📝 使用TXT目录模式: {input_dir}")
    
    try:
        # 严格的三段式串行处理，彻底切断内存重叠
        if producer.phase_1_generate_scripts(input_source):
            producer.phase_2_render_dry_audio()
            producer.phase_3_cinematic_mix()
    except Exception as e:
        logger.error(f"💥 三段式架构执行失败: {e}")

if __name__ == "__main__":
    main()