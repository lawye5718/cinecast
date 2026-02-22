#!/usr/bin/env python3
"""
CineCast 主控程序
三段式物理隔离架构 (Three-Stage Isolated Pipeline)
实现100%防内存溢出和断点续传
"""

import argparse
import gc
import os
import re
import sys
import json
import logging
import time
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
from logging.handlers import RotatingFileHandler

# 配置日志 - 使用轮转处理器防止日志文件无限增长
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 文件轮转处理器 - 每个文件最大 10MB，保留 5 个备份文件
file_handler = RotatingFileHandler(
    'cinecast.log',
    encoding='utf-8',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 添加处理器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 渲染超时阈值（秒）。
# 冷启动阈值：引擎刚初始化时，MLX 需要 JIT 编译 Metal 着色器，首次推理耗时较长。
ENGINE_COLD_START_THRESHOLD_SECONDS = 120.0
# 热运行阈值：引擎热身完成后，正常渲染超过此值视为大模型幻觉/内存碎片化，触发引擎热重启。
ENGINE_WARM_THRESHOLD_SECONDS = 45.0

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

    def _create_tts_engine(self):
        """创建 MLX TTS 引擎，支持 1.7B Model Pool 配置
        
        Returns:
            MLXRenderEngine: 配置好的 TTS 引擎实例
        """
        engine_config = {}
        for key in ("model_path_base", "model_path_design",
                    "model_path_custom", "model_path_fallback",
                    "default_narrator_voice"):
            val = self.config.get(key)
            if val:
                engine_config[key] = val
        return MLXRenderEngine(self.config["model_path"], config=engine_config)

    def _get_default_config(self):
        """获取默认配置"""
        return {
            "assets_dir": "./assets",
            "output_dir": "./output/Audiobooks",
            "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",  # 相对于cinecast目录
            "model_path_base": None,     # 1.7B Base (克隆用)
            "model_path_design": None,   # 1.7B VoiceDesign (设计用)
            "model_path_custom": None,   # 1.7B CustomVoice (内置角色用)
            "model_path_fallback": None, # 0.6B 回退路径
            "ambient_theme": "iceland_wind",  # 环境音主题
            "target_duration_min": 30,  # 目标时长（分钟）
            "min_tail_min": 10,  # 最小尾部时长（分钟）
            "use_local_llm": True,  # 是否使用本地LLM
            "enable_recap": True,  # 🌟 前情提要总开关
            "pure_narrator_mode": False,  # 🌟 纯净旁白模式开关
            "user_recaps": None,  # 🌟 用户提供的前情提要文本（跳过LLM生成）
            "global_cast": {},  # 🌟 外脑全局角色设定集（Character Bible）
            "custom_recaps": {},  # 🌟 外脑前情提要字典 {Chapter_NNN: recap_text}
            "enable_auto_recap": True,  # 🌟 是否启用本地LLM自动生成摘要
            "default_narrator_voice": "aiden",  # 🌟 默认旁白基底音色 (Qwen3-TTS Preset)
        }
    
    def _initialize_components(self):
        """初始化各个组件"""
        logger.info("🎬 初始化CineCast电影级有声书生产线...")
        
        try:
            # 1. 初始化资产管理系统
            self.assets = AssetManager(self.config["assets_dir"])
            logger.info("✅ 资产管理系统初始化完成")
            
            # 2. 初始化LLM剧本导演
            self.director = LLMScriptDirector()
            logger.info("✅ LLM剧本导演初始化完成")
            
            # 3. 初始化MLX渲染引擎
            model_path = self.config["model_path"]
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(model_path):
                model_path = os.path.join(project_root.parent, model_path)
            
            # 构建引擎配置（支持 1.7B Model Pool）
            _path_keys = {"model_path_base", "model_path_design",
                          "model_path_custom", "model_path_fallback"}
            engine_config = {}
            for key in (*_path_keys, "default_narrator_voice"):
                val = self.config.get(key)
                if val and key in _path_keys and not os.path.isabs(val):
                    val = os.path.join(project_root.parent, val)
                if val:
                    engine_config[key] = val

            self.engine = MLXRenderEngine(model_path, config=engine_config)
            logger.info("✅ MLX渲染引擎初始化完成")
            
            # 4. 初始化混音打包器
            target_min = self.config.get("target_duration_min", 30)
            self.packager = CinematicPackager(self.config["output_dir"], target_duration_min=target_min)
            logger.info("✅ 混音打包器初始化完成")
            
            logger.info("🎉 所有组件初始化完成！")
            
        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            raise
    
    @staticmethod
    def _cn_to_int(cn_str: str) -> int:
        """辅助方法：将中文数字转换为阿拉伯数字。

        支持：零-九、十、百、千、两（如 三百四十五 -> 345，两百 -> 200）。
        纯阿拉伯数字字符串直接转换（如 "123" -> 123）。
        不在映射表中的字符会被静默忽略。
        """
        cn_num = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                  '十': 10, '百': 100, '千': 1000, '两': 2}
        if cn_str.isdigit():
            return int(cn_str)
        result, temp = 0, 0
        for char in cn_str:
            if char in cn_num:
                val = cn_num[char]
                if val >= 10:
                    if temp == 0: temp = 1
                    result += temp * val
                    temp = 0
                else:
                    temp = val
        return result + temp

    @staticmethod
    def parse_user_recaps(raw_text: str) -> dict:
        """增强版解析：支持'章'、'回'，支持中文数字（如第一百二十回）

        支持的格式（每章之间用空行或章节标记分隔）：
            第1章：摘要内容...
            第2章：摘要内容...
            第一百二十回：摘要内容...
        或：
            Chapter 1: recap text...
            Chapter 2: recap text...
        或简单的按行分隔（每行对应一章的前情提要，第1行用于第2章，第2行用于第3章，以此类推）：
            第一章的摘要内容（将作为第2章的前情提要）
            第二章的摘要内容（将作为第3章的前情提要）
        """
        if not raw_text or not raw_text.strip():
            return {}

        recaps = {}
        # 兼容: 第1章, 第一章, 第120回, 第一百二十回, Chapter 1
        pattern = re.compile(
            r'(?:第\s*([0-9零一二三四五六七八九十百千两]+)\s*[章回]|Chapter[_ ]?(\d+))\s*[：:]\s*(.+?)(?=\n\s*(?:第\s*[0-9零一二三四五六七八九十百千两]+\s*[章回]|Chapter[_ ]?\d+)|$)',
            re.DOTALL | re.IGNORECASE
        )
        matches = pattern.findall(raw_text)

        if matches:
            for m in matches:
                # m[0] 是中文/阿拉伯数字(章/回), m[1] 是 Chapter 格式的数字
                num_str = m[0] or m[1]
                chapter_num = CineCastProducer._cn_to_int(num_str)
                recap_text = m[2].strip()
                if recap_text and chapter_num > 0:
                    recaps[chapter_num] = recap_text
        else:
            # 回退：按非空行分割，第 N 行对应第 N+1 章（因为第1章没有前情提要）
            lines = [line.strip() for line in raw_text.strip().split('\n') if line.strip()]
            for idx, line in enumerate(lines):
                recaps[idx + 2] = line  # 从第2章开始

        return recaps

    def _extract_epub_chapters(self, epub_path: str) -> dict:
        """🌟 从 EPUB 提取干净的章节文本字典 {章节名: 文本内容}"""
        logger.info(f"📖 正在解析 EPUB 文件: {epub_path}")
        book = epub.read_epub(epub_path)
        chapters = {}
        for idx, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator='\n')
            clean_text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
            if len(clean_text) > 20: # 过滤极短废页（降低阈值以保留简短章节）
                title = f"Chapter_{idx:03d}"
                chapters[title] = clean_text
        return chapters
    
    def check_api_connectivity(self):
        """前置检查：验证云端 API 连通性 (DashScope Qwen-Flash)"""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            logger.error("❌ 未设置 DASHSCOPE_API_KEY 环境变量，无法使用 Qwen API。")
            return False
        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": "qwen-flash",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8,
                },
                timeout=10,
            )
            if response.status_code == 200:
                logger.info("✅ Qwen API 服务前置检查通过")
                return True
            else:
                logger.error(f"❌ Qwen API 服务响应异常 (HTTP {response.status_code})")
                return False
        except Exception as e:
            logger.error(f"❌ Qwen API 服务不可达: {e}")
            return False

    # ==========================================
    # 🌟 小说集故事边界检测 (Novella Collection Boundary Detection)
    # ==========================================
    @staticmethod
    def _is_new_story_start(chapter_name: str, content: str, prev_chapter_name: str = None) -> bool:
        """检测当前章节是否是小说集中新故事的起始。

        通过章节标题模式匹配判断：如果章节名暗示"第一章"或"序言"，
        且不是全书的首个章节，则视为新故事的开始。

        Args:
            chapter_name: 当前章节名称
            content: 当前章节内容
            prev_chapter_name: 上一章节名称（None 表示这是第一个章节）

        Returns:
            True 表示检测到新故事的开始
        """
        if prev_chapter_name is None:
            return False

        # 检测"第一章"、"第1章"、"Chapter 1"、"序章"、"序言"、"楔子"等新故事标志
        new_story_patterns = [
            r'第[一1]章',
            r'序[章言]',
            r'楔子',
            r'(?i)chapter[_ ]?0*1\b',
            r'(?i)prologue',
        ]
        for pattern in new_story_patterns:
            if re.search(pattern, chapter_name):
                return True
            # 也检测内容前100字
            if re.search(pattern, content[:100]):
                return True

        return False

    @staticmethod
    def _find_recap_insert_index(micro_script: list) -> int:
        """Find the insertion index for recap entries.

        Scans the script for the first ``narration`` or ``dialogue`` entry and
        returns its index so that title / subtitle entries at the chapter
        beginning are preserved intact.  Falls back to index 0 when the script
        is empty or contains only header-type entries.
        """
        for i, entry in enumerate(micro_script):
            if entry.get("type") in ("narration", "dialogue"):
                return i
        return 0

    # ==========================================
    # 🎬 阶段一：剧本化与微切片 (Script & Micro-chunking)
    # ==========================================
    def phase_1_generate_scripts(self, input_source, max_chapters=None, is_preview=False):
        """阶段一：编剧期 (Qwen API) - 生成包含chunk_id和停顿时间的微切片剧本

        Args:
            input_source: EPUB文件路径或TXT目录路径
            max_chapters: 最多处理的章节数（None表示全部，试听模式传1）
            is_preview: 是否为试听模式（强制注入摘要、截断前10句）
        """
        logger.info("\n" + "="*50 + "\n🎬 [阶段一] 编剧期 (Qwen API)\n" + "="*50)
        
        pure_mode = self.config.get("pure_narrator_mode", False)

        # 🌟 前置检查：纯净模式下不需要 Qwen API 服务
        if not pure_mode and not self.check_api_connectivity():
            logger.error("❌ Qwen API 服务不可用，阶段一中止。请检查 DASHSCOPE_API_KEY 是否已配置。")
            return False

        # 支持EPUB和TXT两种输入格式
        if input_source.endswith('.epub'):
            chapters = self._extract_epub_chapters(input_source)
            if not chapters:
                logger.error("❌ EPUB 解析失败或无有效文本！")
                return False

        # 🌟 修复：新增支持 WebUI 上传单文件 TXT 模式
        elif os.path.isfile(input_source) and input_source.endswith(('.txt', '.md')):
            try:
                with open(input_source, 'r', encoding='utf-8') as f:
                    chapters = {os.path.splitext(os.path.basename(input_source))[0]: f.read()}
            except UnicodeDecodeError:
                logger.error("❌ 文本读取失败：请确保你的 TXT 文件是标准的 UTF-8 编码！")
                return False
            except OSError as e:
                logger.error(f"❌ 文本文件读取失败: {e}")
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

        # 🌟 试听模式优化：只处理前 max_chapters 个章节，避免全书解析
        if max_chapters is not None:
            chapter_items = list(chapters.items())[:max_chapters]
            chapters = dict(chapter_items)
            logger.info(f"🎧 试听模式：仅处理前 {max_chapters} 个章节")
        
        # 🌟 试听模式核心拦截：只取第一章，且只保留前1000字
        if is_preview:
            first_chap_key = list(chapters.keys())[0]
            first_chap_content = chapters[first_chap_key][:1000]
            chapters = {first_chap_key: first_chap_content}
            logger.info(f"🎧 试听防卡死：已切断全书遍历，仅处理首章前1000字")

        # 🌟 项目级角色库物理隔离：根据输入文件名动态生成 cast_db_path
        project_name = os.path.splitext(os.path.basename(input_source))[0]
        cast_db_path = os.path.join("workspace", f"{project_name}_cast.json")

        director = LLMScriptDirector(
            global_cast=self.config.get("global_cast", {}),
            cast_db_path=cast_db_path,
        )
        prev_chapter_content = None  # 用于存储上一章内容
        failed_chapters = []

        # 🌟 解析用户提供的前情提要（如果有）
        user_recaps = {}
        user_recap_text = self.config.get("user_recaps")
        if user_recap_text:
            user_recaps = self.parse_user_recaps(user_recap_text)
            if user_recaps:
                logger.info(f"📋 检测到用户提供的前情提要，共 {len(user_recaps)} 章")

        # 🌟 获取外脑提供的前情提要字典 (按章节名索引, 如 "Chapter_002")
        custom_recaps = self.config.get("custom_recaps", {})

        story_chapter_index = 0  # 🌟 正文章节计数器，只对正文累加，确保与用户提供的第N章精确对齐
        prev_chapter_name = None  # 🌟 用于小说集边界检测
        for chapter_name, content in chapters.items():

            # 🌟 先判定是否为正文（用于正文计数器累加）
            is_main_text = True
            non_main_keywords = ["版权", "目录", "出版", "ISBN", "序言", "致谢", "前言", "引言", "楔子", "Project Gutenberg"]
            if len(content) < 500 or any(keyword in content[:200] for keyword in non_main_keywords):
                is_main_text = False

            # 辅助防御：如果物理文件名是 000 或 001，且开头没有明确的"第一章"标志，强制视为非正文
            if re.search(r'(?i)chapter_00[01]\b', chapter_name) and not re.search(r'第[一1]章', content[:100]):
                is_main_text = False

            # 🌟 只有正文才累加计数器，确保与外部传入的第N章精确对齐！
            if is_main_text:
                story_chapter_index += 1

            # 🌟 小说集 (Novella Collection) 故事边界检测与上下文重置
            if self._is_new_story_start(chapter_name, content, prev_chapter_name):
                director.reset_context()
                prev_chapter_content = None  # 重置前情提要上下文，防止跨书摘要污染

            prev_chapter_name = chapter_name
            script_path = os.path.join(self.script_dir, f"{chapter_name}_micro.json")
            if os.path.exists(script_path) and not is_preview:
                logger.info(f"⏭️ 微切片剧本已存在，跳过: {chapter_name}")
                # 保留已有章节的文本给下一章用
                prev_chapter_content = content
                continue
                
            logger.info(f"✍️ 正在调用 Qwen-Flash 解析剧本: {chapter_name} (字数: {len(content)})")
            try:
                # 🌟 核心双轨制分流：纯净模式 或 非正文内容，直接走纯净旁白模式（免 LLM）
                if pure_mode or not is_main_text:
                    logger.info(f"⚡ {'纯净旁白模式' if pure_mode else '检测到附属文本(序言/版权)'}，启用免LLM规则解析: {chapter_name}")
                    micro_script = director.generate_pure_narrator_script(content, chapter_prefix=chapter_name)
                else:
                    # 🌟 Qwen-Flash 整章直出，设为 10000 既高效又绝对防止 32K 输出溢出
                    micro_script = director.parse_and_micro_chunk(
                        content, chapter_prefix=chapter_name,
                        max_length=10000  # 🌟 解除 4000 封印，对齐底层引擎的最佳甜点位
                    )
                
                # 验证生成的剧本数据结构
                if not micro_script:
                    logger.error(f"❌ {chapter_name} 生成的微切片剧本为空，跳过该章节")
                    failed_chapters.append(chapter_name)
                    continue
                
                # 🌟 核心逻辑：智能前情提要判断（纯净模式下跳过）
                recap_injected = False
                if not pure_mode:
                    recap_text = None

                    # 🌟 1. 强制最高优先级：只要用户/外脑提供了前情提要，无视章节长度，直接使用！
                    if chapter_name in custom_recaps:
                        recap_text = custom_recaps[chapter_name]
                        logger.info(f"📋 强制使用外脑提供的前情提要: {chapter_name}")
                    elif story_chapter_index in user_recaps:
                        recap_text = user_recaps[story_chapter_index]
                        logger.info(f"📋 强制使用用户提供的前情提要 (匹配正文第 {story_chapter_index} 章): {chapter_name}")
                    
                    # 🌟 2. 如果用户没提供，再去判断是否是正文，以及是否需要大模型自动生成
                    elif self.config.get("enable_recap", True):
                        if not is_main_text:
                            logger.info(f"⏭️ 判定 {chapter_name} 为非正文/短章节，跳过生成前情摘要。")

                        if is_main_text and self.config.get("enable_auto_recap", True) and prev_chapter_content is not None:
                            if len(prev_chapter_content) >= 800:
                                logger.info(f"🔄 正在为 {chapter_name} 生成前情摘要 (Map-Reduce 引擎)...")
                                recap_text = director.generate_chapter_recap(prev_chapter_content)

                    # 🌟 3. 执行提要注入
                    if recap_text:
                        intro_unit = {
                            "chunk_id": f"{chapter_name}_recap_intro",
                            "type": "recap",
                            "speaker": "talkover",
                            "content": "前情提要：",
                            "pause_ms": 500
                        }
                        recap_unit = {
                            "chunk_id": f"{chapter_name}_recap_body",
                            "type": "recap",
                            "speaker": "talkover",
                            "content": recap_text,
                            "pause_ms": 1500
                        }
                        # 安全插入法：扫描第一个 narration/dialogue 位置，保持标题结构完整
                        insert_idx = self._find_recap_insert_index(micro_script)
                        micro_script.insert(insert_idx, intro_unit)
                        micro_script.insert(insert_idx + 1, recap_unit)
                        recap_injected = True

                # 🌟 试听强制注入逻辑（核心）
                # 如果是试听模式，且原本这章没摘要（比如第一章），但用户传了外脑字典，我们就强行借用一条来试听！
                if is_preview and not recap_injected and custom_recaps:
                    borrowed_recap = next(iter(custom_recaps.values()))
                    logger.info(f"🎧 试听连通性测试：强制借用一条前情提要进行 Talkover 音色验证！")
                    intro_unit = {
                        "chunk_id": f"{chapter_name}_recap_intro",
                        "type": "recap",
                        "speaker": "talkover",
                        "content": "前情提要：",
                        "pause_ms": 500
                    }
                    recap_unit = {
                        "chunk_id": f"{chapter_name}_recap_body",
                        "type": "recap",
                        "speaker": "talkover",
                        "content": borrowed_recap,
                        "pause_ms": 1500
                    }
                    # 🌟 安全插入法：扫描第一个 narration/dialogue 位置，保持标题结构完整
                    insert_idx = self._find_recap_insert_index(micro_script)
                    micro_script.insert(insert_idx, intro_unit)
                    micro_script.insert(insert_idx + 1, recap_unit)
                
                # 保存当前章的原始文本，供下一章使用
                prev_chapter_content = content
                
                # 🌟 试听模式极速截断：只保留前 10 句话（包含刚注入的提要）
                if is_preview:
                    micro_script = micro_script[:10]
                
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
                
        # 阶段一完成（Qwen API 无需释放本地内存）

        if failed_chapters:
            logger.warning(f"⚠️ 以下章节处理失败: {', '.join(failed_chapters)}")

        logger.info("✅ 阶段一完成，剧本生成已完毕！")
        return True
    
    # ==========================================
    # 🎧 试听模式：极速通道，只处理前 10 句话
    # ==========================================
    def run_preview_mode(self, input_source: str, preview_text: str = None) -> str:
        """🌟 专属的试听模式：极速通道，测试外脑连通性，只处理前 10 句话

        当 preview_text 非空时，跳过阶段一（LLM 切片），直接使用用户在网页
        上编辑的试听文本构建微切片剧本。

        流程：先完成第一阶段微切片，再从第一章剧本中截取前 10 句，
        写入独立的临时剧本文件（不覆盖原始剧本），直接渲染并压制。
        """
        logger.info("🎧 启动极速试听通道...")

        # 🌟 连通性探针：检查外脑数据是否成功穿透 WebUI 到达底层
        global_cast = self.config.get("global_cast", {})
        custom_recaps = self.config.get("custom_recaps", {})

        if global_cast:
            logger.info(f"✅ 试听连通性测试: 成功接收外脑【角色设定集】 ({len(global_cast)} 个角色)")
        else:
            logger.info(f"ℹ️ 试听连通性测试: 未检测到外脑角色设定，将使用默认分配策略")

        if custom_recaps:
            logger.info(f"✅ 试听连通性测试: 成功接收外脑【前情摘要库】 ({len(custom_recaps)} 章)")
        else:
            logger.info(f"ℹ️ 试听连通性测试: 未检测到外部前情摘要")

        # 临时强制设为极短时长，迫使 CinematicPackager 提前触发导出
        original_duration = self.config["target_duration_min"]
        self.config["target_duration_min"] = 0.5  # 30秒就发版
        preview_script_path = os.path.join(self.script_dir, "_preview_temp_micro.json")

        try:
            # 🌟 如果用户提供了编辑后的试听文本，直接构建微切片，跳过 LLM
            if preview_text and preview_text.strip():
                sentences = re.split(r'(?<=[。！？!?])', preview_text)
                expanded = []
                for s in sentences:
                    expanded.extend(s.split('\n'))
                sentences = [s.strip() for s in expanded if s.strip()]
                preview_script = []
                for i, sent in enumerate(sentences[:10]):
                    preview_script.append({
                        "chunk_id": f"preview_{i:03d}",
                        "type": "narration",
                        "speaker": "narrator",
                        "gender": "unknown",
                        "emotion": "平静",
                        "content": sent,
                        "pause_ms": 300,
                    })
                logger.info(f"🎧 使用用户编辑的试听文本（{len(preview_script)} 句）")
            else:
                # ── 第一阶段：微切片（仅处理第一章，传入 is_preview 标识）──
                self.phase_1_generate_scripts(input_source, is_preview=True)

                # 找到第一个生成的剧本
                script_files = sorted([f for f in os.listdir(self.script_dir) if f.endswith('_micro.json')])
                if not script_files:
                    raise Exception(f"未找到剧本，请检查阶段一是否成功 (script_dir={self.script_dir})")

                first_script_path = os.path.join(self.script_dir, script_files[0])
                with open(first_script_path, 'r', encoding='utf-8') as f:
                    micro_script = json.load(f)

                # 🌟 核心截断：只取前 10 句！
                preview_script = micro_script[:10]

            # 🌟 写入独立的临时预览剧本，不覆盖原始剧本（保护全本压制的断点续传）
            with open(preview_script_path, 'w', encoding='utf-8') as f:
                json.dump(preview_script, f, ensure_ascii=False)

            # ── 第二阶段：仅渲染预览片段的干音 ──
            self._render_script_chunks(preview_script)

            # ── 第三阶段：仅混音预览片段 ──
            self._mix_script_chunks(preview_script)

            # 找到压制出的第一个文件返回给网页
            preview_files = [f for f in os.listdir(self.config["output_dir"]) if f.endswith('.mp3')]
            if preview_files:
                return os.path.join(self.config["output_dir"], sorted(preview_files)[0])
            return None

        finally:
            # 恢复配置以免污染正式的全本压制
            self.config["target_duration_min"] = original_duration
            # 清理临时预览剧本（无论成功/失败都要清理）
            if os.path.exists(preview_script_path):
                os.remove(preview_script_path)

    def _render_script_chunks(self, micro_script: list):
        """渲染指定的微切片列表为干音 WAV 文件（供试听模式直接调用）"""
        from modules.mlx_tts_engine import MLXRenderEngine, group_indices_by_voice_type

        # 构建引擎配置（支持 1.7B Model Pool）
        engine_config = {}
        for key in ("model_path_base", "model_path_design",
                    "model_path_custom", "model_path_fallback"):
            val = self.config.get(key)
            if val:
                engine_config[key] = val

        engine = self._create_tts_engine()

        voice_groups = group_indices_by_voice_type(micro_script)
        for voice_key, indices in voice_groups.items():
            first_item = micro_script[indices[0]]
            group_voice_cfg = self.assets.get_voice_for_role(
                first_item["type"],
                first_item.get("speaker"),
                first_item.get("gender")
            )
            for idx in indices:
                item = micro_script[idx]
                save_path = os.path.join(self.cache_dir, f"{item['chunk_id']}.wav")
                engine.render_dry_chunk(item["content"], group_voice_cfg, save_path)

        if hasattr(engine, 'destroy'):
            engine.destroy()
        del engine

    def _mix_script_chunks(self, micro_script: list):
        """将指定的微切片列表混音压制为 MP3（供试听模式直接调用）"""
        target_min = self.config.get("target_duration_min", 30)
        packager = CinematicPackager(self.config["output_dir"], target_duration_min=target_min)

        if self.config.get("pure_narrator_mode", False):
            ambient_bgm = None
            chime_sound = None
        else:
            ambient_bgm = self.assets.get_ambient_sound(self.config["ambient_theme"])
            chime_sound = self.assets.get_transition_chime()

        packager.process_from_cache(micro_script, self.cache_dir, self.assets, ambient_bgm, chime_sound)

    # ==========================================
    # 🎙️ 阶段二：纯净干音渲染 (Dry Voice Rendering)
    # ==========================================
    def phase_2_render_dry_audio(self):
        """阶段二：录音期 (MLX TTS) - 纯净干音渲染，只产生WAV文件
        
        Uses a "group-by-voice" strategy: chunks sharing the same voice type
        are rendered consecutively to minimise MLX embedding switches.
        """
        logger.info("\n" + "="*50 + "\n🎙️ [阶段二] 录音期 (MLX TTS)\n" + "="*50)

        # 构建引擎配置（支持 1.7B Model Pool）
        engine_config = {}
        for key in ("model_path_base", "model_path_design",
                    "model_path_custom", "model_path_fallback"):
            val = self.config.get(key)
            if val:
                engine_config[key] = val

        engine = self._create_tts_engine()

        # 🔥 预热：在渲染开始前预加载模型，利用 M4 统一内存带宽优势
        warmup_modes = ["preset"]
        if engine_config.get("model_path_base"):
            warmup_modes.append("clone")
        engine.warmup(warmup_modes)
        
        # 全局冷启动标记，引擎刚初始化时必定是冷启动
        is_cold_start = True
        
        script_files = sorted([f for f in os.listdir(self.script_dir)
                               if f.endswith('_micro.json') and not f.startswith('_preview_')])
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
                # 🌟 修复：每个音色组只解析一次 voice_cfg，确保组内所有微切片
                # 使用完全相同的音色配置，杜绝音色在微切片之间切换
                first_item = micro_script[indices[0]]
                group_voice_cfg = self.assets.get_voice_for_role(
                    first_item["type"],
                    first_item.get("speaker"),
                    first_item.get("gender")
                )
                for idx in indices:
                    item = micro_script[idx]
                    save_path = os.path.join(self.cache_dir, f"{item['chunk_id']}.wav")
                    
                    # 断点续传：缓存命中直接跳过，不参与看门狗计时
                    if os.path.exists(save_path):
                        rendered_chunks += 1
                        if rendered_chunks > 0 and rendered_chunks % 50 == 0:
                            logger.info(f"   🎵 进度: {rendered_chunks}/{total_chunks} 片段已渲染(跳过)")
                        continue

                    start_time = time.time()

                    try:
                        success = engine.render_dry_chunk(item["content"], group_voice_cfg, save_path)
                        if not success:
                            logger.error(
                                f"🔇 渲染返回失败: chunk_id={item.get('chunk_id')}, "
                                f"speaker={item.get('speaker')}, "
                                f"content='{item['content'][:50]}...'"
                            )
                    except Exception as e:
                        import traceback
                        logger.error(
                            f"❌ 渲染异常: chunk_id={item.get('chunk_id')}, "
                            f"speaker={item.get('speaker')}, "
                            f"content='{item['content'][:50]}...', "
                            f"error={e}"
                        )
                        logger.error(f"📋 异常堆栈:\n{traceback.format_exc()}")
                        success = False

                    elapsed_time = time.time() - start_time
                    rendered_chunks += 1

                    # 动态看门狗阈值（冷启动120秒，热运行45秒）
                    timeout_threshold = ENGINE_COLD_START_THRESHOLD_SECONDS if is_cold_start else ENGINE_WARM_THRESHOLD_SECONDS

                    if elapsed_time > timeout_threshold:
                        logger.warning(
                            f"🚨 严重警告: 切片 {item.get('chunk_id')} 渲染耗时 "
                            f"{elapsed_time:.1f} 秒！(当前阈值: {timeout_threshold}s)"
                        )
                        # 🔥 销毁超时产生的脏音频，防止污染混音
                        if os.path.exists(save_path):
                            os.remove(save_path)
                            logger.info(f"🗑️ 已销毁超时产生的脏音频: {save_path}")
                        logger.info("🔄 正在触发引擎自愈重置协议...")
                        if hasattr(engine, 'destroy'):
                            engine.destroy()
                        del engine
                        gc.collect()
                        logger.info("✨ 内存已清空，正在重新加载 MLX TTS 引擎...")
                        engine = self._create_tts_engine()
                        logger.info("✅ 引擎热重启完成，恢复生产！")
                        # 重启后的下一个片段又将面临 JIT 编译，重置为冷启动状态
                        is_cold_start = True
                        # 跳过当前失败片段的进度计数，重新渲染
                        rendered_chunks -= 1
                        continue
                    else:
                        # 渲染在阈值内平稳度过，引擎热身完毕，切换为严苛状态
                        is_cold_start = False
                    
                    if rendered_chunks > 0 and rendered_chunks % 50 == 0:
                        logger.info(f"   🎵 进度: {rendered_chunks}/{total_chunks} 片段已渲染")
        
        # 释放 MLX 模型显存
        if hasattr(engine, 'destroy'):
            engine.destroy()
        del engine
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

        # 🌟 全量跳过：如果分卷已全部存在且剧本无更新，直接跳过整个混音阶段
        output_dir = self.config["output_dir"]
        existing_volumes = sorted([f for f in os.listdir(output_dir)
                                   if f.startswith("Audiobook_Part_") and f.endswith(".mp3")])
        script_files = sorted([f for f in os.listdir(self.script_dir)
                               if f.endswith('_micro.json') and not f.startswith('_preview_')])
        if existing_volumes and script_files:
            latest_volume_mtime = max(
                os.path.getmtime(os.path.join(output_dir, f)) for f in existing_volumes
            )
            latest_script_mtime = max(
                os.path.getmtime(os.path.join(self.script_dir, f)) for f in script_files
            )
            if latest_volume_mtime >= latest_script_mtime:
                logger.info(f"⏭️ 检测到 {len(existing_volumes)} 个分卷已存在且剧本无更新，跳过整个混音阶段")
                return

        target_min = self.config.get("target_duration_min", 30)
        packager = CinematicPackager(self.config["output_dir"], target_duration_min=target_min)

        # 🌟 核心拦截：纯净模式下，强行将音效设为 None
        if self.config.get("pure_narrator_mode", False):
            logger.info("🔇 纯净模式已开启：关闭环境背景音与章节过渡音效")
            ambient_bgm = None
            chime_sound = None
        else:
            ambient_bgm = self.assets.get_ambient_sound(self.config["ambient_theme"])
            chime_sound = self.assets.get_transition_chime()
        
        for file in script_files:
            with open(os.path.join(self.script_dir, file), 'r', encoding='utf-8') as f:
                micro_script = json.load(f)
            # 🌟 Pydub 开始组装，此时已经没有大模型在抢占内存了
            packager.process_from_cache(micro_script, self.cache_dir, self.assets, ambient_bgm, chime_sound)
        
        logger.info("🎉 三段式架构全流程完成！全书压制完毕，请前往 output 目录查收。")

    def phase_4_quality_control(self, target_dir=None):
        """阶段四：质检期 (Audio Shield) - 自动扫描并处理爆音

        Args:
            target_dir: 要扫描的目录。默认为 output_dir。
        """
        logger.info("\n" + "="*50 + "\n🔍 [阶段四] 质检期 (Audio Shield)\n" + "="*50)

        # 检查是否有输出文件
        output_dir = target_dir or self.config["output_dir"]
        if not os.path.exists(output_dir):
            logger.error("❌ 未发现输出目录，质检中止。")
            return

        # 自动拉起 GUI，并直接进入扫描模式
        from audio_shield.gui import launch_gui_with_context
        logger.info("🚀 正在启动质检工作台...")

        # 通过封装后的函数启动，自动载入当前项目的 output 目录
        launch_gui_with_context(output_dir, sensitivity=0.4)
    
def main():
    """主函数 - 引入命令行参数"""
    parser = argparse.ArgumentParser(description="CineCast 电影级有声书生产线")
    parser.add_argument("input", nargs="?", default="./input_chapters", help="输入文件(EPUB)或目录(TXT)")
    parser.add_argument("--pure-narrator", action="store_true", help="启用纯净旁白模式(单音色/无背景音/无摘要/免LLM)")
    args = parser.parse_args()

    producer = CineCastProducer()
    producer.config["pure_narrator_mode"] = args.pure_narrator  # 🌟 将命令行参数写入全局配置
    input_source = args.input
    
    if input_source.endswith('.epub') and os.path.exists(input_source):
        logger.info(f"📚 检测到EPUB文件: {input_source}")
    elif os.path.isfile(input_source) and input_source.endswith(('.txt', '.md')):
        logger.info(f"📝 检测到单文件TXT模式: {input_source}")
    elif os.path.isdir(input_source):
        if not os.listdir(input_source):
            logger.warning(f"⚠️ 请先在 {input_source} 文件夹中放入测试用的 .txt 章节！")
            with open(os.path.join(input_source, "第一章_测试.txt"), 'w', encoding='utf-8') as f:
                f.write("第一章 风雪\n1976年\n夜幕降临港口。\"你相信命运吗？\"老渔夫问。\n\"我不信。\"年轻人回答。")
        logger.info(f"📝 使用TXT目录模式: {input_source}")
    else:
        # 回退到默认TXT目录模式
        input_source = "./input_chapters"
        os.makedirs(input_source, exist_ok=True)
        if not os.listdir(input_source):
            logger.warning(f"⚠️ 请先在 {input_source} 文件夹中放入测试用的 .txt 章节！")
            with open(os.path.join(input_source, "第一章_测试.txt"), 'w', encoding='utf-8') as f:
                f.write("第一章 风雪\n1976年\n夜幕降临港口。\"你相信命运吗？\"老渔夫问。\n\"我不信。\"年轻人回答。")
        logger.info(f"📝 使用TXT目录模式: {input_source}")
    
    try:
        # 严格的三段式串行处理，彻底切断内存重叠
        if producer.phase_1_generate_scripts(input_source):
            producer.phase_2_render_dry_audio()

            # 🛡️ 新增：阶段二后质检（干音质检）
            logger.info("🛡️ 进入干音缓存质检阶段...")
            producer.phase_4_quality_control(target_dir=producer.cache_dir)

            producer.phase_3_cinematic_mix()

            # 🛡️ 新增：阶段三后质检（成品发布质检）
            logger.info("🛡️ 进入成品发布质检阶段...")
            producer.phase_4_quality_control(target_dir=producer.config["output_dir"])
    except Exception as e:
        logger.error(f"💥 三段式架构执行失败: {e}")

if __name__ == "__main__":
    main()