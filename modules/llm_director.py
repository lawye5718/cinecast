#!/usr/bin/env python3
"""
CineCast 大模型剧本预处理器
阶段一：剧本化与微切片 (Script & Micro-chunking)
实现宏观剧本解析 -> 自动展开为微切片剧本
"""

import json
import re
import logging
import os
import tempfile
import time
from typing import List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


def atomic_json_write(path: str, data, **kwargs) -> None:
    """Atomic JSON write: write to a temporary file first, then replace.

    This prevents JSON corruption if the process crashes mid-write.
    """
    dir_name = os.path.dirname(path) or "."
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("indent", 2)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **kwargs)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def repair_json_array(raw: str) -> Optional[List[Dict]]:
    """Attempt to repair a truncated or malformed JSON array.

    Tries progressively more aggressive strategies:
    1. Strip trailing garbage after the last ``}`` and close the array.
    2. Use regex to salvage individual JSON objects.

    Returns ``None`` if nothing can be recovered.
    """
    # Strategy 1: find last complete object and close the array
    raw = raw.strip()
    if raw.startswith("["):
        last_brace = raw.rfind("}")
        if last_brace > 0:
            candidate = raw[: last_brace + 1].rstrip().rstrip(",") + "\n]"
            try:
                result = json.loads(candidate)
                if isinstance(result, list) and result:
                    return result
            except json.JSONDecodeError:
                pass

    # Strategy 2: regex salvage individual entries
    return salvage_json_entries(raw)


def _extract_fields_from_object(obj_text: str) -> Optional[Dict]:
    """Extract known fields from a single JSON object text in any order.

    Uses individual per-field regexes so that field ordering does not matter.
    Returns a dict with defaults for missing fields, or ``None`` if neither
    ``speaker`` nor ``content`` could be found.
    """
    field_re = re.compile(r'"(\w+)"\s*:\s*"([^"]*)"')
    fields: Dict[str, str] = {}
    for m in field_re.finditer(obj_text):
        fields[m.group(1)] = m.group(2)

    # Map known aliases
    speaker = fields.get("speaker", "")
    content = fields.get("content", "")
    if not speaker and not content:
        return None

    return {
        "type": fields.get("type", "narration") or "narration",
        "speaker": speaker or "narrator",
        "gender": fields.get("gender", "unknown") or "unknown",
        "emotion": fields.get("emotion") or fields.get("instruct") or "平静",
        "content": content or "",
    }


def salvage_json_entries(raw: str) -> Optional[List[Dict]]:
    """Use regex to extract valid script entries from broken JSON text.

    Each entry is expected to have at least ``speaker`` and ``content`` fields.
    Uses order-independent field extraction so that reordered or extra-spaced
    LLM output can still be recovered.
    """
    # Find all brace-delimited object candidates
    obj_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL)
    entries = []
    for m in obj_pattern.finditer(raw):
        entry = _extract_fields_from_object(m.group(0))
        if entry and entry.get("content"):
            entries.append(entry)

    if not entries:
        # Looser pattern: just find speaker + content anywhere
        loose = re.compile(
            r'"speaker"\s*:\s*"([^"]*)"\s*[,}].*?"content"\s*:\s*"([^"]*)"',
            re.DOTALL,
        )
        for m in loose.finditer(raw):
            entries.append({
                "type": "narration",
                "speaker": m.group(1) or "narrator",
                "gender": "unknown",
                "emotion": "平静",
                "content": m.group(2) or "",
            })

    return entries if entries else None


def merge_consecutive_narrators(script: List[Dict], max_chars: int = 800) -> List[Dict]:
    """Merge consecutive narrator entries that share the same emotion.

    This reduces TTS startup overhead and avoids fragmented short sentences
    that cause jarring tonal shifts.
    """
    if not script:
        return script

    merged: List[Dict] = []
    for entry in script:
        if (
            merged
            and entry.get("speaker") == "narrator"
            and merged[-1].get("speaker") == "narrator"
            and entry.get("emotion", "平静") == merged[-1].get("emotion", "平静")
            and entry.get("type") == merged[-1].get("type")
            and len(merged[-1].get("content", "")) + len(entry.get("content", "")) <= max_chars
        ):
            merged[-1]["content"] = merged[-1]["content"] + entry["content"]
            # Keep the longer pause
            merged[-1]["pause_ms"] = max(
                merged[-1].get("pause_ms", 0), entry.get("pause_ms", 0)
            )
        else:
            merged.append(entry.copy())

    return merged

class LLMScriptDirector:
    # 🌟 高阶角色音色映射表 (Voice Archetype Mapping)
    VOICE_ARCHETYPES = {
        "intellectual": "Clear, articulate, mid-range voice, steady pacing, calm and intellectual.",
        "villain_sly": "Slightly nasal, fast-paced voice, bright tone, with a hint of sarcasm.",
        "melancholic": "Breathier, soft voice, melancholic undertones, slow and emotional.",
        "authoritative": "Resonant, deep baritone, slow and authoritative, gravelly texture.",
        "innocent": "Bright, high-pitched, energetic and innocent, clear enunciation.",
    }

    def __init__(self, api_key=None, model_name=None, base_url=None, global_cast=None, cast_db_path=None, **kwargs):
        if kwargs:
            logger.warning(f"⚠️ LLMScriptDirector 收到未识别的参数（已忽略）: {list(kwargs.keys())}")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model_name = model_name or "qwen-flash"
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # 🌟 优化：使用标准 OpenAI SDK 客户端，支持用户自定义 LLM 配置
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,
        )
        
        self.max_chars_per_chunk = 150 # 🎯 修改点：微切片红线调整为 150 字
        self.pure_narrator_chunk_limit = 100  # 纯净旁白模式切片上限（更长更流畅）
        self.global_cast = global_cast or {}  # 🌟 外脑全局角色设定集
        
        # Context sliding window state
        self._prev_characters: List[str] = []
        self._prev_tail_entries: List[Dict] = []
        self._local_session_cast: Dict[str, str] = {}  # 🌟 局部会话角色音色表（跨 chunk 音色一致性）

        # 🌟 音色一致性持久化 (Voice Consistency Persistence)
        self.cast_db_path = cast_db_path or os.path.join("workspace", "cast_profiles.json")
        self.cast_profiles: Dict[str, Dict] = self._load_cast_profiles()
        
        # 测试 Qwen API 连接
        self._test_api_connection()

    # ------------------------------------------------------------------
    # 🌟 音色一致性持久化 (Voice Consistency Persistence)
    # ------------------------------------------------------------------

    def _load_cast_profiles(self) -> Dict[str, Dict]:
        """加载已保存的角色音色库"""
        if os.path.exists(self.cast_db_path):
            try:
                with open(self.cast_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"⚠️ 加载角色音色库失败: {e}")
        return {}

    def _save_cast_profile(self, name: str, gender: str, description: str) -> None:
        """发现新角色或更新角色时持久化"""
        if name not in self.cast_profiles:
            self.cast_profiles[name] = {
                "gender": gender,
                "voice_instruction": description,
            }
            os.makedirs(os.path.dirname(self.cast_db_path) or ".", exist_ok=True)
            atomic_json_write(self.cast_db_path, self.cast_profiles)

    def _update_cast_db(self, script_list: List[Dict]) -> None:
        """解析完一个 chunk 后，提取新角色并持久化"""
        updated = False
        for item in script_list:
            speaker = item.get("speaker")
            if not speaker or speaker == "narrator":
                continue
            emotion = item.get("emotion", "")
            gender = item.get("gender", "unknown")
            # 提取括号内的英文描述（使用正则匹配更可靠）
            if speaker not in self.cast_profiles:
                m = re.search(r'\(([^)]+)\)', emotion)
                if m:
                    self.cast_profiles[speaker] = {
                        "gender": gender,
                        "voice_instruction": m.group(1),
                    }
                    updated = True

        if updated:
            os.makedirs(os.path.dirname(self.cast_db_path) or ".", exist_ok=True)
            atomic_json_write(self.cast_db_path, self.cast_profiles)

    # ------------------------------------------------------------------
    # 🌟 高阶角色音色映射表 Prompt 生成
    # ------------------------------------------------------------------

    def _get_archetype_prompt(self) -> str:
        """生成注入 System Prompt 的音色映射指南"""
        guidelines = "\n".join(
            [f"  - {k}: {v}" for k, v in self.VOICE_ARCHETYPES.items()]
        )
        return (
            "\n【音色设计参考手册】\n"
            "当为新角色生成 (Acoustic Description) 时，请优先参考以下文学原型描述词：\n"
            f"{guidelines}\n"
        )

    # ------------------------------------------------------------------
    # 🌟 小说集上下文重置 (Novella Collection Context Reset)
    # ------------------------------------------------------------------

    def reset_context(self) -> None:
        """强制重置滑动窗口，用于小说集中的新故事"""
        self._prev_characters = []
        self._prev_tail_entries = []
        self._local_session_cast = {}
        logger.info("♻️ 检测到故事边界，导演引擎已重置上下文。")

    def _test_api_connection(self):
        """测试 LLM API 服务连接"""
        if not self.api_key:
            logger.warning("⚠️ 未设置 API Key，智能配音模式将无法使用大模型服务。")
            return False
        try:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "请回复\u201c连接正常\u201d四个字。"}],
                max_tokens=16,
            )
            logger.info("✅ LLM API 服务连接正常")
            return True
        except Exception as e:
            logger.warning(f"❌ 无法连接到 LLM API 服务: {e}")
            return False
    
    def _chunk_text_for_llm(self, text: str, max_length: int = 8000) -> List[str]:
        """🌟 防止章节过长，按段落切分为安全大小给 LLM 处理
        
        虽然上下文窗口 1M，但输出限制 32K token，为防止 JSON 膨胀截断，
        建议单块 8000 字符。超过 max_length 的章节会尽量分成大小相近的几部分，
        避免出现一部分 7800 字而另一部分只有 800 字的不均匀切割。
        """
        total_len = len(text)
        if total_len <= max_length:
            return [text] if text.strip() else []

        # 计算需要几块才能让每块大小尽量均匀
        num_parts = (total_len + max_length - 1) // max_length
        target_size = min(total_len // num_parts, max_length)

        paragraphs = text.split('\n')
        chunks, current_chunk = [], ""
        for para in paragraphs:
            if not para.strip():
                continue
            if len(current_chunk) + len(para) > target_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para + "\n"
            else:
                current_chunk += para + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def verify_integrity(self, original_text: str, script_list: List[Dict]) -> bool:
        """🌟 内容完整性全面监控：对比原文与剧本字数，分级预警

        分级策略：
        - 保留率 < 90%：严重丢失，记录详细差异，返回 False（触发旁白回退）
        - 保留率 < 99%：轻微偏差，记录详细差异到日志，返回 True
        - 保留率 >= 99%：完全达标，静默通过

        Args:
            original_text: 原始输入文本
            script_list: LLM 解析后的剧本列表

        Returns:
            True 表示内容完整性达标，False 表示内容丢失严重
        """
        if not original_text or not script_list:
            return True
        content_text = "".join([item.get("content", "") for item in script_list])
        original_len = len(original_text.strip())
        if original_len == 0:
            return True
        content_len = len(content_text)
        ratio = content_len / original_len
        diff_chars = original_len - content_len

        if ratio < 0.9:
            logger.error(
                f"🚨 内容丢失严重！原文{original_len}字，"
                f"解析后仅{content_len}字 (保留率{ratio:.1%})"
            )
            logger.error(
                f"📊 详细差异: 原文字数={original_len}, 剧本字数={content_len}, "
                f"缺失字数={diff_chars}, 保留率={ratio:.2%}"
            )
            self._log_content_diff(original_text.strip(), content_text)
            return False

        if ratio < 0.99:
            logger.warning(
                f"⚠️ 内容差异检测: 原文{original_len}字，剧本{content_len}字 "
                f"(保留率{ratio:.2%}, 缺失{diff_chars}字)"
            )
            self._log_content_diff(original_text.strip(), content_text)

        logger.info(f"✅ 内容完整性校验通过 (保留率{ratio:.1%})")
        return True

    def _log_content_diff(self, original_text: str, script_text: str) -> None:
        """将原文与剧本的段落级差异写入日志，便于定位丢失内容。"""
        orig_paras = [p.strip() for p in original_text.split('\n') if p.strip()]
        if not orig_paras:
            return
        missing_paras = []
        for i, para in enumerate(orig_paras):
            check_prefix = para[:20] if len(para) > 20 else para
            if check_prefix and check_prefix not in script_text:
                missing_paras.append((i + 1, para[:80]))
        if missing_paras:
            logger.warning(f"📝 疑似缺失段落 ({len(missing_paras)}/{len(orig_paras)}段):")
            for para_num, preview in missing_paras[:10]:
                logger.warning(f"   第{para_num}段: {preview}...")
            if len(missing_paras) > 10:
                logger.warning(f"   ... 及其余 {len(missing_paras) - 10} 段")
    
    def generate_pure_narrator_script(self, text: str, chapter_prefix: str = "chunk") -> List[Dict]:
        """
        纯净旁白模式专用的剧本生成器（绕过LLM，秒级生成，100%忠实原著）
        纯净旁白模式下，切片长度放宽到 100 字左右，减少切片数量，提升朗读流畅度。
        """
        micro_script = []
        chunk_id = 1

        # 🌟 纯净旁白模式下，切片上限放宽到 ~100 字
        pure_chunk_limit = self.pure_narrator_chunk_limit

        # 1. 按段落切分
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        for p_idx, para in enumerate(paragraphs):
            # 2. 按长句标点切分（保留标点）
            sentences = re.split(r'([。！？；.!?;])', para)

            temp_sentence = ""
            for part in sentences:
                if not part.strip() and not re.match(r'[。！？；.!?;]', part):
                    continue

                if re.match(r'^[。！？；.!?;]$', part.strip()):
                    temp_sentence += part

                    # 3. 如果单句仍然超长，启动逗号/顿号的次级切分
                    if len(temp_sentence) > pure_chunk_limit:
                        sub_parts = re.split(r'([，、：,:])', temp_sentence)
                        sub_temp = ""
                        for sub in sub_parts:
                            if re.match(r'^[，、：,:]$', sub):
                                sub_temp += sub
                                pause = self._calculate_pause(sub_temp, False)
                                micro_script.append({
                                    "chunk_id": f"{chapter_prefix}_{chunk_id:05d}",
                                    "type": "narration",
                                    "speaker": "narrator",
                                    "gender": "male",
                                    "emotion": "平静",
                                    "content": sub_temp.strip(),
                                    "pause_ms": pause
                                })
                                chunk_id += 1
                                sub_temp = ""
                            else:
                                sub_temp += sub
                        if sub_temp.strip():
                            pause = self._calculate_pause(sub_temp, p_idx == len(paragraphs)-1)
                            micro_script.append({
                                "chunk_id": f"{chapter_prefix}_{chunk_id:05d}",
                                "type": "narration",
                                "speaker": "narrator",
                                "gender": "male",
                                "emotion": "平静",
                                "content": sub_temp.strip(),
                                "pause_ms": pause
                            })
                            chunk_id += 1
                    else:
                        # 正常长度的句子直接推入
                        pause = self._calculate_pause(temp_sentence, p_idx == len(paragraphs)-1)
                        micro_script.append({
                            "chunk_id": f"{chapter_prefix}_{chunk_id:05d}",
                            "type": "narration",
                            "speaker": "narrator",
                            "gender": "male",
                            "emotion": "平静",
                            "content": temp_sentence.strip(),
                            "pause_ms": pause
                        })
                        chunk_id += 1
                    temp_sentence = ""
                else:
                    temp_sentence += part

            # 处理段落末尾没有标点的残留部分
            if temp_sentence.strip():
                pause = self._calculate_pause(temp_sentence, p_idx == len(paragraphs)-1)
                micro_script.append({
                    "chunk_id": f"{chapter_prefix}_{chunk_id:05d}",
                    "type": "narration",
                    "speaker": "narrator",
                    "gender": "male",
                    "emotion": "平静",
                    "content": temp_sentence.strip(),
                    "pause_ms": pause
                })
                chunk_id += 1

        return micro_script

    def parse_and_micro_chunk(self, text: str, chapter_prefix: str = "chunk", max_length: int = 8000) -> List[Dict]:
        """宏观剧本解析 -> 自动展开为微切片剧本
        
        Args:
            text: 待处理的章节文本
            chapter_prefix: 章节名称前缀，用于避免文件名冲突
            max_length: LLM 单次处理的最大字符数上限，默认8000
        """
        # 第一步：生成宏观剧本
        macro_script = self.parse_text_to_script(text, max_length=max_length)

        # 🛡️ 剧本监控：内容差异超过90%时，自动回退旁白模式渲染原文
        content_text = "".join(item.get("content", "") for item in macro_script)
        original_len = len(text.strip())
        if original_len > 0:
            ratio = len(content_text) / original_len
            if ratio < 0.9:
                logger.warning(
                    f"🛡️ 剧本内容保留率过低 ({ratio:.1%})，"
                    f"自动切换旁白模式渲染原文: {chapter_prefix}"
                )
                return self.generate_pure_narrator_script(text, chapter_prefix=chapter_prefix)

        micro_script = []
        chunk_id = 1
        
        # 适当放宽微切片红线，避免正常句子被无故切断
        smart_chunk_limit = max(self.max_chars_per_chunk, 150) # 🎯 修改点：从 90 改为 150
        
        for unit in macro_script:
            content = unit.get("content", "")
            if not content or not content.strip():
                continue

            # 🌟 修复：实施智能微切片，优先按大标点切分
            raw_sentences = re.split(r'([。！？；.!?;])', content)
            chunks = []
            temp = ""
            for part in raw_sentences:
                if not part.strip():
                    continue
                if re.match(r'^[。！？；.!?;]$', part.strip()):
                    temp += part
                    # 如果这句长度正常，直接加入（不再被逗号切碎）
                    if len(temp) <= smart_chunk_limit:
                        chunks.append(temp)
                        temp = ""
                    else:
                        # 🚨 只有当单句超长时，才启动逗号/顿号的次级切分
                        sub_parts = re.split(r'([，、：,:])', temp)
                        sub_temp = ""
                        for sub in sub_parts:
                            if re.match(r'^[，、：,:]$', sub):
                                sub_temp += sub
                                chunks.append(sub_temp)
                                sub_temp = ""
                            else:
                                sub_temp += sub
                        if sub_temp:
                            chunks.append(sub_temp)
                        temp = ""
                else:
                    temp += part
            if temp: chunks.append(temp)
            
            # 清理空块并计算停顿
            valid_chunks = [c.strip() for c in chunks if c.strip()]

            # 🌟 兜底逻辑：如果正则切分后无有效块，按硬切
            if not valid_chunks and content.strip():
                hard_cut_chunk_size = smart_chunk_limit
                stripped = content.strip()
                valid_chunks = [
                    stripped[i:i + hard_cut_chunk_size]
                    for i in range(0, len(stripped), hard_cut_chunk_size)
                ]
                logger.warning(
                    f"⚠️ 正则切分无结果，已按每{hard_cut_chunk_size}字硬切: "
                    f"'{content[:30]}...'"
                )

            for idx, chunk in enumerate(valid_chunks):
                is_para_end = (idx == len(valid_chunks) - 1)
                pause_ms = self._calculate_pause(chunk, is_para_end)
                
                # 🌟 修复：将章节名称前缀加入ID，杜绝文件覆盖！
                micro_script.append({
                    "chunk_id": f"{chapter_prefix}_{chunk_id:05d}",
                    "type": unit["type"],
                    "speaker": unit["speaker"],
                    "gender": unit.get("gender", "male"),
                    "content": chunk,
                    "pause_ms": pause_ms
                })
                chunk_id += 1
                
        return micro_script

    def _calculate_pause(self, chunk_text: str, is_para_end: bool) -> int:
        """提前计算好物理停顿时间"""
        if is_para_end: return 1000
        if chunk_text.endswith(('。', '！', '？', '.', '!', '?')): return 600
        elif chunk_text.endswith(('；', ';')): return 400
        elif chunk_text.endswith(('，', '、', ',', '：', ':')): return 250
        else: return 100

    @staticmethod
    def _normalize_text(text: str) -> str:
        """将数字和常见符号转换为中文可读形式，防止 TTS 误读。

        采用逐字转换策略，确保 TTS 朗读一致性。

        Examples:
            "10%" -> "百分之一零"
            "100" -> "一零零"
            "3.14" -> "三点一四"
        """
        _DIGIT_MAP = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
        }

        def _digits_to_chinese(m: re.Match) -> str:
            """Convert a matched digit string to simple Chinese reading."""
            s = m.group(0)
            return ''.join(_DIGIT_MAP.get(c, c) for c in s)

        # 百分号：10% -> 百分之十, 12.5% -> 百分之一二点五
        def _percent_repl(m: re.Match) -> str:
            num_str = m.group(1)
            if '.' in num_str:
                int_part, dec_part = num_str.split('.', 1)
                cn_int = ''.join(_DIGIT_MAP.get(c, c) for c in int_part)
                cn_dec = ''.join(_DIGIT_MAP.get(c, c) for c in dec_part)
                return f'百分之{cn_int}点{cn_dec}'
            cn = ''.join(_DIGIT_MAP.get(c, c) for c in num_str)
            return f'百分之{cn}'

        text = re.sub(r'(\d+(?:\.\d+)?)%', _percent_repl, text)

        # 小数：3.14 -> 三点一四
        def _decimal_repl(m: re.Match) -> str:
            integer_part = m.group(1)
            decimal_part = m.group(2)
            cn_int = ''.join(_DIGIT_MAP.get(c, c) for c in integer_part)
            cn_dec = ''.join(_DIGIT_MAP.get(c, c) for c in decimal_part)
            return f'{cn_int}点{cn_dec}'

        text = re.sub(r'(\d+)\.(\d+)', _decimal_repl, text)

        # 纯整数序列：连续数字 -> 逐字转换
        text = re.sub(r'\d+', _digits_to_chinese, text)

        return text

    def parse_text_to_script(self, text: str, max_length: int = 8000) -> List[Dict]:
        """阶段一：宏观剧本解析 (Qwen-Flash 高效并发版)

        虽然 Qwen-Flash 拥有 1M token 上下文，但输出限制 32K token。
        为防止 JSON 膨胀截断，将切片长度调整为 8000 字符。

        Args:
            text: 待处理的章节文本
            max_length: LLM 单次处理的最大字符数上限，默认8000
        """
        logger.info(f"🚀 启动 {self.model_name} 剧本解析，当前章节字数: {len(text)}")

        # 🌟 Qwen-Flash 拥有 1M 超大上下文，整章直出，仅超长章节才切分
        text_chunks = self._chunk_text_for_llm(text, max_length=max_length)
        full_script = []
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"   🧠 正在解析剧情片段 {i+1}/{len(text_chunks)}...")
            
            # Build context from previous chunk
            context_parts: List[str] = []
            if self._prev_characters:
                context_parts.append(
                    "前一段出场角色: " + ", ".join(self._prev_characters)
                )
            if self._prev_tail_entries:
                try:
                    tail_json = json.dumps(
                        self._prev_tail_entries, ensure_ascii=False
                    )
                    context_parts.append(
                        "\nPrevious section ended with:\n" + tail_json
                    )
                except Exception:
                    pass
            
            chunk_script = self._request_llm(chunk, context="\n".join(context_parts) if context_parts else None)
            
            # Update sliding window state
            if chunk_script:
                speakers = {
                    e.get("speaker")
                    for e in chunk_script
                    if e.get("speaker") and e.get("speaker") != "narrator"
                }
                if speakers:
                    self._prev_characters = list(speakers)
                self._prev_tail_entries = chunk_script[-3:]

                # 🌟 音色一致性防护：记录角色的音色描述到局部会话角色表
                for e in chunk_script:
                    speaker = e.get("speaker")
                    emotion = e.get("emotion", "")
                    if speaker and speaker != "narrator" and emotion:
                        if speaker not in self._local_session_cast:
                            self._local_session_cast[speaker] = emotion

                # 🌟 音色一致性持久化：将新角色音色写入 JSON 角色库
                self._update_cast_db(chunk_script)
            
            full_script.extend(chunk_script)

            # 云端 API 的频率限制由 _request_llm 内部的 429 退避逻辑自动控制，无需人为节流
        
        # 🌟 优化：移除 merge_consecutive_narrators 调用。
        # 因为 parse_and_micro_chunk 会对结果进行严格的 60 字微切片，
        # 合并后的 800 字长文本会被立即碾碎，属于无谓的算力浪费。
        
        # 如果解析结果为空，直接报错退出
        if not full_script or len(full_script) == 0:
            raise RuntimeError("❌ 剧本解析结果为空，请检查输入文本和大模型服务是否正常。")

        # 🌟 内容完整性守门员：检测 LLM 是否严重删节内容
        if not self.verify_integrity(text, full_script):
            logger.warning("⚠️ 内容完整性校验未通过，请检查大模型输出质量。")
            logger.error("❌ 内容完整性低。建议降低 parse_and_micro_chunk() 的 max_length 参数后重试。")
            
        return full_script
    
    def generate_chapter_recap(self, prev_chapter_text: str) -> str:
        """
        🌟 前情摘要引擎 (Qwen-Flash 超大上下文版)
        利用 Qwen-Flash 的 1M 超大上下文，直接整章传入生成摘要，
        无需 Map-Reduce 分块处理。
        """
        # 1. 基础清理
        text = prev_chapter_text.strip()
        if not text:
            return ""

        logger.info(f"🚀 启动 {self.model_name} 前情摘要生成，上一章字数: {len(text)}")

        # 直接生成终极摘要 + 悬念钩子（Qwen 1M 上下文足以容纳整章内容）
        reduce_prompt = (
            '你是一位顶级的有声书剧本编辑和悬疑大师。'
            '请根据提供的上一章内容，写一段不超过100字的\u201c前情摘要\u201d。'
            '绝对纪律：'
            '1. 语言必须高度凝练，具有美剧片头的电影感（\u201cPreviously on...\u201d的风格）。'
            '2. 只保留最具张力的剧情矛盾。'
            '3. 最后一句必须是一个引出下一章的\u201c悬念钩子\u201d。'
            '4. 绝对不要输出\u201c前情提要：\u201d这样的标题，直接输出正文。'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": reduce_prompt},
                    {"role": "user", "content": f"上一章内容：\n{text}"}
                ],
                stream=False,
                temperature=0.5,
                top_p=0.8,
                max_tokens=8192,
            )
            recap_result = response.choices[0].message.content.strip()

            # 清理大模型可能违规加上的前缀
            recap_result = re.sub(r'^(前情提要|前情摘要|回顾|摘要)[:：]\s*', '', recap_result)
            return recap_result
        except Exception as e:
            logger.error(f"终极摘要生成失败: {e}")
            return ""
    
    def _request_llm(self, text_chunk: str, *, context: Optional[str] = None) -> List[Dict]:
        """向 Qwen API 发送单个文本块请求

        Args:
            text_chunk: The raw text to convert into a script.
            context: Optional sliding-window context from the previous chunk
                     (character list + tail entries) to maintain consistency.
        """
        # 🌟 防幻觉加固：定义 Qwen3-TTS 官方支持的感情子集，防止模型乱写
        EMOTION_SET = "平静, 愤怒, 悲伤, 喜悦, 恐惧, 惊讶, 沧桑, 柔和, 激动, 嘲讽, 哽咽, 冰冷, 狂喜"

        # 🌟 防幻觉加固：高精度有声书剧本转换接口 System Prompt
        system_prompt = f"""你是一个高精度的有声书剧本转换接口。
任务：将输入文本逐句解析为 JSON 数组格式。
核心规则：
1. 完整性：原文必须被完全保留，严禁删减。
2. 连贯性原则（最核心指令）：为保证有声书朗读的流畅感，对于同一角色的连续多句台词，或连续的一整段旁白，在总字数不超过 150 字的情况下，必须合并在同一个 JSON 对象内！绝对不允许把一个角色的一句完整的话切碎！
3. 边界切分：只有当说话人发生改变（例如从角色A转为角色B，或角色转为旁白），或者单条文本长度超过 150 字时，才新建一个 JSON 对象。
4. 根节点约束：必须是标准的 JSON 数组（以 `[` 开头）。
5. 字段要求：包含 type, speaker, gender, emotion, content 字段。

【🚨 防截断死亡红线】
请秉持极度的耐心，逐字逐句解析直到最后，切忌过度碎片化！
"""

        # 🌟 优化 Few-Shot，示范正确的合并保留行为
        one_shot_example = """
【输入】：
"你好啊年轻人，这海风可真够冷的。"老渔夫紧紧裹了裹大衣，叹了口气，"昨晚的暴风雪差点把我的船给掀翻了。"
【输出】：
[
  {"type": "dialogue", "speaker": "老渔夫", "gender": "male", "emotion": "沧桑", "content": "你好啊年轻人，这海风可真够冷的。"},
  {"type": "narration", "speaker": "narrator", "gender": "male", "emotion": "平静", "content": "老渔夫紧紧裹了裹大衣，叹了口气，"},
  {"type": "dialogue", "speaker": "老渔夫", "gender": "male", "emotion": "后怕", "content": "昨晚的暴风雪差点把我的船给掀翻了。"}
]
"""

        # 🌟 全局选角纪律注入：如果有外脑提供的角色白名单，追加到 system_prompt
        if self.global_cast:
            cast_names = list(self.global_cast.keys())
            cast_info_parts = []
            for name, info in self.global_cast.items():
                if isinstance(info, dict):
                    g = info.get("gender", "unknown")
                    cast_info_parts.append(f'"{name}"(gender={g})')
                else:
                    cast_info_parts.append(f'"{name}"')
            cast_listing = ", ".join(cast_info_parts)
            system_prompt += f"""

        【全局选角纪律（Cast Whitelist）】
        - 以下是本书的官方角色名单（标准名）：{cast_listing}
        - 你在 speaker 字段中使用的角色名，必须严格使用上述标准名！
        - 严禁自行发明或使用任何不在名单中的角色名！
        - 如果遇到名单外的龙套角色，统一使用 "路人" 作为 speaker。
        - 如果角色不在名单中，请在该角色的 emotion 字段中额外生成一个 10 词以内的英文音色描述（如：A deep, husky voice），以便 TTS 引擎进行音色设计。
        """

        # 🌟 Qwen3-TTS 音色映射指南注入（动态使用 VOICE_ARCHETYPES）
        system_prompt += self._get_archetype_prompt()

        # 🌟 音色一致性防护：注入持久化角色音色库中的已知角色
        if self.cast_profiles:
            known_cast_str = ", ".join(
                [f"{k}({v.get('gender', 'unknown')})" for k, v in self.cast_profiles.items()]
            )
            system_prompt += f"""

        【已知角色音色库（Persistent Cast DB）】
        以下角色在之前的章节中已确定音色，请严格复用：{known_cast_str}
        """

        # 🌟 音色一致性防护：注入上一 chunk 中已确定的音色描述
        if self._local_session_cast:
            cast_desc_parts = []
            for name, desc in self._local_session_cast.items():
                cast_desc_parts.append(f'"{name}": "{desc}"')
            cast_desc_listing = ", ".join(cast_desc_parts)
            system_prompt += f"""

        【角色音色锁定（Voice Lock）】
        以下角色在前文中已确定音色，请严格复用，禁止更改：
        {cast_desc_listing}
        """

        # 🌟 文本预处理：数字/符号规范化
        text_chunk = self._normalize_text(text_chunk)

        # 🌟 防幻觉加固：将 ASCII 双引号替换为中文双引号，避免与 JSON 结构冲突
        # 先处理成对的 ASCII 引号，再将剩余的散引号统一替换以消除 JSON 解析干扰
        text_chunk = re.sub(
            r'"([^"]*)"',
            lambda m: '\u201c' + m.group(1) + '\u201d',
            text_chunk,
        )
        text_chunk = text_chunk.replace('"', '\u2018')

        # 🌟 模型状态监控与 Debug 提示
        input_len = len(text_chunk) + (len(context) if context else 0)
        logger.info(f"🚀 模型: {self.model_name} | 发起请求，估计上下文长度: {input_len} 字符")

        # 🌟 Qwen API 使用 1M 上下文窗口，最大输出 32K token

        # 🌟 防幻觉加固：结构化 User Prompt（使用温和的任务描述，避免触发内容安全过滤）
        user_content = "请将以下小说文本转换为标准 JSON 数组格式（最外层为数组），用于有声书制作。\n\n"

        if context:
            user_content += f"上下文参考：\n{context}\n\n"

        user_content += f"待处理原文：\n{text_chunk}"

        messages = [
            {"role": "system", "content": system_prompt + "\n示例参考：" + one_shot_example},
            {"role": "user", "content": user_content}
        ]

        logger.info(f"🚀 发起 {self.model_name} 解析请求 | 原文字数: {len(text_chunk)}")

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # 🌟 优化：使用原生的 OpenAI SDK 发起请求
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    temperature=0.1,
                    max_tokens=32000,
                )

                full_content = ""

                # 🌟 优化：优雅的流式读取，没有任何阻碍速度的 sleep
                for chunk in completion:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        full_content += delta.content

                content = full_content.strip()

                # 🌟 清理 Markdown 标记
                content = content.replace('\t', ' ').replace('\r', '')
                content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE)
                content = re.sub(r'\s*```$', '', content)

                try:
                    script = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("⚠️ JSON 解析失败，尝试修复截断的 JSON ...")
                    script = repair_json_array(content)
                    if script is None:
                        # 【终极降级 1】：JSON彻底损坏，直接拿原文本做旁白
                        logger.warning("⚠️ JSON彻底损坏，启用终极降级方案：原文本作为旁白。")
                        return self._validate_script_elements([
                            {"type": "narration", "speaker": "narrator", "content": text_chunk}
                        ])
                    return self._validate_script_elements(script)

                if isinstance(script, list):
                    return self._validate_script_elements(script)

                if isinstance(script, dict):
                    # 容错 1: 空字典 {}
                    if not script:
                        logger.warning("⚠️ 模型返回了空字典，启用终极降级方案。")
                        return self._validate_script_elements([
                            {"type": "narration", "speaker": "narrator", "content": text_chunk}
                        ])

                    # 容错 2a: LLM 返回了 {"name": "...", "content": "..."} 结构
                    if "content" in script and "name" in script:
                        logger.warning("⚠️ 检测到非数组结构（含 name/content），正在将其转换为单条旁白")
                        script = [{"type": "narration", "speaker": "narrator", "content": script["content"]}]
                        return self._validate_script_elements(script)
                    # 容错 2b: LLM 返回了单个 JSON 对象（如 {"type": "narration", "speaker": "narrator", "content": "..."}）
                    if "content" in script or "type" in script:
                        logger.warning("⚠️ 模型返回了单个 JSON 对象而非数组，自动使用列表包裹以恢复流水线。")
                        return self._validate_script_elements([script])
                    # 容错 3: LLM 返回了包含列表的字典 (如 {"script": [...]})
                    for value in script.values():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            return self._validate_script_elements(value)
                    
                    # 【终极降级 2】：模型返回了版权页、书籍元数据等无法识别的字典
                    logger.warning(f"⚠️ 模型返回了无法识别的字典结构（如版权信息），启用终极降级方案。")
                    return self._validate_script_elements([
                        {"type": "narration", "speaker": "narrator", "content": text_chunk}
                    ])
                    
                # 【终极降级 3】：大模型返回了字符串或数字等完全不是对象的格式
                logger.warning("⚠️ 模型返回了非预期结构，启用终极降级方案。")
                return self._validate_script_elements([
                    {"type": "narration", "speaker": "narrator", "content": text_chunk}
                ])

            except Exception as e:
                error_msg = str(e)
                # 🌟 修复：精准拦截阿里云风控系统的特有报错
                if "inappropriate content" in error_msg or "Data inspection failed" in error_msg:
                    logger.error("🚨 致命拦截：触发阿里云底线安全风控！内容涉嫌敏感。")
                    logger.error("⚡ 放弃无意义的重试，瞬间触发终极降级方案（全量原文本转旁白），拯救本章节！")
                    return self._validate_script_elements([
                        {"type": "narration", "speaker": "narrator", "content": text_chunk}
                    ])
                
                # 正常的网络波动或超时，继续退避重试
                wait_time = 5 * (2 ** attempt)
                logger.warning(f"⚠️ 请求异常 ({e})，等待 {wait_time}s 后重试 (尝试 {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue

        raise RuntimeError("❌ 超过最大重试次数，Qwen API 请求彻底失败。请检查您的 DASHSCOPE_API_KEY 是否有效以及账户额度是否充足。")
    
    def _validate_script_elements(self, script: List[Dict]) -> List[Dict]:
        """验证并修复脚本元素，确保包含所有必需字段"""
        required_fields = ['type', 'speaker', 'content']
        validated_script = []
        
        for i, element in enumerate(script):
            # 确保是字典类型
            if not isinstance(element, dict):
                logger.warning(f"⚠️ 脚本元素 {i} 不是字典类型，跳过: {element}")
                continue
                
            # 检查并补充缺失的字段
            fixed_element = element.copy()

            # 【核心修复】：如果大模型把 content 写成了数组，强制拼成字符串
            if 'content' in fixed_element:
                if isinstance(fixed_element['content'], list):
                    fixed_element['content'] = "\n".join(str(x) for x in fixed_element['content'])
                elif not isinstance(fixed_element['content'], str):
                    fixed_element['content'] = str(fixed_element['content'])
            
            # 确保必需字段存在
            for field in required_fields:
                if field not in fixed_element:
                    if field == 'type':
                        fixed_element['type'] = 'narration'  # 默认为旁白
                    elif field == 'speaker':
                        fixed_element['speaker'] = 'narrator'  # 默认说话者
                    elif field == 'content':
                        fixed_element['content'] = ''  # 空内容
                    logger.warning(f"⚠️ 补充缺失字段 '{field}' 在元素 {i}: {element}")
            
            # 强化修复逻辑：处理 None 值
            if fixed_element.get('speaker') is None:
                fixed_element['speaker'] = 'narrator'
                logger.warning(f"⚠️ 修复 None 值字段 'speaker' 在元素 {i}")
            if fixed_element.get('gender') is None:
                fixed_element['gender'] = 'unknown'
                logger.warning(f"⚠️ 修复 None/缺失字段 'gender' 在元素 {i}")
            
            # 确保 gender 字段存在（兼容原有逻辑）
            if 'gender' not in fixed_element:
                fixed_element['gender'] = 'unknown'
                logger.warning(f"⚠️ 补充缺失字段 'gender' 在元素 {i}: {element}")
            
            # 确保 emotion 字段存在
            if 'emotion' not in fixed_element:
                fixed_element['emotion'] = '平静'

            # 🌟 音色防护：如果 emotion 为空，且角色不是 narrator，
            # 根据性别赋予 VOICE_ARCHETYPES 中的默认音色描述，防止 TTS 压制出"机械音"
            emotion_val = fixed_element.get('emotion', '')
            speaker_val = fixed_element.get('speaker', 'narrator')
            if speaker_val != 'narrator' and isinstance(emotion_val, str):
                stripped_emotion = emotion_val.strip()
                if not stripped_emotion:
                    gender_val = fixed_element.get('gender', 'unknown')
                    if gender_val == 'female':
                        default_desc = self.VOICE_ARCHETYPES.get("melancholic", "")
                    else:
                        default_desc = self.VOICE_ARCHETYPES.get("intellectual", "")
                    fixed_element['emotion'] = f"平静 ({default_desc})"
                    logger.warning(
                        f"⚠️ 角色 '{speaker_val}' 的 emotion 为空，已自动补充默认音色描述"
                    )

            # 🌟 音色冲突检测：female 角色不应使用 baritone/bass 描述
            gender = fixed_element.get('gender', 'unknown')
            emotion = fixed_element.get('emotion', '')
            if gender == 'female' and isinstance(emotion, str):
                emotion_lower = emotion.lower()
                if any(kw in emotion_lower for kw in ('baritone', 'bass', 'deep baritone')):
                    logger.warning(
                        f"⚠️ 音色冲突：女性角色 '{fixed_element.get('speaker')}' "
                        f"的 emotion 包含男性音色描述 '{emotion}'，已自动修正"
                    )
                    fixed_element['emotion'] = re.sub(
                        r'\b(baritone|bass)\b', 'alto', emotion, flags=re.IGNORECASE
                    )
            
            validated_script.append(fixed_element)
            
        return validated_script
    

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    director = LLMScriptDirector()
    
    # 测试文本
    test_text = """
第一章 凯夫拉维克的风雪

夜幕降临，港口的灯火开始闪烁。

"你相信命运吗？"老渔夫说道。

年轻人摇摇头："我只相信海。"

远处传来汽笛声，划破了寂静的夜空。
"""
    
    script = director.parse_text_to_script(test_text)
    print("解析结果:")
    for i, unit in enumerate(script, 1):
        print(f"{i}. {unit}")