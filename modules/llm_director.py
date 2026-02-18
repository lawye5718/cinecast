#!/usr/bin/env python3
"""
CineCast 大模型剧本预处理器
阶段一：剧本化与微切片 (Script & Micro-chunking)
实现宏观剧本解析 -> 自动展开为微切片剧本
"""

import json
import re
import logging
import requests
import os
import tempfile
from typing import List, Dict, Optional

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


def salvage_json_entries(raw: str) -> Optional[List[Dict]]:
    """Use regex to extract valid script entries from broken JSON text.

    Each entry is expected to have at least ``speaker`` and ``content`` fields.
    """
    pattern = re.compile(
        r'\{\s*'
        r'"(?:type)"\s*:\s*"([^"]*)"\s*,\s*'
        r'"(?:speaker)"\s*:\s*"([^"]*)"\s*,\s*'
        r'"(?:gender)"\s*:\s*"([^"]*)"\s*,\s*'
        r'"(?:emotion|instruct)"\s*:\s*"([^"]*)"\s*,\s*'
        r'"(?:content)"\s*:\s*"([^"]*)"',
        re.DOTALL,
    )
    entries = []
    for m in pattern.finditer(raw):
        entries.append({
            "type": m.group(1) or "narration",
            "speaker": m.group(2) or "narrator",
            "gender": m.group(3) or "unknown",
            "emotion": m.group(4) or "平静",
            "content": m.group(5) or "",
        })

    if not entries:
        # Looser pattern: just find speaker + content
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
    def __init__(self, ollama_url="http://127.0.0.1:11434", use_local_mlx_lm=False, global_cast=None):
        self.api_url = f"{ollama_url}/api/chat"
        self.model_name = "qwen14b-pro"
        self.max_chars_per_chunk = 60 # 微切片红线（智能配音模式）
        self.pure_narrator_chunk_limit = 100  # 纯净旁白模式切片上限（更长更流畅）
        self.use_local_mlx_lm = use_local_mlx_lm
        self.global_cast = global_cast or {}  # 🌟 外脑全局角色设定集
        
        # Context sliding window state
        self._prev_characters: List[str] = []
        self._prev_tail_entries: List[Dict] = []
        
        # 测试Ollama连接
        self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """测试Ollama服务连接"""
        try:
            response = requests.get(f"{self.api_url.replace('/api/chat', '')}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama服务连接正常")
                return True
            else:
                logger.warning("❌ Ollama服务响应异常")
                return False
        except Exception as e:
            logger.warning(f"❌ 无法连接到Ollama服务: {e}")
            return False
    
    def _try_ollama_qwen(self) -> bool:
        """尝试使用Ollama的Qwen14B模型"""
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0 and "qwen14b-pro" in result.stdout:
                logger.info("✅ 成功检测到本地Ollama Qwen14B模型")
                self.model_type = "ollama"
                self.model_name = "qwen14b-pro"
                return True
            else:
                logger.info("未找到Ollama Qwen14B模型")
                return False
                
        except Exception as e:
            logger.warning(f"检查Ollama模型时出错: {e}")
            return False
    
    def _chunk_text_for_llm(self, text: str, max_length: int = 800) -> List[str]:
        """🌟 防止章节过长，按段落切分为安全大小给 LLM 处理"""
        paragraphs = text.split('\n')
        chunks, current_chunk = [], ""
        for para in paragraphs:
            if not para.strip(): continue
            if len(current_chunk) + len(para) > max_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para + "\n"
            else:
                current_chunk += para + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def verify_integrity(self, original_text: str, script_list: List[Dict]) -> bool:
        """🌟 内容完整性校验：检测 LLM 解析是否严重丢失内容

        对比原文总字数与剧本 content 总字数，如果丢失超过 10% 则报错。

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
        ratio = len(content_text) / original_len
        if ratio < 0.9:
            logger.error(
                f"🚨 内容丢失严重！原文{original_len}字，"
                f"解析后仅{len(content_text)}字 (保留率{ratio:.1%})"
            )
            return False
        logger.info(f"✅ 内容完整性校验通过 (保留率{ratio:.1%})")
        return True
    
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

    def parse_and_micro_chunk(self, text: str, chapter_prefix: str = "chunk") -> List[Dict]:
        """宏观剧本解析 -> 自动展开为微切片剧本
        
        Args:
            text: 待处理的章节文本
            chapter_prefix: 章节名称前缀，用于避免文件名冲突
        """
        # 第一步：生成宏观剧本
        macro_script = self.parse_text_to_script(text)
        micro_script = []
        chunk_id = 1
        
        for unit in macro_script:
            content = unit.get("content", "")
            if not content or not content.strip():
                continue

            # 实施微切片
            raw_sentences = re.split(r'([。！？；，、：])', content)
            chunks, temp = [], ""
            for part in raw_sentences:
                if not part.strip(): continue
                if re.match(r'^[。！？；，、：]$', part.strip()):
                    chunks.append(temp + part)
                    temp = ""
                else:
                    temp += part
                    if len(temp) >= self.max_chars_per_chunk:
                        chunks.append(temp)
                        temp = ""
            if temp: chunks.append(temp)
            
            # 清理空块并计算停顿
            valid_chunks = [c.strip() for c in chunks if c.strip()]

            # 🌟 兜底逻辑：如果正则切分后无有效块，按每60字硬切
            if not valid_chunks and content.strip():
                hard_cut_chunk_size = self.max_chars_per_chunk
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

    def parse_text_to_script(self, text: str) -> List[Dict]:
        """阶段一：宏观剧本解析（保持原有逻辑）
        
        Implements a context sliding window: each chunk receives the previous
        chunk's cast list and last three entries as context so that character
        names and speaking styles stay consistent across slices.
        """
        # 🌟 修复截断漏洞：按段落切分长章节
        text_chunks = self._chunk_text_for_llm(text)
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
            
            chunk_script = self._request_ollama(chunk, context="\n".join(context_parts) if context_parts else None)
            
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
            
            full_script.extend(chunk_script)
        
        # 🌟 优化：移除 merge_consecutive_narrators 调用。
        # 因为 parse_and_micro_chunk 会对结果进行严格的 60 字微切片，
        # 合并后的 800 字长文本会被立即碾碎，属于无谓的算力浪费。
        
        # 如果解析结果为空，直接报错退出
        if not full_script or len(full_script) == 0:
            raise RuntimeError("❌ 剧本解析结果为空，请检查输入文本和大模型服务是否正常。")

        # 🌟 内容完整性守门员：检测 LLM 是否严重删节内容
        self.verify_integrity(text, full_script)
            
        return full_script
    
    def generate_chapter_recap(self, prev_chapter_text: str) -> str:
        """
        🌟 Map-Reduce 前情摘要引擎
        解决长章节导致大模型 OOM 或注意力丢失的问题
        - 短文本 (<=5000字): 直接生成终极摘要
        - 长文本 (>5000字): 分块提炼 -> 合并 -> 终极摘要
        """
        chunk_size = 5000

        # 1. 基础清理
        text = prev_chapter_text.strip()
        if not text:
            return ""

        # 2. 如果文本极长，执行 Map 阶段 (分块总结)
        if len(text) > chunk_size:
            logger.info(f"🔄 上一章超长 ({len(text)}字)，启动 Map-Reduce 摘要分块处理...")
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            sub_summaries = []

            for idx, chunk in enumerate(chunks):
                map_prompt = ("你是一个剧情提炼专家。请将以下小说片段压缩成不超过200字的纯剧情摘要。"
                              "要求：只保留核心冲突、关键人物动作和重要线索。不加任何前缀和废话。")

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": map_prompt},
                        {"role": "user", "content": f"小说片段：\n{chunk}"}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
                try:
                    resp = requests.post(self.api_url, json=payload, timeout=120)
                    resp.raise_for_status()
                    sub_sum = resp.json().get('message', {}).get('content', '').strip()
                    sub_summaries.append(sub_sum)
                    logger.debug(f"   ✓ 完成第 {idx+1}/{len(chunks)} 块提炼")
                except Exception as e:
                    logger.warning(f"区块 {idx+1} 总结失败，已跳过: {e}")

            # 组合所有子摘要作为 Reduce 阶段的输入
            final_input = "\n".join(sub_summaries)
        else:
            final_input = text

        # 3. Reduce 阶段 (终极摘要 + 悬念钩子)
        reduce_prompt = (
            '你是一位顶级的有声书剧本编辑和悬疑大师。'
            '请根据提供的上一章内容（或内容摘要），写一段不超过100字的\u201c前情摘要\u201d。'
            '绝对纪律：'
            '1. 语言必须高度凝练，具有美剧片头的电影感（\u201cPreviously on...\u201d的风格）。'
            '2. 只保留最具张力的剧情矛盾。'
            '3. 最后一句必须是一个引出下一章的\u201c悬念钩子\u201d。'
            '4. 绝对不要输出\u201c前情提要：\u201d这样的标题，直接输出正文。'
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": reduce_prompt},
                {"role": "user", "content": f"上一章内容：\n{final_input}"}
            ],
            "stream": False,
            "options": {"temperature": 0.5, "top_p": 0.8}
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            recap_result = response.json().get('message', {}).get('content', '').strip()

            # 清理大模型可能违规加上的前缀
            recap_result = re.sub(r'^(前情提要|前情摘要|回顾|摘要)[:：]\s*', '', recap_result)
            return recap_result
        except Exception as e:
            logger.error(f"终极摘要生成失败: {e}")
            return ""
    
    def _request_ollama(self, text_chunk: str, *, context: Optional[str] = None) -> List[Dict]:
        """向Ollama发送单个文本块请求

        Args:
            text_chunk: The raw text to convert into a script.
            context: Optional sliding-window context from the previous chunk
                     (character list + tail entries) to maintain consistency.
        """
        system_prompt = """
        你是一位顶级的有声书导演兼数据清洗专家，负责将原始小说文本转换为标准化的录音剧本。
        你必须严格遵守以下四大纪律，任何违反都将导致系统崩溃：

        【一、 绝对忠实原则（Iron Rule）】
        - 必须 100% 逐字保留原文内容！
        - 严禁任何形式的概括、改写、缩写、续写或润色！
        - 严禁自行添加原文中不存在的台词或动作描写！
        - 严禁在 content 中保留归属标签（如"他说"、"她叫道"），归属信息只能出现在 speaker 字段！
        - 严禁总结全文，必须从第一句话开始逐句拆解。严禁输出章节名称作为唯一内容！
        - 严禁漏掉任何一个自然段！如果发现内容缺失，系统将判定任务失败并重试！

        【二、 字符净化原则】
        - 剔除所有不可发音的特殊符号（如 Emoji表情、Markdown标记 * _ ~ #、制表符 \t、不可见控制字符）。
        - 仅保留基础标点符号（，。！？：；、""''（））。
        - 数字、英文字母允许保留，但禁止出现复杂的数学公式符号。

        【三、 粒度拆分原则】
        - 必须将"对白"和"旁白/动作描写"严格剥离为独立的对象！
        - 例如原文："你好，"老渔夫笑着说。
          必须拆分为两个对象：1. 角色对白("你好，") 2. 旁白描述("老渔夫笑着说。")

        【四、 JSON 格式规范（生死攸关）】
        必须且只能输出合法的 JSON 数组（Array）！
        最外层必须是中括号 [ ]，绝对不能是大括号 { }！
        严禁将整个章节合并为一个单一的 JSON 对象！你必须将文本逐句拆解为多个数组元素！
        禁止任何解释性前言或后缀，禁止输出 Markdown 代码块标记（```json）。
        数组元素字段要求：
        - "type": 仅限 "title"(章节名), "subtitle"(小标题), "narration"(旁白), "dialogue"(对白)。
        - "speaker": 对白填具体的角色名（需根据上下文推断并保持全书统一）；旁白和标题统一填 "narrator"。
        - "gender": 仅限 "male"、"female" 或 "unknown"。对白请推测性别；旁白固定为 "male"。
        - "emotion": 情感标签（如"平静"、"激动"、"沧桑/叹息"、"愤怒"、"悲伤"等），用于未来语音合成的情感控制。
        - "content": 纯净的文本内容。如果 type 是 "dialogue"，必须去掉最外层的引号（如""或""）。

        【输出格式示例（One-Shot）】
        [
          {
            "type": "narration",
            "speaker": "narrator",
            "gender": "male",
            "emotion": "平静",
            "content": "夜幕降临，港口的灯火开始闪烁。"
          },
          {
            "type": "dialogue",
            "speaker": "老渔夫",
            "gender": "male",
            "emotion": "沧桑/叹息",
            "content": "你相信命运吗？"
          },
          {
            "type": "narration",
            "speaker": "narrator",
            "gender": "male",
            "emotion": "平静",
            "content": "老渔夫说道。"
          }
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
        """

        user_content = "请严格按照规范，将以下文本拆解为纯净的 JSON 剧本（绝不改写原意）：\n\n"
        if context:
            user_content += f"【上文参考（仅供角色一致性参考，不要翻译此段）】\n{context}\n\n"
        user_content += text_chunk

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": 8192,
                "temperature": 0.0,
                "top_p": 0.1
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            content = response.json().get('message', {}).get('content', '[]')

            # 🌟 预处理：清洗实际控制字符（防止 LLM 输出破坏 JSON 解析）
            # Only strip real control characters; keep escaped sequences
            # like \n and \t inside JSON strings intact.
            content = content.replace('\t', ' ').replace('\r', '')

            # Strip Markdown code-block wrappers the LLM may hallucinate
            content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content.strip())

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

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"❌ Ollama 解析失败: {e}") from e
    
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