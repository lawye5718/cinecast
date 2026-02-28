#!/usr/bin/env python3
"""
CineCast Qwen-Flash 大模型剧本预处理器 (简化版)
专为商业API优化，移除所有免费模型限制
"""

import json
import re
import logging
import os
import tempfile
from typing import List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


def atomic_json_write(path: str, data, **kwargs) -> None:
    """Atomic JSON write: write to a temporary file first, then replace."""
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
    """Attempt to repair a truncated or malformed JSON array."""
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

    return salvage_json_entries(raw)


def _extract_fields_from_object(obj_text: str) -> Optional[Dict]:
    """Extract known fields from a single JSON object text in any order."""
    field_re = re.compile(r'"(\w+)"\s*:\s*"([^"]*)"')
    fields: Dict[str, str] = {}
    for m in field_re.finditer(obj_text):
        fields[m.group(1)] = m.group(2)

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
    """Use regex to extract valid script entries from broken JSON text."""
    obj_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL)
    entries = []
    for m in obj_pattern.finditer(raw):
        entry = _extract_fields_from_object(m.group(0))
        if entry and entry.get("content"):
            entries.append(entry)

    if not entries:
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


class QwenScriptDirector:
    # 高阶角色音色映射表
    VOICE_ARCHETYPES = {
        "intellectual": "Clear, articulate, mid-range voice, steady pacing, calm and intellectual.",
        "villain_sly": "Slightly nasal, fast-paced voice, bright tone, with a hint of sarcasm.",
        "melancholic": "Breathier, soft voice, melancholic undertones, slow and emotional.",
        "authoritative": "Resonant, deep baritone, slow and authoritative, gravelly texture.",
        "innocent": "Bright, high-pitched, energetic and innocent, clear enunciation.",
    }

    def __init__(self, api_key=None, global_cast=None, cast_db_path=None, **kwargs):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            logger.warning("⚠️ 未设置 DASHSCOPE_API_KEY")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        self.model_name = "qwen-flash"
        self.max_chars_per_chunk = 60
        self.pure_narrator_chunk_limit = 100
        self.global_cast = global_cast or {}
        self.cast_db_path = cast_db_path or os.path.join("workspace", "cast_profiles.json")
        self.cast_profiles = self._load_cast_profiles()

    def _load_cast_profiles(self) -> Dict[str, Dict]:
        """加载已保存的角色音色库"""
        if os.path.exists(self.cast_db_path):
            try:
                with open(self.cast_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"⚠️ 加载角色音色库失败: {e}")
        return {}

    def reset_context(self) -> None:
        """重置上下文"""
        logger.info("♻️ 重置上下文")

    def _chunk_text_for_llm(self, text: str, max_length: int = 997000) -> List[str]:
        """按段落切分为安全大小给 LLM 处理"""
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
        """内容完整性校验"""
        if not original_text or not script_list:
            return True
        content_text = "".join([item.get("content", "") for item in script_list])
        original_len = len(original_text.strip())
        if original_len == 0:
            return True
        ratio = len(content_text) / original_len
        if ratio < 0.9:
            logger.error(f"🚨 内容丢失严重！保留率{ratio:.1%}")
            return False
        logger.info(f"✅ 内容完整性校验通过 (保留率{ratio:.1%})")
        return True
    
    def generate_pure_narrator_script(self, text: str, chapter_prefix: str = "chunk") -> List[Dict]:
        """纯净旁白模式专用的剧本生成器"""
        micro_script = []
        chunk_id = 1
        pure_chunk_limit = self.pure_narrator_chunk_limit
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        for p_idx, para in enumerate(paragraphs):
            sentences = re.split(r'([。！？；.!?;])', para)
            temp_sentence = ""
            for part in sentences:
                if not part.strip() and not re.match(r'[。！？；.!?;]', part):
                    continue

                if re.match(r'^[。！？；.!?;]$', part.strip()):
                    temp_sentence += part
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

    def parse_and_micro_chunk(self, text: str, chapter_prefix: str = "chunk", max_length: int = 997000) -> List[Dict]:
        """宏观剧本解析 -> 自动展开为微切片剧本"""
        macro_script = self.parse_text_to_script(text, max_length=max_length)
        micro_script = []
        chunk_id = 1
        smart_chunk_limit = max(self.max_chars_per_chunk, 90) 
        
        for unit in macro_script:
            content = unit.get("content", "")
            if not content or not content.strip():
                continue

            raw_sentences = re.split(r'([。！？；.!?;])', content)
            chunks = []
            temp = ""
            for part in raw_sentences:
                if not part.strip():
                    continue
                if re.match(r'^[。！？；.!?;]$', part.strip()):
                    temp += part
                    if len(temp) <= smart_chunk_limit:
                        chunks.append(temp)
                        temp = ""
                    else:
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
            
            valid_chunks = [c.strip() for c in chunks if c.strip()]

            if not valid_chunks and content.strip():
                hard_cut_chunk_size = smart_chunk_limit
                stripped = content.strip()
                valid_chunks = [
                    stripped[i:i + hard_cut_chunk_size]
                    for i in range(0, len(stripped), hard_cut_chunk_size)
                ]

            for idx, chunk in enumerate(valid_chunks):
                is_para_end = (idx == len(valid_chunks) - 1)
                pause_ms = self._calculate_pause(chunk, is_para_end)
                
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
        """计算停顿时间"""
        if is_para_end: return 1000
        if chunk_text.endswith(('。', '！', '？', '.', '!', '?')): return 600
        elif chunk_text.endswith(('；', ';')): return 400
        elif chunk_text.endswith(('，', '、', ',', '：', ':')): return 250
        else: return 100

    @staticmethod
    def _normalize_text(text: str) -> str:
        """数字和符号转换为中文可读形式"""
        _DIGIT_MAP = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
        }

        def _digits_to_chinese(m: re.Match) -> str:
            s = m.group(0)
            return ''.join(_DIGIT_MAP.get(c, c) for c in s)

        text = re.sub(r'(\d+(?:\.\d+)?)%', lambda m: f'百分之{"".join(_DIGIT_MAP.get(c, c) for c in m.group(1))}', text)
        text = re.sub(r'(\d+)\.(\d+)', lambda m: f'{"".join(_DIGIT_MAP.get(c, c) for c in m.group(1))}点{"".join(_DIGIT_MAP.get(c, c) for c in m.group(2))}', text)
        text = re.sub(r'\d+', _digits_to_chinese, text)
        return text

    def parse_text_to_script(self, text: str, max_length: int = 997000) -> List[Dict]:
        """使用Qwen-Flash进行剧本解析"""
        logger.info(f"🚀 启动 Qwen-Flash 剧本解析，当前章节字数: {len(text)}")
        text_chunks = self._chunk_text_for_llm(text, max_length=max_length)
        full_script = []
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"   🧠 正在解析剧情片段 {i+1}/{len(text_chunks)}...")
            chunk_script = self._request_llm(chunk)
            full_script.extend(chunk_script)
        
        if not full_script:
            raise RuntimeError("❌ 剧本解析结果为空")
            
        if not self.verify_integrity(text, full_script):
            logger.warning("⚠️ 内容完整性校验未通过")
            
        return full_script
    
    def generate_chapter_recap(self, prev_chapter_text: str) -> str:
        """前情摘要生成"""
        text = prev_chapter_text.strip()
        if not text:
            return ""

        logger.info(f"🚀 启动 Qwen-Flash 前情摘要生成，上一章字数: {len(text)}")
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
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": reduce_prompt},
                    {"role": "user", "content": f"上一章内容：\n{text}"}
                ],
                extra_body={"enable_thinking": True},
                stream=False,
                temperature=0.5,
                top_p=0.8,
                max_tokens=32768,
            )
            recap_result = completion.choices[0].message.content.strip()
            recap_result = re.sub(r'^(前情提要|前情摘要|回顾|摘要)[:：]\s*', '', recap_result)
            return recap_result
        except Exception as e:
            logger.error(f"终极摘要生成失败: {e}")
            return ""
    
    def _request_llm(self, text_chunk: str) -> List[Dict]:
        """向 Qwen-Flash API 发送请求"""
        EMOTION_SET = "平静, 愤怒, 悲伤, 喜悦, 恐惧, 惊讶, 沧桑, 柔和, 激动, 嘲讽, 哽咽, 冰冷, 狂喜"

        system_prompt = f"""你是一个高精度的有声书剧本转换接口。
任务：将输入文本逐句解析为 JSON 数组格式。
核心规则：
1. 物理对齐：原文的每一句、每一段必须对应数组中的一个对象。严禁合并，严禁删减。
2. 根节点约束：输出结果必须是一个标准的 JSON 数组（即以 `[` 开头）。严禁输出 `{{"data": [...]}}` 这种格式。
3. 字段要求：每个对象必须包含 type, speaker, gender, emotion, content 字段。
4. 角色一致性：speaker 必须根据上下文推断。
5. 情绪约束：仅限 [{EMOTION_SET}]。如伴随特定发音特征（如"叹气", "低语"），可在情绪后加括号说明，例如："悲伤 (带哭腔)"。
"""

        one_shot_example = """
【输入】：
"你好，"老渔夫说。他看着大海。
【输出】：
[
  {"type": "dialogue", "speaker": "老渔夫", "gender": "male", "emotion": "平静", "content": "你好，"},
  {"type": "narration", "speaker": "narrator", "gender": "male", "emotion": "平静", "content": "老渔夫说。"},
  {"type": "narration", "speaker": "narrator", "gender": "male", "emotion": "平静", "content": "他看着大海。"}
]
"""

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

        system_prompt += self._get_archetype_prompt()

        text_chunk = self._normalize_text(text_chunk)
        text_chunk = re.sub(r'"([^"]*)"', lambda m: '\u201c' + m.group(1) + '\u201d', text_chunk)
        text_chunk = text_chunk.replace('"', '\u2018')

        user_content = f"【指令：将以下文本转换为平铺的 JSON 数组，严禁最外层使用字典】\n\n待处理原文：\n{text_chunk}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt + "\n示例参考：" + one_shot_example},
                    {"role": "user", "content": user_content}
                ],
                extra_body={"enable_thinking": True},
                stream=False,
                temperature=0.1,
                top_p=0.1,
                max_tokens=32768,
            )

            content = completion.choices[0].message.content.strip()
            content = content.replace('\t', ' ').replace('\r', '')
            content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content.strip())

            try:
                script = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("⚠️ JSON 解析失败，尝试修复...")
                script = repair_json_array(content)
                if script is None:
                    logger.warning("⚠️ JSON彻底损坏，启用降级方案")
                    return self._validate_script_elements([
                        {"type": "narration", "speaker": "narrator", "content": text_chunk}
                    ])
                return self._validate_script_elements(script)

            if isinstance(script, list):
                return self._validate_script_elements(script)

            if isinstance(script, dict):
                if not script:
                    return self._validate_script_elements([
                        {"type": "narration", "speaker": "narrator", "content": text_chunk}
                    ])
                if "content" in script and "name" in script:
                    script = [{"type": "narration", "speaker": "narrator", "content": script["content"]}]
                    return self._validate_script_elements(script)
                if "content" in script or "type" in script:
                    return self._validate_script_elements([script])
                for value in script.values():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        return self._validate_script_elements(value)
                return self._validate_script_elements([
                    {"type": "narration", "speaker": "narrator", "content": text_chunk}
                ])
                    
            return self._validate_script_elements([
                {"type": "narration", "speaker": "narrator", "content": text_chunk}
            ])

        except Exception as e:
            raise RuntimeError(f"❌ Qwen-Flash API 请求失败: {e}")

    def _get_archetype_prompt(self) -> str:
        """生成音色映射指南"""
        guidelines = "\n".join([f"  - {k}: {v}" for k, v in self.VOICE_ARCHETYPES.items()])
        return (
            "\n【音色设计参考手册】\n"
            "当为新角色生成 (Acoustic Description) 时，请优先参考以下文学原型描述词：\n"
            f"{guidelines}\n"
        )
    
    def _validate_script_elements(self, script: List[Dict]) -> List[Dict]:
        """验证并修复脚本元素"""
        required_fields = ['type', 'speaker', 'content']
        validated_script = []
        
        for i, element in enumerate(script):
            if not isinstance(element, dict):
                continue
                
            fixed_element = element.copy()

            if 'content' in fixed_element:
                if isinstance(fixed_element['content'], list):
                    fixed_element['content'] = "\n".join(str(x) for x in fixed_element['content'])
                elif not isinstance(fixed_element['content'], str):
                    fixed_element['content'] = str(fixed_element['content'])
            
            for field in required_fields:
                if field not in fixed_element:
                    if field == 'type':
                        fixed_element['type'] = 'narration'
                    elif field == 'speaker':
                        fixed_element['speaker'] = 'narrator'
                    elif field == 'content':
                        fixed_element['content'] = ''
            
            if fixed_element.get('speaker') is None:
                fixed_element['speaker'] = 'narrator'
            if 'gender' not in fixed_element:
                fixed_element['gender'] = 'unknown'
            if 'emotion' not in fixed_element:
                fixed_element['emotion'] = '平静'

            validated_script.append(fixed_element)
            
        return validated_script
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    director = QwenScriptDirector()
    
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