#!/usr/bin/env python3
"""
CineCast Web UI
基于 Gradio Blocks API 的现代化图形界面
支持纯净旁白/智能配音双模式、云端外脑 Master JSON 统一输入、极速试听与全本压制
包含：工作区断点记忆与自动恢复功能、实时制片日志流式展示、自动质检
"""

import copy
import os
import json
import re
import shutil
import uuid
import requests
import gradio as gr
from main_producer import CineCastProducer

# Qwen3-TTS 官方支持的预设音色列表
QWEN_PRESET_VOICES = [
    "Eric (默认男声)", "Serena (默认女声)",
    "Aiden", "Dylan", "Ono_anna", "Ryan", "Sohee", "Uncle_fu", "Vivian",
]

# 流式API配置
STREAM_API_URL = "http://localhost:8000"

# 🌟 弃用名单：eric 和 serena 默认不使用，除非用户主动选择（且仅当次有效）
DEPRECATED_VOICES = {"eric", "serena"}
# 优先分配的默认音色顺序（排除 eric/serena）
DEFAULT_VOICE_ORDER = ["aiden", "dylan", "ryan", "uncle_fu", "ono_anna", "sohee", "vivian"]


# --- 流式API功能 ---
def test_stream_api_connection():
    """测试流式API连接"""
    try:
        response = requests.get(f"{STREAM_API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✅ 流式API连接成功 - 状态: {data.get('status', 'unknown')}"
        else:
            return f"❌ 流式API连接失败 - 状态码: {response.status_code}"
    except Exception as e:
        return f"❌ 流式API连接异常: {str(e)}"

def get_available_voices():
    """获取可用音色列表"""
    try:
        response = requests.get(f"{STREAM_API_URL}/voices", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.warning(f"获取音色列表失败: {e}")
        return None

def set_stream_voice(voice_name: str, audio_file=None):
    """设置流式API的当前音色"""
    try:
        if audio_file:
            # 上传音频文件进行音色克隆
            files = {'file': audio_file}
            data = {'voice_name': voice_name}
            response = requests.post(
                f"{STREAM_API_URL}/set_voice",
                data=data,
                files=files,
                timeout=30
            )
        else:
            # 使用预设音色
            data = {'voice_name': voice_name}
            response = requests.post(
                f"{STREAM_API_URL}/set_voice",
                data=data,
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            return f"✅ 音色设置成功: {result.get('voice_name', 'unknown')}"
        else:
            return f"❌ 音色设置失败: {response.text}"
    except Exception as e:
        return f"❌ 音色设置异常: {str(e)}"

def stream_tts_read(text: str, language: str = "zh"):
    """调用流式API进行实时朗读"""
    try:
        params = {'text': text, 'lang': language}
        response = requests.get(
            f"{STREAM_API_URL}/read_stream",
            params=params,
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            # 保存音频到临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                temp_path = f.name
            return temp_path
        else:
            logger.error(f"流式朗读失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"流式朗读异常: {e}")
        return None

# --- 新增：大模型连接测试 ---
def test_llm_connection(model_name, base_url, api_key):
    """测试兼容 OpenAI API 格式的大模型连接"""
    if not all([model_name, base_url, api_key]):
        return "❌ 请完整填写大模型名称、Base URL 和 API Key！"

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "测试连接，请只回复1个字"}],
            "max_tokens": 10,
        }
        api_endpoint = f"{base_url.rstrip('/')}/chat/completions"

        response = requests.post(
            api_endpoint, json=payload, headers=headers, timeout=30
        )

        if response.status_code == 200:
            save_llm_config(model_name, base_url, api_key)
            return f"✅ 连接成功！已成功握手 {model_name}。配置已保存到本地。"
        else:
            return (
                f"❌ 测试失败 (HTTP {response.status_code}): {response.text}\n"
                "请检查各项参数。"
            )
    except Exception as e:
        return (
            f"❌ 请求异常：{str(e)}\n"
            "请检查网络和 Base URL 格式（例如需包含 /v1）。"
        )

# --- 🌟 新增：工作区状态持久化 ---
WORKSPACE_FILE = "./.cinecast_workspace.json"
ROLE_VOICE_FILE = "./.cinecast_role_voices.json"
LLM_CONFIG_FILE = "./.cinecast_llm_config.json"


def load_llm_config():
    """读取本地保存的大模型 API 配置"""
    if os.path.exists(LLM_CONFIG_FILE):
        try:
            with open(LLM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 大模型配置读取失败，使用默认设置: {e}")
    return {"model_name": "qwen-plus", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": ""}


def save_llm_config(model_name, base_url, api_key):
    """将大模型 API 配置保存到本地文件"""
    config = {"model_name": model_name, "base_url": base_url, "api_key": api_key}
    try:
        with open(LLM_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 大模型配置保存失败: {e}")


def load_role_voices():
    """读取全局固化的身份音色配置"""
    if os.path.exists(ROLE_VOICE_FILE):
        try:
            with open(ROLE_VOICE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    # 🌟 默认使用 aiden 而非 eric（eric/serena 已弃用）
    return {"narrator": {"mode": "preset", "voice": "aiden"}}


def save_role_voice(role, voice_cfg):
    """保存用户为特定身份锁定的音色。

    eric/serena 为弃用音色，仅允许当次使用，不写入持久化配置。
    """
    if role not in ["m1", "f1", "m2", "f2", "narrator"]:
        return
    # 🌟 eric/serena 单次使用，不持久化
    voice_id = voice_cfg.get("voice", "")
    if isinstance(voice_id, str) and voice_id.lower() in DEPRECATED_VOICES:
        return
    voices = load_role_voices()
    voices[role] = voice_cfg
    try:
        with open(ROLE_VOICE_FILE, 'w', encoding='utf-8') as f:
            json.dump(voices, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 全局身份音色存档失败: {e}")


def load_workspace():
    """启动时加载上一次的工作区状态"""
    if os.path.exists(WORKSPACE_FILE):
        try:
            with open(WORKSPACE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
                print(f"🔄 已恢复上次的工作区断点状态: {state.get('book_file', '无文件')}")
                return state
        except Exception as e:
            print(f"⚠️ 工作区状态读取失败，使用默认设置: {e}")
    return {"book_file": None, "mode": "🎙️ 纯净旁白模式", "master_json": ""}


def save_workspace(book_file, mode, master_json):
    """每次触发任务时，保存当前状态"""
    # 获取文件的绝对路径 (Gradio 的 file_obj 可能是路径字符串或具有 name 属性的对象)
    if book_file is None:
        file_path = None
    elif hasattr(book_file, "name"):
        file_path = book_file.name
    else:
        file_path = book_file
    state = {
        "book_file": file_path,
        "mode": mode,
        "master_json": master_json
    }
    try:
        with open(WORKSPACE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 工作区状态保存失败: {e}")


# --- 🌟 新增：实时日志流式读取 ---
LOG_FILE = "cinecast.log"


def get_logs():
    """读取 cinecast.log 的最后 50 行，供 WebUI 定时轮询展示"""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return "".join(lines[-50:])
        except Exception as e:
            return f"⚠️ 日志读取失败: {e}"
    return "等待日志输出..."


# --- 🌟 新增：无头质检（Headless QC） ---
def run_headless_qc(output_dir, sensitivity=0.4):
    """在无 GUI 环境下自动执行噪音检测，返回文本报告

    Args:
        output_dir: 要扫描的音频输出目录
        sensitivity: 噪音检测灵敏度 (0.1–1.0)

    Returns:
        质检结果的文本摘要
    """
    try:
        from audio_shield.scanner import AudioScanner
        from audio_shield.analyzer import detect_audio_glitches
    except ImportError:
        return "⚠️ 质检模块依赖缺失 (librosa)，跳过自动质检。"

    if not os.path.isdir(output_dir):
        return "⚠️ 未发现输出目录，跳过质检。"

    scanner = AudioScanner(output_dir)
    scanner.scan()

    if not scanner.files:
        return "⚠️ 未发现可质检的音频文件。"

    results = []
    total = len(scanner.files)
    for i, finfo in enumerate(scanner.files, 1):
        try:
            glitches = detect_audio_glitches(finfo.file_path, sensitivity=sensitivity)
            status = f"⚠️ {len(glitches)}处异常" if glitches else "✅ 通过"
            results.append(f"[{i}/{total}] {finfo.filename}: {status}")
        except Exception as e:
            results.append(f"[{i}/{total}] {finfo.filename}: ❌ 分析失败 ({e})")

    passed = sum(1 for r in results if "✅" in r)
    summary = f"🔍 质检完成: {passed}/{total} 个文件通过\n" + "\n".join(results)
    return summary

# 🌟 终极"云端外脑" Prompt 规范（供用户复制给 Kimi、豆包或 Claude 等长文本大模型）
BRAIN_PROMPT_TEMPLATE = """\
你是一位顶级的有声书"总导演兼剧本编审"。我已经上传了一本小说的全本文件。
请你通读全书，完成【角色选角】与【前情提要】两项核心任务，并按 JSON 格式输出。

【任务一：建立全局角色设定集】
1. 提取所有有台词的角色，统一【标准名】。
2. 必须为每个角色分配一个【身份标签(role)】，只能从以下选择：
   - m1 (男主/核心男配)
   - f1 (女主/核心女配)
   - m2 (男配)
   - f2 (女配)
   - extra (路人或龙套)
3. 推断性别(gender)和情感(emotion)。包含名为"路人"的默认角色。

【任务二：撰写各章前情提要】
1. 为**除第一章以外**的每一章，生成一段用于片头播报的前情提要（80-120字）。
2. 语言必须高度凝练，具有美剧片头的电影感。
3. 最后一句必须是一个引出本章内容的"悬念钩子"。

【⚠️ 格式生死攸关 ⚠️】
你必须且只能输出一个合法的纯 JSON 字典格式！包含 "characters" 和 "recaps" 两个根节点。
绝对不要输出任何 markdown 标记（如 ```json），不要包含任何解释性废话，直接输出大括号包裹的 JSON！

【输出格式示例】
{
  "characters": {
    "老渔夫": {"role": "m1", "gender": "male", "emotion": "沧桑"},
    "艾米莉": {"role": "f1", "gender": "female", "emotion": "活泼"},
    "路人": {"role": "extra", "gender": "unknown", "emotion": "平淡"}
  },
  "recaps": {
    "Chapter_002": "上一章中，老渔夫在暴风雪中带回了一个神秘的黑匣子……然而他没意识到，危险才刚刚降临。",
    "Chapter_003": "警长的调查陷入僵局，唯一的目击者却在昨夜离奇失踪……一通电话突然打进了警局。"
  }
}"""


# --- 🌟 新增：极速试听 首章前10句提取 ---
def extract_preview_sentences(book_file, num_sentences=10):
    """从小说文件中提取首章前N句，用于极速试听文本展示。

    支持 EPUB 和 TXT 格式。返回提取的句子文本（一行一句）。

    Args:
        book_file: 文件路径或带有 .name 属性的 Gradio 文件对象。
        num_sentences: 提取的句子数，默认10。

    Returns:
        str: 提取的句子文本（每行一句），失败时返回错误提示。
    """
    if book_file is None:
        return "❌ 请先上传小说文件。"

    file_path = book_file.name if hasattr(book_file, "name") else book_file
    if not os.path.exists(file_path):
        return "❌ 文件不存在。"

    text = ""
    try:
        if file_path.lower().endswith(".epub"):
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(file_path)
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                chapter_text = soup.get_text(separator="\n").strip()
                if len(chapter_text) > 100:
                    text = chapter_text
                    break
        elif file_path.lower().endswith((".txt", ".md")):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            return "❌ 不支持的文件格式。"
    except Exception as e:
        return f"❌ 文件读取失败：{e}"

    if not text.strip():
        return "❌ 未能从文件中提取有效文本。"

    # 按中英文标点分句
    sentences = re.split(r'(?<=[。！？!?])', text)
    # 同时按换行拆分
    expanded = []
    for s in sentences:
        expanded.extend(s.split('\n'))
    sentences = [s.strip() for s in expanded if s.strip()]
    selected = sentences[:num_sentences]
    return "\n".join(selected)


# --- 辅助函数：保存用户上传的资产 ---
def save_uploaded_asset(file_obj, target_filename, folder):
    """将用户上传的音频文件复制到 assets 目录的指定子文件夹

    Args:
        file_obj: 文件路径字符串，或带有 .name 属性的 Gradio 文件对象，
                  或 None（跳过）。
        target_filename: 目标文件名。如果为 None，则使用原始文件名。
        folder: assets 下的子文件夹名称。

    Returns:
        保存后的目标路径，或 None。
    """
    if file_obj is None:
        return None
    target_dir = os.path.join("./assets", folder)
    os.makedirs(target_dir, exist_ok=True)
    # 兼容路径字符串和 Gradio 文件对象
    src_path = file_obj.name if hasattr(file_obj, "name") else file_obj
    final_name = target_filename if target_filename else os.path.basename(src_path)
    target_path = os.path.join(target_dir, final_name)
    shutil.copy(src_path, target_path)
    return target_path


def process_master_json(master_json_str):
    """🌟 核心解析：将统一的 Master JSON 拆包为 角色库 和 摘要库

    Args:
        master_json_str: 外脑返回的 JSON 字符串，包含 "characters" 和 "recaps" 两个根节点。

    Returns:
        (global_cast, custom_recaps, success, message) 四元组
    """
    global_cast = {}
    custom_recaps = {}

    if not master_json_str or not master_json_str.strip():
        return global_cast, custom_recaps, True, ""

    try:
        master_data = json.loads(master_json_str)
        
        # 验证必需的根节点字段
        if "characters" not in master_data:
            return {}, {}, False, "❌ 外脑 JSON 缺少必需的 'characters' 字段"
        if "recaps" not in master_data:
            return {}, {}, False, "❌ 外脑 JSON 缺少必需的 'recaps' 字段"
        
        # 验证字段类型
        if not isinstance(master_data["characters"], dict):
            return {}, {}, False, "❌ 'characters' 必须是字典格式"
        if not isinstance(master_data["recaps"], dict):
            return {}, {}, False, "❌ 'recaps' 必须是字典格式"
        
        # 提取两个核心字典
        global_cast = master_data["characters"]
        custom_recaps = master_data["recaps"]
        return global_cast, custom_recaps, True, "✅ 外脑数据解析成功"
    except json.JSONDecodeError as e:
        return {}, {}, False, f"❌ 外脑 JSON 格式错误：{str(e)}"
    except Exception as e:
        return {}, {}, False, f"❌ 解析失败：{str(e)}"


# --- 🌟 新增：角色试音与定妆室 后端函数 ---

def parse_json_to_cast_state(json_str):
    """解析 Master JSON，提取角色列表并初始化 cast_state。

    Args:
        json_str: Master JSON 字符串，需包含 "characters" 根节点。

    Returns:
        dict: 角色状态字典，格式为
              {"角色名": {"role": ..., "gender": ..., "emotion": ..., "locked": False, "voice_cfg": {...}}, ...}
              解析失败时返回空字典。
    """
    try:
        data = json.loads(json_str)
        characters = data.get("characters", {})
    except Exception:
        return {}

    cast_state = {}
    role_voices = load_role_voices()
    # 🌟 用于未在配置中找到的角色，按顺序分配非弃用音色
    voice_idx = 0

    for char_name, char_info in characters.items():
        if not isinstance(char_info, dict):
            continue

        role = char_info.get("role", "extra")
        if role in role_voices:
            default_voice = role_voices[role]
        else:
            # 未配置的角色从 DEFAULT_VOICE_ORDER 中依次分配
            assigned_voice = DEFAULT_VOICE_ORDER[voice_idx % len(DEFAULT_VOICE_ORDER)]
            voice_idx += 1
            default_voice = {"mode": "preset", "voice": assigned_voice}

        cast_state[char_name] = {
            "role": role,
            "gender": char_info.get("gender", "unknown"),
            "emotion": char_info.get("emotion", "平静"),
            "locked": False,
            "voice_cfg": default_voice,
        }
    return cast_state


def build_voice_cfg_from_ui(mode, preset_voice, clone_file, design_text):
    """根据用户在角色卡片中的选择，组装 voice_cfg 字典。

    Args:
        mode: "预设基底" | "声音克隆" | "文本设计"
        preset_voice: 预设音色下拉值（如 "Eric (默认男声)"）
        clone_file: 上传的克隆参考音频路径
        design_text: 音色设计提示词

    Returns:
        dict: 引擎可用的 voice_cfg
    """
    voice_cfg = {"mode": "preset", "voice": "aiden"}

    if mode == "预设基底":
        voice_id = preset_voice.split(" ")[0].lower() if preset_voice else "aiden"
        voice_cfg = {"mode": "preset", "voice": voice_id}
    elif mode == "声音克隆" and clone_file is not None:
        ref_path = clone_file if isinstance(clone_file, str) else getattr(clone_file, "name", "")
        voice_cfg = {"mode": "clone", "ref_audio": ref_path, "ref_text": ""}
    elif mode == "文本设计" and design_text:
        voice_cfg = {"mode": "design", "instruct": design_text}

    return voice_cfg


def test_single_voice(char_name, mode, preset_voice, clone_file, design_text, test_text):
    """为单个角色生成试听音频。

    组装 voice_cfg 并调用底层 MLXRenderEngine.render_dry_chunk，
    绕过复杂的剧本切片逻辑，仅返回一个 WAV 文件路径。

    Args:
        char_name: 角色名称（用于日志，不影响音色选择）。
        mode: 音色模式，"预设基底" | "声音克隆" | "文本设计"。
        preset_voice: 预设音色下拉值（如 "Eric (默认男声)"）。
        clone_file: 上传的克隆参考音频路径或文件对象。
        design_text: 音色设计提示词。
        test_text: 试听文本内容。

    Returns:
        str or None: 生成的 WAV 文件路径，失败时返回 None。
    """
    voice_cfg = build_voice_cfg_from_ui(mode, preset_voice, clone_file, design_text)

    if not test_text or not test_text.strip():
        test_text = "这是一段录音，请确认是否可以。"

    temp_save_path = os.path.join(
        "./output/Preview", f"test_{uuid.uuid4().hex[:8]}.wav"
    )
    os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)

    try:
        from modules.mlx_tts_engine import MLXRenderEngine

        engine = MLXRenderEngine()
        engine.render_dry_chunk(test_text, voice_cfg, temp_save_path)
        engine.destroy()
        return temp_save_path
    except Exception as e:
        return None


def _persist_clone_ref_audio(voice_cfg, role):
    """将克隆模式的参考音频复制到持久化目录，防止 Gradio 临时文件丢失。

    如果 voice_cfg 为克隆模式且 ref_audio 存在，则拷贝到
    ``assets/voices/role_<role>.wav``，并原地更新 voice_cfg 中的路径。

    Args:
        voice_cfg: 音色配置字典（会被原地修改）。
        role: 角色身份标签（如 m1, f1, narrator）。
    """
    if voice_cfg.get("mode") != "clone":
        return
    ref_audio = voice_cfg.get("ref_audio", "")
    if not ref_audio or not os.path.exists(ref_audio):
        return
    persistent_dir = os.path.join("./assets", "voices")
    os.makedirs(persistent_dir, exist_ok=True)
    ext = os.path.splitext(ref_audio)[1] or ".wav"
    persistent_path = os.path.join(persistent_dir, f"role_{role}{ext}")
    shutil.copy(ref_audio, persistent_path)
    voice_cfg["ref_audio"] = persistent_path


def update_cast_voice_cfg(cast_state, char_name, mode, preset_voice, clone_file, design_text):
    """锁定角色音色：将用户确认的配置写入 cast_state 并标记为 locked。

    Args:
        cast_state: 全局角色状态字典。
        char_name: 要锁定的角色名称。
        mode: 音色模式，"预设基底" | "声音克隆" | "文本设计"。
        preset_voice: 预设音色下拉值。
        clone_file: 克隆参考音频路径或文件对象。
        design_text: 音色设计提示词。

    Returns:
        dict: 更新后的 cast_state（Gradio State 需要返回新值）。
    """
    if not cast_state or char_name not in cast_state:
        return cast_state

    voice_cfg = build_voice_cfg_from_ui(mode, preset_voice, clone_file, design_text)

    # 🎯 触发核心功能：当用户点击锁定时，如果他是男女主，立刻将其音色跨书籍全局固化
    role = cast_state[char_name].get("role", "extra")

    # 🌟 克隆模式：将参考音频持久化到 assets/voices/，防止临时文件丢失
    _persist_clone_ref_audio(voice_cfg, role)

    cast_state[char_name]["voice_cfg"] = voice_cfg
    cast_state[char_name]["locked"] = True

    save_role_voice(role, voice_cfg)

    return cast_state


def unlock_cast_voice_cfg(cast_state, char_name):
    """解锁角色音色：将已锁定的角色标记为未锁定，允许用户继续修改。

    Args:
        cast_state: 全局角色状态字典。
        char_name: 要解锁的角色名称。

    Returns:
        dict: 更新后的 cast_state（Gradio State 需要返回新值）。
    """
    if not cast_state or char_name not in cast_state:
        return cast_state

    cast_state[char_name]["locked"] = False
    return cast_state


def inject_cast_state_into_global_cast(global_cast, cast_state):
    """将用户逐个试听并锁定的 voice_cfg 注入 global_cast，供全本压制使用。

    仅覆盖已锁定的角色配置。

    Args:
        global_cast: 从 Master JSON 解析出的角色字典。
        cast_state: 用户在选角控制台中维护的角色状态字典。

    Returns:
        dict: 注入了已锁定角色音色配置的 global_cast。
    """
    if not cast_state:
        return global_cast
    for char_name, info in cast_state.items():
        if info.get("locked") and char_name in global_cast:
            global_cast[char_name]["voice_cfg"] = info["voice_cfg"]
    return global_cast


# --- 核心逻辑封装 ---
def run_cinecast(epub_file, mode_choice,
                 master_json_str, character_voice_files,
                 preset_voice_selection,
                 narrator_file, ambient_file, chime_file,
                 llm_model_name="", llm_base_url="", llm_api_key="",
                 is_preview=False, cast_state=None, preview_text=None):
    """统一处理入口：试听 / 全本压制"""
    if epub_file is None:
        return None, "❌ 请先上传小说文件"

    # 🌟 新增：触发任务时，静默存档当前工作区状态
    save_workspace(epub_file, mode_choice, master_json_str)

    # 1. 拆包 Master JSON
    global_cast, custom_recaps, success, msg = process_master_json(master_json_str)
    if not success:
        return None, msg

    # 2. 部署通用资产与角色专属音色
    save_uploaded_asset(narrator_file, "narrator.wav", "voices")
    save_uploaded_asset(ambient_file, "iceland_wind.wav", "ambient")
    save_uploaded_asset(chime_file, "soft_chime.wav", "transitions")

    if character_voice_files:
        for file_obj in character_voice_files:
            save_uploaded_asset(file_obj, None, "voices")

    # 3. 提取用户选择的基底音色 ID
    base_voice_id = preset_voice_selection.split(" ")[0].lower() if preset_voice_selection and isinstance(preset_voice_selection, str) else "aiden"

    # 如果外脑 JSON 有旁白角色但未配音色，强制指定基底音色
    if global_cast and isinstance(global_cast.get("旁白"), dict):
        global_cast["旁白"]["voice"] = base_voice_id

    # 🌟 注入用户在选角控制台中锁定的角色音色配置
    if cast_state:
        global_cast = inject_cast_state_into_global_cast(global_cast, cast_state)

    # 4. 组装配置，将拆解后的数据分别注入
    is_pure = "纯净" in mode_choice
    # 🌟 优先使用 UI 界面当前值，回退到本地持久化配置，确保编剧阶段使用用户最新的大模型设置
    saved_llm_cfg = load_llm_config()
    ui_model = (llm_model_name or "").strip()
    ui_base_url = (llm_base_url or "").strip()
    ui_api_key = (llm_api_key or "").strip()
    active_llm_model = ui_model or saved_llm_cfg.get("model_name", "")
    active_llm_base_url = ui_base_url or saved_llm_cfg.get("base_url", "")
    active_llm_api_key = ui_api_key or saved_llm_cfg.get("api_key", "")
    # 🌟 同步持久化最新的 LLM 配置，保证下次启动时也能读到
    if active_llm_model and active_llm_base_url and active_llm_api_key:
        save_llm_config(active_llm_model, active_llm_base_url, active_llm_api_key)
    config = {
        "assets_dir": "./assets",
        "output_dir": "./output/Preview" if is_preview else "./output/Audiobooks",
        "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",
        "ambient_theme": "iceland_wind" if ambient_file else "default",
        "target_duration_min": 30,
        "min_tail_min": 10,
        "use_local_llm": False,
        "pure_narrator_mode": is_pure,
        "global_cast": global_cast,        # 🌟 路由给 LLM 导演选角用
        "custom_recaps": custom_recaps,    # 🌟 路由给主控程序拼接摘要用
        "enable_auto_recap": False,        # 默认关闭本地摘要，彻底依赖外脑
        "enable_recap": bool(custom_recaps),  # 有摘要数据时自动启用
        "user_recaps": None,               # 兼容旧版配置
        "default_narrator_voice": base_voice_id,  # 🌟 注入底层 TTS 引擎
        "llm_model_name": active_llm_model,       # 🌟 用户配置的大模型名称（UI 实时值优先）
        "llm_base_url": active_llm_base_url,      # 🌟 用户配置的 Base URL（UI 实时值优先）
        "llm_api_key": active_llm_api_key,         # 🌟 用户配置的 API Key（UI 实时值优先）
    }

    try:
        producer = CineCastProducer(config=config)
        if is_preview:
            mp3 = producer.run_preview_mode(epub_file.name, preview_text=preview_text)
            return mp3, "✅ 试听生成成功！(已应用全局外脑设定)"
        else:
            if producer.phase_1_generate_scripts(epub_file.name):
                producer.phase_2_render_dry_audio()
                producer.phase_3_cinematic_mix()
                # 🌟 混音完成后自动进行无头质检
                qc_report = run_headless_qc(config["output_dir"])
                return None, "✅ 全本压制完成！\n\n" + qc_report
            return None, "❌ 阶段一（微切片剧本生成）失败，请检查输入文件和服务状态。"
    except Exception as e:
        return None, f"❌ 错误: {str(e)}"


# --- Web UI 界面构建 ---
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")

# 🌟 启动前加载上次存档
last_state = load_workspace()
saved_llm = load_llm_config()

with gr.Blocks(title="CineCast Pro 3.0") as ui:
    gr.Markdown("# 🎬 CineCast Pro 电影级有声书制片厂")
    gr.Markdown("上传你的小说，定义你的声场，一键压制具备沉浸式体验的电影级有声书。")

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("### 📖 第一步：剧本与模式")
                # 🌟 从存档恢复上次文件（验证文件是否还存在）
                saved_file = last_state.get("book_file")
                default_file = saved_file if saved_file and os.path.exists(saved_file) else None
                book_file = gr.File(
                    label="上传小说 (EPUB/TXT)",
                    file_types=[".epub", ".txt"],
                    value=default_file,
                )
                mode_selector = gr.Radio(
                    choices=[
                        "🎙️ 纯净旁白模式",
                        "🎭 智能配音模式 (外脑控制版)",
                    ],
                    value=last_state.get("mode", "🎙️ 纯净旁白模式"),
                    label="制作模式",
                )

            # 🌟 大一统外脑控制台（根据上次保存的模式动态设置可见性）
            init_brain_visible = "智能配音" in last_state.get("mode", "")
            with gr.Accordion("🧠 第二步：云端外脑控制台 (Brain Node)", open=True, visible=init_brain_visible) as brain_panel:
                gr.Markdown("您可以粘贴 Master JSON，**或者**直接配置大模型 API 让系统自动呼叫。")

                # --- 新增：大模型直连配置区 ---
                with gr.Group():
                    gr.Markdown("#### 🔌 Custom LLM 在线剧本分析")
                    with gr.Row():
                        llm_model = gr.Textbox(label="模型名称 (如 qwen3.5-plus)", value=saved_llm.get("model_name", "qwen-plus"), scale=1)
                        llm_baseurl = gr.Textbox(label="Base URL (包含 /v1)", value=saved_llm.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"), scale=2)
                        llm_apikey = gr.Textbox(label="API Key", type="password", value=saved_llm.get("api_key", ""), placeholder="sk-...", scale=2)

                    btn_test_llm = gr.Button("🔄 测试大模型连接", variant="secondary")
                    llm_status = gr.Textbox(label="测试结果", interactive=False, lines=1)

                    btn_test_llm.click(
                        fn=test_llm_connection,
                        inputs=[llm_model, llm_baseurl, llm_apikey],
                        outputs=[llm_status],
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        master_json = gr.Textbox(
                            label="或者手动粘贴 Master JSON (若配置了上方LLM，可留空由程序自动生成)",
                            placeholder='{\n  "characters": {...},\n  "recaps": {...}\n}',
                            lines=10,
                            value=last_state.get("master_json", ""),
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### 专属音色注入")
                        gr.Markdown("请上传角色音色文件，**文件名必须与 JSON 中的角色标准名一致** (如 `老渔夫.wav`)。系统将自动完成声纹绑定。")
                        char_voice_files = gr.File(
                            label="批量上传角色音色 (.wav)",
                            file_count="multiple",
                            file_types=[".wav"],
                        )

                with gr.Accordion("💡 复制此 Prompt 给外部大模型", open=False):
                    gr.Code(
                        value=BRAIN_PROMPT_TEMPLATE,
                        language="markdown",
                    )

            # 🌟 角色试音与定妆室：存放当前所有角色状态的全局变量
            cast_state = gr.State({})

            with gr.Accordion("🎭 角色试音与定妆室 (选角控制台)", open=True, visible=init_brain_visible) as audition_panel:
                gr.Markdown("解析 Master JSON 后，可为每个角色独立试听、切换音色模式、确认锁定。所有角色锁定后方可全本压制。")

                with gr.Row():
                    btn_parse_cast = gr.Button("🔍 解析角色列表", variant="secondary")
                    cast_parse_status = gr.Textbox(label="解析状态", interactive=False, scale=2)

                def _parse_and_update(json_str):
                    state = parse_json_to_cast_state(json_str)
                    if state:
                        names = ", ".join(state.keys())
                        return state, f"✅ 已解析 {len(state)} 个角色：{names}"
                    return {}, "❌ 解析失败，请检查 Master JSON 格式。"

                btn_parse_cast.click(
                    fn=_parse_and_update,
                    inputs=master_json,
                    outputs=[cast_state, cast_parse_status],
                )

                # 🌟 核心：使用 @gr.render 动态生成角色调音卡片
                @gr.render(inputs=cast_state)
                def render_character_cards(characters):
                    if not characters:
                        gr.Markdown("*暂无角色，请先在上方粘贴 Master JSON 并点击「解析角色列表」。*")
                        return

                    for char_name, char_info in characters.items():
                        locked = char_info.get("locked", False)
                        voice_cfg = char_info.get("voice_cfg", {})
                        saved_mode = voice_cfg.get("mode", "preset")

                        # 🌟 根据已保存的 voice_cfg 还原 UI 显示值
                        if saved_mode == "clone":
                            mode_default = "声音克隆"
                        elif saved_mode == "design":
                            mode_default = "文本设计"
                        else:
                            mode_default = "预设基底"

                        # 还原预设音色下拉值
                        preset_default = "Aiden"
                        if saved_mode == "preset":
                            saved_voice_id = voice_cfg.get("voice", "aiden")
                            for v in QWEN_PRESET_VOICES:
                                if v.split(" ")[0].lower() == saved_voice_id.lower():
                                    preset_default = v
                                    break

                        # 还原克隆参考音频路径
                        clone_default = voice_cfg.get("ref_audio", None) if saved_mode == "clone" else None

                        # 还原音色设计提示词
                        design_default = voice_cfg.get("instruct", "") if saved_mode == "design" else ""

                        with gr.Group():
                            with gr.Row():
                                lock_icon = "🔒" if locked else "🗣️"
                                gr.Markdown(f"### {lock_icon} {char_name}")
                                gr.Markdown(
                                    f"*设定：{char_info.get('gender', '未知')} / {char_info.get('emotion', '无')}*"
                                )

                            with gr.Row():
                                # --- 左侧：音色调优参数 ---
                                with gr.Column(scale=2):
                                    mode_radio = gr.Radio(
                                        ["预设基底", "声音克隆", "文本设计"],
                                        value=mode_default,
                                        label="音色生成模式",
                                        interactive=(not locked),
                                    )

                                    preset_dropdown = gr.Dropdown(
                                        choices=QWEN_PRESET_VOICES,
                                        value=preset_default,
                                        label="选择无口音预设",
                                        visible=(mode_default == "预设基底"),
                                        interactive=(not locked),
                                    )
                                    clone_upload = gr.File(
                                        label="上传参考干音 (.wav)",
                                        visible=(mode_default == "声音克隆"),
                                        file_types=[".wav"],
                                        value=clone_default,
                                        interactive=(not locked),
                                    )
                                    design_prompt = gr.Textbox(
                                        label="音色设计提示词 (英/中)",
                                        visible=(mode_default == "文本设计"),
                                        value=design_default,
                                        interactive=(not locked),
                                    )

                                    def toggle_mode(m):
                                        return [
                                            gr.update(visible=(m == "预设基底")),
                                            gr.update(visible=(m == "声音克隆")),
                                            gr.update(visible=(m == "文本设计")),
                                        ]

                                    mode_radio.change(
                                        toggle_mode,
                                        inputs=mode_radio,
                                        outputs=[preset_dropdown, clone_upload, design_prompt],
                                    )

                                # --- 右侧：独立试听沙盒 ---
                                with gr.Column(scale=3):
                                    test_text = gr.Textbox(
                                        value="这是一段录音，请确认是否可以。",
                                        label="试听文本 (可自由编辑)",
                                        interactive=(not locked),
                                    )
                                    with gr.Row():
                                        btn_test = gr.Button("🎧 生成试听", variant="secondary")
                                        btn_lock = gr.Button(
                                            "🔓 解锁修改" if locked else "✅ 确认使用此音色",
                                            variant="primary",
                                        )

                                    card_audio_player = gr.Audio(label="试听结果", interactive=False)

                                    # 绑定试听逻辑
                                    btn_test.click(
                                        fn=test_single_voice,
                                        inputs=[
                                            gr.State(char_name),
                                            mode_radio,
                                            preset_dropdown,
                                            clone_upload,
                                            design_prompt,
                                            test_text,
                                        ],
                                        outputs=card_audio_player,
                                    )

                                    # 🌟 锁定/解锁切换逻辑
                                    def _toggle_lock(state, locked_char, mode_val, preset_val, clone_val, design_val):
                                        # 深拷贝 state，确保返回新对象以触发 @gr.render 重新渲染
                                        state = copy.deepcopy(state)
                                        if state.get(locked_char, {}).get("locked", False):
                                            # 当前已锁定 → 解锁，允许用户继续修改
                                            state = unlock_cast_voice_cfg(state, locked_char)
                                        else:
                                            # 当前未锁定 → 锁定并保存配置
                                            state = update_cast_voice_cfg(
                                                state, locked_char, mode_val, preset_val, clone_val, design_val
                                            )
                                        return state

                                    btn_lock.click(
                                        fn=_toggle_lock,
                                        inputs=[cast_state, gr.State(char_name), mode_radio, preset_dropdown, clone_upload, design_prompt],
                                        outputs=[cast_state],
                                    )

            with gr.Accordion("🎛️ 第三步：通用声场与旁白", open=False):
                with gr.Row():
                    preset_voice_dropdown = gr.Dropdown(
                        label="默认旁白基底音色 (Qwen3-TTS Preset)",
                        choices=QWEN_PRESET_VOICES,
                        value="Aiden",
                    )
                    narrator_audio = gr.Audio(label="或上传旁白克隆音 (Narrator)", type="filepath")
                with gr.Row():
                    ambient_audio = gr.Audio(label="环境音 (Ambient)", type="filepath")
                    chime_audio = gr.Audio(label="转场音 (Chime)", type="filepath")

            # 🌟 极速试听：首章前10句预览与编辑
            with gr.Accordion("🎧 极速试听 (首章前10句预览)", open=True):
                gr.Markdown("点击「提取」自动获取首章前10句，可自由编辑后再生成试听音频。")
                with gr.Row():
                    btn_extract_preview = gr.Button("📖 提取首章前10句", variant="secondary")
                preview_text = gr.Textbox(
                    label="试听文本 (可自由编辑)",
                    lines=8,
                    placeholder="点击上方「提取」按钮或手动输入试听文本...",
                )

                btn_extract_preview.click(
                    fn=extract_preview_sentences,
                    inputs=[book_file],
                    outputs=[preview_text],
                )

            with gr.Row():
                btn_preview = gr.Button(
                    "🎧 极速试听 (首章前10句)", variant="secondary", size="lg"
                )
                btn_full = gr.Button(
                    "🚀 全本压制", variant="primary", size="lg"
                )

        with gr.Column(scale=3):
            gr.Markdown("### 🎵 审听室")
            audio_player = gr.Audio(label="审听室播放器", interactive=False)
            status_box = gr.Textbox(
                label="制片状态", lines=8, interactive=False
            )
            log_viewer = gr.Textbox(
                label="📋 实时制片日志", lines=15, interactive=False
            )
            # 🌟 每 2 秒自动轮询日志文件并刷新展示
            timer = gr.Timer(2)
            timer.tick(get_logs, outputs=log_viewer)

            gr.Markdown("---")
            gr.Markdown(
                """
            ### 💡 操作指南：
            1. **纯净旁白模式**：完全绕过大模型，按标点切分，速度极快，适合严肃文学和网文。
            2. **智能配音模式**：将全书发给外部大模型，一次性获取角色设定与前情提要的 Master JSON，粘贴即可。
            3. **选角控制台**：解析 JSON 后，可为每个角色独立试听三种音色模式（预设/克隆/设计），确认后锁定。
            4. **试听功能**：强烈建议在全本压制前，先点击【极速试听】确认音色与混音比例。
            5. **断点续传**：如果在压制途中停止，再次点击全本压制，系统会自动跳过已生成的音频。
            """
            )

    # --- 动态交互逻辑 ---
    def on_mode_change(mode):
        is_cast_mode = "智能配音" in mode
        return gr.update(visible=is_cast_mode), gr.update(visible=is_cast_mode)

    mode_selector.change(on_mode_change, mode_selector, [brain_panel, audition_panel])

    # --- 按钮绑定 ---
    inputs_list = [
        book_file,
        mode_selector,
        master_json,
        char_voice_files,
        preset_voice_dropdown,
        narrator_audio,
        ambient_audio,
        chime_audio,
        llm_model,
        llm_baseurl,
        llm_apikey,
    ]

    btn_preview.click(
        fn=lambda *args: run_cinecast(*args[:-2], is_preview=True, cast_state=args[-2], preview_text=args[-1]),
        inputs=inputs_list + [cast_state, preview_text],
        outputs=[audio_player, status_box],
    )

    btn_full.click(
        fn=lambda *args: run_cinecast(*args[:-1], is_preview=False, cast_state=args[-1]),
        inputs=inputs_list + [cast_state],
        outputs=[audio_player, status_box],
    )

# --- 新增：流式API独立界面 ---
with gr.Blocks(title="CineCast 流式朗读API") as stream_ui:
    gr.Markdown("# 🎵 CineCast 流式实时朗读API")
    gr.Markdown("""
    实时文本转语音服务，支持动态音色切换和音色克隆功能。
    可以实现"边读边推"的流式体验。
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # API连接测试
            btn_test_stream_api = gr.Button("🔄 测试流式API连接", variant="secondary")
            stream_api_status = gr.Textbox(label="API状态", interactive=False, lines=2)
            
            # 音色管理
            gr.Markdown("### 🎤 音色管理")
            with gr.Tab("预设音色"):
                preset_voice_selector = gr.Dropdown(
                    label="选择预设音色",
                    choices=["aiden", "dylan", "ono_anna", "ryan", "sohee", "uncle_fu", "vivian", "eric", "serena"],
                    value="aiden"
                )
                btn_set_preset_voice = gr.Button("✅ 使用此音色", variant="primary")
            
            with gr.Tab("音色克隆"):
                clone_upload = gr.File(
                    label="上传参考音频 (WAV/MP3/FLAC)",
                    file_types=[".wav", ".mp3", ".flac"]
                )
                clone_voice_name = gr.Textbox(
                    label="音色名称",
                    placeholder="给这个音色起个名字..."
                )
                btn_clone_voice = gr.Button("🎯 克隆音色", variant="primary")
            
            voice_status = gr.Textbox(label="音色状态", interactive=False, lines=2)
            
            # 实时朗读
            gr.Markdown("### 📖 实时朗读")
            stream_text_area = gr.TextArea(
                label="朗读文本",
                placeholder="请输入要朗读的文本内容...",
                lines=5
            )
            stream_language = gr.Radio(
                choices=[("中文", "zh"), ("English", "en")],
                value="zh",
                label="语言选择"
            )
            btn_start_stream = gr.Button("▶️ 开始流式朗读", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            stream_audio_player = gr.Audio(
                label="实时音频输出",
                interactive=False,
                autoplay=True
            )
            stream_progress = gr.Progress()
            stream_logs = gr.Textbox(
                label="实时日志",
                interactive=False,
                lines=8,
                max_lines=10
            )
    
    # 事件绑定
    btn_test_stream_api.click(
        fn=test_stream_api_connection,
        inputs=[],
        outputs=stream_api_status
    )
    
    btn_set_preset_voice.click(
        fn=lambda voice: set_stream_voice(voice),
        inputs=[preset_voice_selector],
        outputs=voice_status
    )
    
    btn_clone_voice.click(
        fn=lambda file, name: set_stream_voice(name, file) if file else "请上传音频文件",
        inputs=[clone_upload, clone_voice_name],
        outputs=voice_status
    )
    
    def stream_read_handler(text, language):
        if not text.strip():
            return None, "❌ 请输入朗读文本", gr.update()
        
        log_updates = ["🎙️ 开始流式朗读..."]
        yield None, "\n".join(log_updates), gr.update(value=0.1)
        
        # 调用流式API
        audio_path = stream_tts_read(text, language)
        
        if audio_path:
            log_updates.append("✅ 音频生成完成!")
            yield audio_path, "\n".join(log_updates), gr.update(value=1.0)
        else:
            log_updates.append("❌ 朗读失败，请检查连接")
            yield None, "\n".join(log_updates), gr.update()
    
    btn_start_stream.click(
        fn=stream_read_handler,
        inputs=[stream_text_area, stream_language],
        outputs=[stream_audio_player, stream_logs]
    )

# 启动选项
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["main", "stream"], default="main",
                       help="选择启动模式: main(主界面) 或 stream(流式API界面)")
    args = parser.parse_args()
    
    if args.mode == "stream":
        print("🚀 启动流式API界面...")
        stream_ui.launch(inbrowser=True, server_name="127.0.0.1", server_port=7861)
    else:
        print("🎬 启动主制片界面...")
        ui.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860, theme=theme)
