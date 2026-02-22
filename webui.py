#!/usr/bin/env python3
"""
CineCast Web UI
基于 Gradio Blocks API 的现代化图形界面
支持纯净旁白/智能配音双模式、云端外脑 Master JSON 统一输入、极速试听与全本压制
包含：工作区断点记忆与自动恢复功能、实时制片日志流式展示、自动质检
"""

import os
import json
import shutil
import requests
import gradio as gr
from main_producer import CineCastProducer

# Qwen3-TTS 官方支持的预设音色列表
QWEN_PRESET_VOICES = [
    "Eric (默认男声)", "Serena (默认女声)",
    "Aiden", "Dylan", "Ono_anna", "Ryan", "Sohee", "Uncle_fu", "Vivian",
]


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
            api_endpoint, json=payload, headers=headers, timeout=10
        )

        if response.status_code == 200:
            return f"✅ 连接成功！已成功握手 {model_name}。"
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
请你通读全书，完成【角色选角】与【前情提要】两项核心任务，并严格按要求的 JSON 格式输出。

【任务一：建立全局角色设定集 (Character Bible)】
1. 提取所有有台词的角色，将他们同一个人的不同称呼统一为一个【标准名】（如"老李"统一为"李局长"）。
2. 推断角色的性别（male/female）和声音特质情感（如：沉稳、沧桑、活泼、阴冷等）。
3. 必须包含一个名为 "路人" 的特殊角色，用于兜底那些只有一两句台词的群演。

【任务二：撰写各章前情提要 (Recaps)】
1. 为**除第一章以外**的每一章，生成一段用于片头播报的前情提要（80-120字）。
2. 语言必须高度凝练，具有美剧片头的电影感。
3. 最后一句必须是一个引出本章内容的"悬念钩子"。

【⚠️ 格式生死攸关 ⚠️】
你必须且只能输出一个合法的纯 JSON 字典格式！包含 "characters" 和 "recaps" 两个根节点。
绝对不要输出任何 markdown 标记（如 ```json），不要包含任何解释性废话，直接输出大括号包裹的 JSON！

【输出格式示例】
{
  "characters": {
    "旁白": {"gender": "male", "emotion": "平静"},
    "老渔夫": {"gender": "male", "emotion": "沧桑"},
    "艾米莉": {"gender": "female", "emotion": "活泼"},
    "路人": {"gender": "unknown", "emotion": "平淡"}
  },
  "recaps": {
    "Chapter_002": "上一章中，老渔夫在暴风雪中带回了一个神秘的黑匣子……然而他没意识到，危险才刚刚降临。",
    "Chapter_003": "警长的调查陷入僵局，唯一的目击者却在昨夜离奇失踪……一通电话突然打进了警局。"
  }
}"""


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


# --- 核心逻辑封装 ---
def run_cinecast(epub_file, mode_choice,
                 master_json_str, character_voice_files,
                 preset_voice_selection,
                 narrator_file, ambient_file, chime_file,
                 is_preview=False):
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
    base_voice_id = preset_voice_selection.split(" ")[0].lower() if preset_voice_selection else "eric"

    # 如果外脑 JSON 有旁白角色但未配音色，强制指定基底音色
    if global_cast and "旁白" in global_cast:
        global_cast["旁白"]["voice"] = base_voice_id

    # 4. 组装配置，将拆解后的数据分别注入
    is_pure = "纯净" in mode_choice
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
    }

    try:
        producer = CineCastProducer(config=config)
        if is_preview:
            mp3 = producer.run_preview_mode(epub_file.name)
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
                        llm_model = gr.Textbox(label="模型名称 (如 qwen3.5-plus)", value="qwen-plus", scale=1)
                        llm_baseurl = gr.Textbox(label="Base URL (包含 /v1)", value="https://dashscope.aliyuncs.com/compatible-mode/v1", scale=2)
                        llm_apikey = gr.Textbox(label="API Key", type="password", placeholder="sk-...", scale=2)

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

            with gr.Accordion("🎛️ 第三步：通用声场与旁白", open=False):
                with gr.Row():
                    preset_voice_dropdown = gr.Dropdown(
                        label="默认旁白基底音色 (Qwen3-TTS Preset)",
                        choices=QWEN_PRESET_VOICES,
                        value="Eric (默认男声)",
                    )
                    narrator_audio = gr.Audio(label="或上传旁白克隆音 (Narrator)", type="filepath")
                with gr.Row():
                    ambient_audio = gr.Audio(label="环境音 (Ambient)", type="filepath")
                    chime_audio = gr.Audio(label="转场音 (Chime)", type="filepath")

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
            3. **试听功能**：强烈建议在全本压制前，先点击【极速试听】确认音色与混音比例。
            4. **断点续传**：如果在压制途中停止，再次点击全本压制，系统会自动跳过已生成的音频。
            """
            )

    # --- 动态交互逻辑 ---
    def on_mode_change(mode):
        is_cast_mode = "智能配音" in mode
        return gr.update(visible=is_cast_mode)

    mode_selector.change(on_mode_change, mode_selector, brain_panel)

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
    ]

    btn_preview.click(
        fn=lambda *args: run_cinecast(*args, is_preview=True),
        inputs=inputs_list,
        outputs=[audio_player, status_box],
    )

    btn_full.click(
        fn=lambda *args: run_cinecast(*args, is_preview=False),
        inputs=inputs_list,
        outputs=[audio_player, status_box],
    )

if __name__ == "__main__":
    ui.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860, theme=theme)
