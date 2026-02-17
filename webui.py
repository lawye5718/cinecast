#!/usr/bin/env python3
"""
CineCast Web UI
基于 Gradio Blocks API 的现代化图形界面
支持纯净旁白/智能配音双模式、自定义音色上传、极速试听与全本压制
"""

import os
import shutil
import gradio as gr
from main_producer import CineCastProducer


# --- 辅助函数：保存用户上传的资产 ---
def save_uploaded_asset(file_path, target_filename, folder):
    """将用户上传的音频文件复制到 assets 目录的指定子文件夹"""
    if file_path is None:
        return
    target_dir = os.path.join("./assets", folder)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, target_filename)
    shutil.copy(file_path, target_path)


# --- 核心逻辑封装 ---
def process_audio(epub_file, mode_choice, narrator_file,
                  m1_file, m2_file, f1_file, f2_file,
                  ambient_file, chime_file, is_preview=False):
    """统一处理入口：试听 / 全本压制"""
    if epub_file is None:
        return None, "❌ 请先上传小说文件 (EPUB/TXT)"

    # 1. 保存用户覆盖的资产
    save_uploaded_asset(narrator_file, "narrator.wav", "voices")
    save_uploaded_asset(m1_file, "m1.wav", "voices")
    save_uploaded_asset(m2_file, "m2.wav", "voices")
    save_uploaded_asset(f1_file, "f1.wav", "voices")
    save_uploaded_asset(f2_file, "f2.wav", "voices")
    save_uploaded_asset(ambient_file, "iceland_wind.wav", "ambient")
    save_uploaded_asset(chime_file, "soft_chime.wav", "transitions")

    # 2. 组装配置
    is_pure_narrator = "纯净" in mode_choice
    config = {
        "assets_dir": "./assets",
        "output_dir": "./output/Preview" if is_preview else "./output/Audiobooks",
        "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",
        "ambient_theme": "iceland_wind",
        "target_duration_min": 30,
        "min_tail_min": 10,
        "use_local_llm": True,
        "enable_recap": not is_pure_narrator,
        "pure_narrator_mode": is_pure_narrator,
    }

    try:
        producer = CineCastProducer(config=config)

        # 电影配音模式下，将用户上传的角色音色传递给资产管理器
        if not is_pure_narrator:
            role_voices = {
                "narrator": narrator_file,
                "m1": m1_file,
                "m2": m2_file,
                "f1": f1_file,
                "f2": f2_file,
            }
            producer.assets.set_custom_role_voices(role_voices)

        # 🌟 试听模式：拦截长篇，只处理第一章的前10句话
        if is_preview:
            preview_mp3_path = producer.run_preview_mode(epub_file.name)
            return preview_mp3_path, "✅ 试听生成成功！请点击播放。"

        # 🚀 全本压制模式：必须严格按 微切片 → 渲染 → 混音 三阶段串行执行
        if producer.phase_1_generate_scripts(epub_file.name):
            producer.phase_2_render_dry_audio()
            producer.phase_3_cinematic_mix()
            return None, f"✅ 全本压制完成！请前往 {config['output_dir']} 目录查看。"
        return None, "❌ 阶段一（微切片剧本生成）失败，请检查输入文件和服务状态。"

    except Exception as e:
        return None, f"❌ 发生错误: {e}"


# --- Web UI 界面构建 ---
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")

with gr.Blocks(theme=theme, title="CineCast 电影级有声书") as ui:
    gr.Markdown("# 🎬 CineCast 电影级有声书工业制片厂")
    gr.Markdown("上传你的小说，定义你的声场，一键压制具备沉浸式体验的电影级有声书。")

    with gr.Row():
        # 左侧：配置面板
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 📖 第一步：导入剧本与模式")
                book_file = gr.File(
                    label="上传小说 (支持 .epub 或 .txt)",
                    file_types=[".epub", ".txt"],
                )
                mode_selector = gr.Radio(
                    choices=[
                        "🎙️ 纯净旁白模式 (单音色/秒级解析/100%忠实原文)",
                        "🎭 智能配音模式 (LLM多角色演绎/自动前情摘要)",
                    ],
                    value="🎙️ 纯净旁白模式 (单音色/秒级解析/100%忠实原文)",
                    label="选择制作模式",
                )

            with gr.Group():
                gr.Markdown("### 🗣️ 第二步：选角与音色 (可选)")
                gr.Markdown("*如果不上传，将自动使用系统内置的高优预设音色。当角色数量超过已上传的音色数量时，系统会自动随机分配一个音色，并在全书中保持该分配不变。*")
                narrator_audio = gr.Audio(label="旁白音色样本 (Narrator)", type="filepath")

                # 动态隐藏/显示的配音角色面板
                with gr.Column(visible=False) as role_voices_panel:
                    with gr.Row():
                        f1_audio = gr.Audio(label="女声1 (f1)", type="filepath")
                        m1_audio = gr.Audio(label="男声1 (m1)", type="filepath")
                    with gr.Row():
                        f2_audio = gr.Audio(label="女声2 (f2)", type="filepath")
                        m2_audio = gr.Audio(label="男声2 (m2)", type="filepath")

            with gr.Group():
                gr.Markdown("### 🎛️ 第三步：环境声场 (可选)")
                with gr.Row():
                    ambient_audio = gr.Audio(
                        label="背景环境音 (Ambient BGM)", type="filepath"
                    )
                    chime_audio = gr.Audio(
                        label="过渡提示音 (Transition Chime)", type="filepath"
                    )

            with gr.Row():
                btn_preview = gr.Button(
                    "🎧 生成试听 (前10句)", variant="secondary", size="lg"
                )
                btn_full = gr.Button(
                    "🚀 开始全本压制", variant="primary", size="lg"
                )

        # 右侧：结果与播放面板
        with gr.Column(scale=3):
            gr.Markdown("### 🎵 审听室")
            audio_player = gr.Audio(label="试听成品预览", interactive=False)
            status_box = gr.Textbox(
                label="系统状态日志", lines=5, interactive=False
            )

            gr.Markdown("---")
            gr.Markdown(
                """
            ### 💡 操作指南：
            1. **纯净旁白模式**：完全绕过大模型，按标点切分，速度极快，适合严肃文学和网文。
            2. **试听功能**：强烈建议在全本压制前，先点击【生成试听】，系统会在15秒内合成前10句话供您确认音色与混音比例。
            3. **断点续传**：如果在压制途中停止，再次点击全本压制，系统会自动跳过已生成的音频。
            """
            )

    # --- 动态交互逻辑 ---
    def toggle_mode(choice):
        """纯净模式下隐藏男女主音色上传框"""
        if "纯净" in choice:
            return gr.update(visible=False)
        return gr.update(visible=True)

    mode_selector.change(
        fn=toggle_mode, inputs=mode_selector, outputs=role_voices_panel
    )

    # --- 按钮绑定 ---
    btn_preview.click(
        fn=lambda a, b, c, d, e, f, g, h, i: process_audio(
            a, b, c, d, e, f, g, h, i, is_preview=True
        ),
        inputs=[
            book_file,
            mode_selector,
            narrator_audio,
            m1_audio,
            m2_audio,
            f1_audio,
            f2_audio,
            ambient_audio,
            chime_audio,
        ],
        outputs=[audio_player, status_box],
    )

    btn_full.click(
        fn=lambda a, b, c, d, e, f, g, h, i: process_audio(
            a, b, c, d, e, f, g, h, i, is_preview=False
        ),
        inputs=[
            book_file,
            mode_selector,
            narrator_audio,
            m1_audio,
            m2_audio,
            f1_audio,
            f2_audio,
            ambient_audio,
            chime_audio,
        ],
        outputs=[audio_player, status_box],
    )

if __name__ == "__main__":
    ui.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860)
