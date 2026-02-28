#!/usr/bin/env python3
"""
CineCast 流式API - 最终成功版本
基于已验证的工作方法实现
"""

import sys
import os
from pathlib import Path

# 使用相对路径避免硬编码
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlx.core as mx
import numpy as np
import soundfile as sf
import io
import logging
import time
import re
import asyncio  # 🚨 新增：用于异步线程管控
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="tiktoken")

from pydub import AudioSegment
import numpy as np
import mlx.core as mx

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response  # 🚨 替换 StreamingResponse

# 导入项目模块
from modules.mlx_tts_engine import CinecastMLXEngine as MLXTTSEngine
from modules.asset_manager import AssetManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(title="CineCast Streaming TTS API - Production Ready")

# =====================================================================
# 🌟 修复一：无缝集成原有 Gradio WebUI，共用模型与显存
# =====================================================================
try:
    import gradio as gr
    import traceback  # 🚨 新增导入用于打印详细错误
    
    # 拦截旧版网页中的 launch 防止阻塞
    _original_launch = gr.Blocks.launch
    gr.Blocks.launch = lambda self, *args, **kwargs: None
    
    logger.info("正在尝试导入旧版 webui...")
    import webui  # 🚨 如果 webui.py 里面还有语法错误，这里就会抛出异常
    logger.info(f"webui模块导入成功，可用属性: {[attr for attr in dir(webui) if not attr.startswith('_')][:10]}")
    
    gr.Blocks.launch = _original_launch # 恢复原方法
    
    # 动态寻找实例名称（兼容 demo, app, interface 等常见命名）
    gradio_app_instance = None
    logger.info("开始搜索Gradio实例...")
    if hasattr(webui, 'demo'):
        gradio_app_instance = webui.demo
        logger.info("找到demo实例")
    elif hasattr(webui, 'app') and isinstance(webui.app, gr.Blocks):
        gradio_app_instance = webui.app
        logger.info("找到app实例")
    elif hasattr(webui, 'interface'):
        gradio_app_instance = webui.interface
        logger.info("找到interface实例")
    elif hasattr(webui, 'ui') and isinstance(webui.ui, gr.Blocks):
        gradio_app_instance = webui.ui
        logger.info("找到ui实例")
    elif hasattr(webui, 'stream_ui') and isinstance(webui.stream_ui, gr.Blocks):
        gradio_app_instance = webui.stream_ui
        logger.info("找到stream_ui实例")
    else:
        logger.warning("未找到任何Gradio实例")
        logger.info(f"webui模块中的Blocks对象: {[attr for attr in dir(webui) if isinstance(getattr(webui, attr, None), gr.Blocks)]}")
        
    if gradio_app_instance:
        app = gr.mount_gradio_app(app, gradio_app_instance, path="/webui")
        logger.info("✅ 原有 Cinecast 网页端已成功挂载！请访问 http://localhost:8888/webui/ (注意末尾的斜杠)")
    else:
        logger.warning("⚠️ 成功导入 webui.py，但在里面没有找到名为 demo / app 的 Gradio 实例。")
        
except Exception as e:
    logger.error(f"❌ 挂载原有网页端发生致命错误: {e}")
    # 🚨 打印完整的报错堆栈，帮我们准确定位 webui.py 里面还剩哪个毒瘤！
    logger.error(traceback.format_exc())

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI TTS 兼容请求模型
class OpenAITTSRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str = "aiden"
    response_format: str = "mp3"
    speed: float = 1.0

# 全局状态
class VoiceContext:
    def __init__(self):
        # 默认使用原生配置文件中的旁白设定，或 aiden
        self.current_voice = "aiden" 
        self.engine = None
        self.asset_manager = None
        self.is_ready = False
    
    async def initialize(self):
        """初始化引擎"""
        if not self.is_ready:
            try:
                self.asset_manager = AssetManager()
                # 使用已验证可以工作的模型路径
                self.engine = MLXTTSEngine(
                    model_path="./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
                )
                self.is_ready = True
                logger.info("✅ 流式API引擎初始化成功")
            except Exception as e:
                pass
    
    def get_voice_feature(self, voice_name: str):
        """🌟 架构回归：利用原生的 AssetManager 解析特征，完美支持克隆"""
        if not self.is_ready:
            return {"mode": "preset", "voice": "aiden"}
            
        try:
            # AssetManager 原本就能识别 .cinecast_role_voices.json 里的克隆记录
            return self.asset_manager.load_role(voice_name)
        except Exception as e:
            logger.warning(f"音色 {voice_name} 未在项目中找到，回退到默认: {e}")
            return {"mode": "preset", "voice": "aiden"}
            logger.error(f"❌ 引擎初始化失败: {e}")
            raise

# 全局上下文
voice_context = VoiceContext()

@app.on_event("startup")
async def startup_event():
    """应用启动初始化"""
    await voice_context.initialize()

@app.get("/")
async def root():
    return {
        "message": "CineCast Streaming TTS API - Production Ready",
        "status": "running",
        "ready": voice_context.is_ready
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ready": voice_context.is_ready,
        "current_voice": voice_context.current_voice
    }

@app.get("/voices")
async def list_voices():
    """列出可用音色"""
    if not voice_context.is_ready:
        return {"error": "Service not ready"}
    
    preset_voices = ["aiden", "dylan", "ono_anna", "ryan", "sohee", "uncle_fu", "vivian", "eric", "serena"]
    return {
        "preset_voices": preset_voices,
        "current_voice": voice_context.current_voice
    }

@app.post("/set_voice")
async def set_voice(voice_name: str = Form(...)):
    """设置当前音色"""
    if not voice_context.is_ready:
        return {"error": "Service not ready"}
    
    try:
        # 验证音色名称
        valid_voices = ["aiden", "dylan", "ono_anna", "ryan", "sohee", "uncle_fu", "vivian", "eric", "serena"]
        if voice_name.lower() not in valid_voices:
            return {"error": f"Invalid voice name. Valid options: {valid_voices}"}
        
        voice_context.current_voice = voice_name.lower()
        logger.info(f"✅ 音色已设置为: {voice_context.current_voice}")
        
        return {
            "status": "success",
            "voice_name": voice_context.current_voice
        }
    except Exception as e:
        logger.error(f"❌ 设置音色失败: {e}")
        return {"error": str(e)}



# =====================================================================
# 🌟 修复二：专为 Anxreader 等阅读 App 设计的单头整段响应架构
# =====================================================================
@app.post("/v1/audio/speech")
async def openai_compatible_tts(request: Request, body: OpenAITTSRequest):
    if not voice_context.is_ready:
        raise HTTPException(status_code=503, detail="TTS 服务未就绪")
    
    try:
        feature = voice_context.get_voice_feature(body.voice)
        
        # 暴力清洗特殊符号
        safe_text = re.sub(r'[…]+', '。', body.input)
        safe_text = re.sub(r'\.{2,}', '。', safe_text)
        safe_text = re.sub(r'[—]+', '，', safe_text)
        safe_text = re.sub(r'[-]{2,}', '，', safe_text)
        safe_text = re.sub(r'[~～]+', '。', safe_text)
        safe_text = re.sub(r'\s+', ' ', safe_text).strip()
        
        sentences = [s.strip() for s in re.split(r'([。！？!?])', safe_text) if s.strip()]
        merged_sentences = []
        for i in range(0, len(sentences)-1, 2):
            merged_sentences.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 != 0:
            merged_sentences.append(sentences[-1])
            
        logger.info(f"🎧 收到 App 请求，切分为 {len(merged_sentences)} 句，使用音色: {feature['mode']}")
        
        all_audio_chunks = []
        
        for i, sentence in enumerate(merged_sentences):
            # 🚨 极速并发防御：在生成每一句话前，检查 App 是否已经跳段或断开！
            # 这样就能及时刹车释放 GPU，防止堵死后续的请求！
            if await request.is_disconnected():
                logger.warning(f"⚠️ App 客户端已断开，立即终止本段剩余生成，释放 GPU 资源。")
                return Response(status_code=499) # 499 Client Closed Request
                
            pure_text = re.sub(r'[。，！？；、,.!?;:\'"()\s-]', '', sentence)
            if not pure_text:
                continue
                
            # 将 CPU/GPU 计算放入线程，让异步事件循环可以检测到客户端断开
            def generate_sync():
                return voice_context.engine.generate_with_feature(sentence, feature, language="zh")
                
            logger.info(f"🎵 正在生成第 {i+1}/{len(merged_sentences)} 句...")
            audio_data = await asyncio.to_thread(generate_sync)
            
            if audio_data is not None and audio_data.size > 0:
                all_audio_chunks.append(audio_data)

        if not all_audio_chunks:
            raise HTTPException(status_code=400, detail="生成音频为空")

        # 🚨 核心视听修复：将分句数组在内存中无缝拼接！
        # 抛弃 yield，一次性转为一个带有单一 MP3 头的完整音频。
        # App 播放器会把它当成一首正常歌曲平滑播完，彻底解决只读第一句就跳过的问题！
        final_audio = np.concatenate(all_audio_chunks)
        final_audio = np.clip(final_audio, -1.0, 1.0) # 防爆音
        
        audio_segment = AudioSegment(
            (final_audio * 32767).astype(np.int16).tobytes(),
            frame_rate=24000, sample_width=2, channels=1
        )
        
        mp3_buf = io.BytesIO()
        audio_segment.export(mp3_buf, format="mp3", parameters=["-write_xing", "0", "-id3v2_version", "0"])
        
        logger.info(f"✅ 整段落音频合成完毕，发送给 App ({len(mp3_buf.getvalue())} bytes)")
        return Response(content=mp3_buf.getvalue(), media_type="audio/mpeg")
        
    except Exception as e:
        logger.error(f"❌ API 响应异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read_stream")
async def read_stream(text: str, voice: str = "aiden"):
    """流式朗读API（兼容旧接口）"""
    if not text.strip():
        return {"error": "Text cannot be empty"}
    
    if len(text) > 1000:  # 限制长度
        return {"error": "Text too long (max 1000 characters)"}
    
    # 使用当前设置的音色或指定音色
    voice_name = voice_context.current_voice if voice == "aiden" else voice
    
    logger.info(f"📖 开始流式朗读: {text[:50]}... 使用音色: {voice_name}")
    
    return StreamingResponse(
        generate_mp3_chunks(text, voice_name),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/generate_batch")
async def generate_batch(request: dict):
    """批量生成API（非流式）"""
    if not voice_context.is_ready:
        return {"error": "Service not ready"}
    
    text = request.get("text", "")
    voice_name = request.get("voice", voice_context.current_voice)
    
    if not text.strip():
        return {"error": "Text cannot be empty"}
    
    try:
        # 直接生成完整音频
        render_engine = voice_context.engine._ensure_render_engine()
        render_engine._load_mode("preset")
        results = list(render_engine.model.generate(text=text, voice=voice_name))
        
        if results:
            audio_array = results[0].audio
            mx.eval(audio_array)
            audio_data = np.array(audio_array)
            
            # 转换为字节流
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, 24000, format='WAV')
            audio_bytes = audio_buffer.getvalue()
            
            return StreamingResponse(
                io.BytesIO(audio_bytes),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
            )
        else:
            return {"error": "Failed to generate audio"}
            
    except Exception as e:
        logger.error(f"❌ 批量生成失败: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 CineCast 流式 TTS API (生产就绪版)...")
    print("📍 服务地址: http://localhost:8888")
    print("📊 API文档: http://localhost:8888/docs")
    print("🏥 健康检查: http://localhost:8888/health")
    print("🎤 音色列表: http://localhost:8888/voices")
    print("⏹️  按 Ctrl+C 停止服务")
    print("-" * 50)
    
    uvicorn.run(
        "stream_api_production:app",
        host="0.0.0.0",
        port=8888,
        reload=False,
        log_level="info"
    )