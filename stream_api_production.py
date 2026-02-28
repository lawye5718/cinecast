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
from pydub import AudioSegment

# 导入项目模块
from modules.mlx_tts_engine import CinecastMLXEngine as MLXTTSEngine
from modules.asset_manager import AssetManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(title="CineCast Streaming TTS API - Production Ready")

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

def generate_mp3_chunks(text: str, voice_name: str):
    """生成MP3音频块的生成器函数（解决WAV头部冗余问题）"""
    if not voice_context.is_ready:
        raise RuntimeError("Service not ready")
    
    try:
        # 直接使用已验证的工作方法
        render_engine = voice_context.engine._ensure_render_engine()
        
        # 准备voice配置
        voice_cfg = {
            "mode": "preset",
            "voice": voice_name
        }
        
        # 文本预处理 - 简单按句号分割
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if not sentences[-1].endswith(('。', '.', '!', '?', '！', '？')):
            sentences[-1] += '。'
        
        logger.info(f"📝 开始生成 {len(sentences)} 个句子")
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            logger.info(f"🎵 正在生成第 {i+1}/{len(sentences)} 句: {sentence[:20]}...")
            
            # 直接调用模型生成
            render_engine._load_mode("preset")
            results = list(render_engine.model.generate(text=sentence, voice=voice_name))
            
            if results:
                # 处理音频数据
                audio_array = results[0].audio
                mx.eval(audio_array)
                audio_data = np.array(audio_array)
                
                # 将PCM转换为MP3帧（解决WAV头部冗余问题）
                audio_segment = AudioSegment(
                    (audio_data * 32767).astype(np.int16).tobytes(),
                    frame_rate=24000, sample_width=2, channels=1
                )
                
                # 导出为MP3字节，不带ID3标签以减少开销
                mp3_buf = io.BytesIO()
                audio_segment.export(mp3_buf, format="mp3", parameters=["-write_xing", "0"])
                mp3_bytes = mp3_buf.getvalue()
                
                logger.info(f"✅ 第 {i+1} 句MP3生成完成 ({len(mp3_bytes)} bytes)")
                yield mp3_bytes
                
                # 清理显存
                mx.metal.clear_cache()
                    
    except Exception as e:
        logger.error(f"❌ 音频生成失败: {e}")
        raise

@app.post("/v1/audio/speech")
async def openai_compatible_tts(request: OpenAITTSRequest):
    """符合OpenAI标准的流式TTS接口"""
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")
    
    logger.info(f"🎧 OpenAI兼容TTS请求: {request.input[:50]}... 使用音色: {request.voice}")
    
    return StreamingResponse(
        generate_mp3_chunks(request.input, request.voice),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

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
    print("📍 服务地址: http://localhost:8000")
    print("📊 API文档: http://localhost:8000/docs")
    print("🏥 健康检查: http://localhost:8000/health")
    print("🎤 音色列表: http://localhost:8000/voices")
    print("⏹️  按 Ctrl+C 停止服务")
    print("-" * 50)
    
    uvicorn.run(
        "stream_api_production:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )