#!/usr/bin/env python3
"""
CineCast 流式API - 最终成功版本
基于已验证的工作方法实现
"""

import sys
import os
sys.path.insert(0, '/Users/yuanliang/superstar/superstar3.1/projects/cinecast')

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import mlx.core as mx
import numpy as np
import soundfile as sf
import io
import logging
import time

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

def generate_audio_chunks(text: str, voice_name: str):
    """生成音频块的生成器函数"""
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
            
            # 创建临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            
            try:
                # 直接调用模型生成
                render_engine._load_mode("preset")
                results = list(render_engine.model.generate(text=sentence, voice=voice_name))
                
                if results:
                    # 处理音频数据
                    audio_array = results[0].audio
                    mx.eval(audio_array)
                    audio_data = np.array(audio_array)
                    
                    # 直接写入文件
                    sf.write(tmp_path, audio_data, 24000, format='WAV')
                    
                    # 读取并返回音频数据
                    with open(tmp_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    logger.info(f"✅ 第 {i+1} 句生成完成 ({len(audio_bytes)} bytes)")
                    yield audio_bytes
                    
                    # 清理显存
                    mx.metal.clear_cache()
                    
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    except Exception as e:
        logger.error(f"❌ 音频生成失败: {e}")
        raise

@app.get("/read_stream")
async def read_stream(text: str, voice: str = "aiden"):
    """流式朗读API"""
    if not text.strip():
        return {"error": "Text cannot be empty"}
    
    if len(text) > 1000:  # 限制长度
        return {"error": "Text too long (max 1000 characters)"}
    
    # 使用当前设置的音色或指定音色
    voice_name = voice_context.current_voice if voice == "aiden" else voice
    
    logger.info(f"📖 开始流式朗读: {text[:50]}... 使用音色: {voice_name}")
    
    return StreamingResponse(
        generate_audio_chunks(text, voice_name),
        media_type="audio/wav",
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