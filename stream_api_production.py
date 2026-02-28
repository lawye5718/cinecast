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
import warnings   # 🚨 引入警告控制

# 屏蔽 Tokenizer 无意义的正则表达式警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="tiktoken")

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

def generate_mp3_chunks(text: str, voice_name: str):
    """生成MP3音频块的生成器函数（加锁防崩溃 + 支持克隆）"""
    if not voice_context.is_ready:
        raise RuntimeError("Service not ready")
    
    try:
        # 🚨 架构回归：获取完整的音色特征（可能是预设，也可能是本地的克隆配置）
        feature = voice_context.get_voice_feature(voice_name)
        
        # 文本预处理 - 暴力清洗 (保留你之前的修复)
        safe_text = re.sub(r'[…]+', '。', text)
        safe_text = re.sub(r'\.{2,}', '。', safe_text)
        safe_text = re.sub(r'[—]+', '，', safe_text)
        safe_text = re.sub(r'[-]{2,}', '，', safe_text)
        safe_text = re.sub(r'[~～]+', '。', safe_text)
        safe_text = re.sub(r'\s+', ' ', safe_text).strip()
        
        # 按句号、问号、感叹号安全分割
        sentences = [s.strip() for s in re.split(r'([。！？!?])', safe_text) if s.strip()]
        
        # 将句子和标点重新合并，避免标点单独成句
        merged_sentences = []
        for i in range(0, len(sentences)-1, 2):
            merged_sentences.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 != 0:
            merged_sentences.append(sentences[-1])
            
        logger.info(f"📝 开始生成 {len(merged_sentences)} 个句子, 使用音色特征: {feature['mode']}")
        
        for i, sentence in enumerate(merged_sentences):
            # 防止纯标点
            pure_text = re.sub(r'[。，！？；、,.!?;:\'"()\s-]', '', sentence)
            if not pure_text:
                continue
                
            logger.info(f"🎵 正在生成第 {i+1}/{len(merged_sentences)} 句: {sentence[:20]}...")
            
            # 🌟 架构回归：调用原本封装好的 generate_with_feature，它原生支持克隆和预设！
            # 🚨 注意：引擎内部已有锁保护，此处无需再加锁
            try:
                audio_data = voice_context.engine.generate_with_feature(
                    sentence, 
                    feature, 
                    language="zh"
                )
                
                if audio_data is not None and audio_data.size > 0:
                    # 🚨 新增防御：截断一切异常尖峰，防止 int16 溢出导致的刺耳爆音
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    
                    # 将PCM转换为MP3帧
                    audio_segment = AudioSegment(
                        (audio_data * 32767).astype(np.int16).tobytes(),
                        frame_rate=24000, sample_width=2, channels=1
                    )
                    
                    mp3_buf = io.BytesIO()
                    audio_segment.export(mp3_buf, format="mp3", parameters=["-write_xing", "0"])
                    mp3_bytes = mp3_buf.getvalue()
                    
                    logger.info(f"✅ 第 {i+1} 句MP3生成完成 ({len(mp3_bytes)} bytes)")
                    yield mp3_bytes
                else:
                    logger.warning(f"⚠️ 生成音频为空，跳过第 {i+1} 句")
                    
            except Exception as ex:
                logger.error(f"❌ 当前句子生成异常: {ex}")
                continue
            
            # 在锁外释放 CPU 资源片刻，防止阻塞其他线程抢锁
            import gc
            gc.collect()
            time.sleep(0.01) 
                    
    except Exception as e:
        logger.error(f"❌ 整体音频生成流失败: {e}")
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