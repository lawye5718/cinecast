#!/usr/bin/env python3
"""
CineCast 流式实时读取 API
实现动态音色切换和实时音频流推送功能
"""

import asyncio
import io
import logging
import tempfile
import time
import os
from typing import Optional, AsyncGenerator
import mlx.core as mx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# 导入项目模块
from .mlx_tts_engine import CinecastMLXEngine as MLXTTSEngine
from .asset_manager import AssetManager
from .rhythm_manager import RhythmManager

logger = logging.getLogger(__name__)

# 创建 FastAPI 应用实例
app = FastAPI(
    title="CineCast Streaming TTS API",
    description="实时文本转语音流式API，支持动态音色切换",
    version="1.0.0"
)

# 添加 CORS 中间件支持跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态管理
class GlobalVoiceContext:
    def __init__(self):
        self.current_voice_config = {
            "role": "default",
            "feature": None,
            "voice_name": "aiden",
            "voice_name": "aiden"  # 默认音色
        }
        self.engine = None
        self.asset_manager = None
        self.rhythm_manager = None
        self.is_initialized = False
    
    async def initialize(self):
        """初始化引擎和管理器"""
        if not self.is_initialized:
            try:
                # 初始化各个组件
                self.asset_manager = AssetManager()
                self.rhythm_manager = RhythmManager()
                self.engine = MLXTTSEngine(
                    model_path="./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
                )
                self.is_initialized = True
                logger.info("✅ 流式API引擎初始化成功")
            except Exception as e:
                logger.error(f"❌ 流式API引擎初始化失败: {e}")
                raise

# OpenAI TTS 兼容请求模型
class OpenAITTSRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str = "aiden"
    response_format: str = "mp3"
    speed: float = 1.0

# 全局上下文实例
global_context = GlobalVoiceContext()

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    await global_context.initialize()

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "CineCast Streaming TTS API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "initialized": global_context.is_initialized,
        "current_voice": global_context.current_voice_config["role"]
    }

@app.get("/voices")
async def list_available_voices():
    """获取可用音色列表"""
    if not global_context.is_initialized:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    # 返回预设音色列表
    preset_voices = [
        "aiden", "dylan", "ono_anna", "ryan", 
        "sohee", "uncle_fu", "vivian", "eric", "serena"
    ]
    
    # 获取克隆音色
    clone_voices = list(global_context.asset_manager.clone_voice_features.keys()) if global_context.asset_manager else []
    
    return {
        "preset_voices": preset_voices,
        "clone_voices": clone_voices,
        "current_voice": global_context.current_voice_config["role"]
    }

@app.post("/set_voice")
async def set_voice(
    voice_name: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    设置当前音色
    - voice_name: 音色库中的预设音色名
    - file: 可选的上传音频文件用于音色克隆
    """
    if not global_context.is_initialized:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        if file:
            # 处理上传的音色克隆
            logger.info(f"🎤 开始处理上传音色克隆: {file.filename}")
            
            # 读取上传的音频文件
            audio_bytes = await file.read()
            
            # 保存到临时文件进行处理
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            
            try:
                # 加载音频并重采样到24kHz
                audio_segment = AudioSegment.from_file(temp_path)
                audio_segment = audio_segment.set_frame_rate(24000).set_channels(1)
                
                # 转换为numpy数组
                samples = np.array(audio_segment.get_array_of_samples())
                if audio_segment.sample_width == 2:
                    samples = samples.astype(np.float32) / 32768.0
                elif audio_segment.sample_width == 4:
                    samples = samples.astype(np.float32) / 2147483648.0
                
                # 提取音色特征
                feature = global_context.engine.extract_voice_feature(samples)
                
                # 保存克隆音色
                clone_name = f"clone_{int(time.time())}"
                global_context.asset_manager.save_clone_voice(clone_name, feature)
                
                # 更新当前音色配置
                global_context.current_voice_config.update({
                    "role": "uploaded_clone",
                    "feature": feature,
                    "voice_name": clone_name
                })
                
                logger.info(f"✅ 音色克隆成功: {clone_name}")
                
            finally:
                # 清理临时文件
                os.unlink(temp_path)
                
        else:
            # 使用预设音色
            if voice_name.lower() not in ["aiden", "dylan", "ono_anna", "ryan", 
                                        "sohee", "uncle_fu", "vivian", "eric", "serena"]:
                raise HTTPException(status_code=400, detail=f"不支持的音色: {voice_name}")
            
            # 加载预设音色特征
            feature = global_context.asset_manager.load_role(voice_name.lower())
            global_context.current_voice_config.update({
                "role": "preset",
                "feature": feature,
                "voice_name": voice_name.lower()
            })
            logger.info(f"✅ 切换到预设音色: {voice_name}")
        
        return {
            "status": "success",
            "active_role": global_context.current_voice_config["role"],
            "voice_name": global_context.current_voice_config["voice_name"]
        }
        
    except Exception as e:
        logger.error(f"❌ 设置音色失败: {e}")
        raise HTTPException(status_code=500, detail=f"音色设置失败: {str(e)}")

    def get_active_feature(self, voice_id: str):
        """获取活跃音色特征（实时生效核心）"""
        # 优先级 1: 检查是否是刚上传的临时音色
        if voice_id == "uploaded_clone" and self.current_voice_config["feature"] is not None:
            return self.current_voice_config["feature"]
        
        # 优先级 2: 检查本地持久化音色库
        try:
            return self.asset_manager.load_role(voice_id)
        except:
            # 优先级 3: 最终回退到 aiden 预设
            return self.asset_manager.load_role("aiden")

async def mp3_stream_generator(text: str, voice_id: str = "aiden") -> AsyncGenerator[bytes, None]:
    """
    MP3流式生成器：解决WAV头部冗余问题
    """
    if not global_context.is_initialized:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        # 获取活跃音色特征
        feature = global_context.get_active_feature(voice_id)
        
        # 按句子分割文本
        segments = global_context.rhythm_manager.process_text_with_metadata(text)
        sentences = [seg['text'] for seg in segments if seg['text'].strip()]
        logger.info(f"📝 开始MP3流式生成，共 {len(sentences)} 个句子")
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            logger.debug(f"🎵 正在生成第 {i+1}/{len(sentences)} 句: {sentence[:30]}...")
            
            # 生成原始PCM数据
            wav_data = global_context.engine.generate_with_feature(
                sentence.strip(),
                feature,
                language="zh"
            )
            
            # 将PCM转换为MP3帧（解决WAV头部冗余问题）
            audio_segment = AudioSegment(
                (wav_data * 32767).astype(np.int16).tobytes(),
                frame_rate=24000, sample_width=2, channels=1
            )
            
            # 导出为MP3字节，不带ID3标签以减少开销
            mp3_buf = io.BytesIO()
            audio_segment.export(mp3_buf, format="mp3", parameters=["-write_xing", "0"])
            yield mp3_buf.getvalue()
            
            # Mac mini显存自愈
            mx.metal.clear_cache()
            
            logger.debug(f"✅ 第 {i+1} 句MP3推送完成")
            
    except Exception as e:
        logger.error(f"❌ 流式生成过程中出错: {e}")
        raise

@app.get("/read_stream")
async def read_stream(text: str, lang: str = "zh"):
    """
    实时读书API访问入口
    返回音频流，支持边生成边播放
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="文本内容不能为空")
    
    if len(text) > 5000:  # 限制文本长度
        raise HTTPException(status_code=400, detail="文本长度超过限制（5000字符）")
    
    logger.info(f"📖 开始流式朗读: {text[:50]}...")
    
    return StreamingResponse(
        mp3_stream_generator(text),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/v1/audio/speech")
async def openai_compatible_tts(request: OpenAITTSRequest):
    """
    符合OpenAI标准的流式TTS接口
    支持实时动态音色选择
    """
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")
    
    return StreamingResponse(
        mp3_stream_generator(request.input, request.voice), 
        media_type="audio/mpeg"
    )

@app.post("/batch_generate")
async def batch_generate(request: dict):
    """
    批量生成API（非流式）
    适用于需要完整音频文件的场景
    """
    if not global_context.is_initialized:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    text = request.get("text", "")
    voice_name = request.get("voice_name", "aiden")
    language = request.get("language", "zh")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="文本内容不能为空")
    
    try:
        # 设置音色
        feature = global_context.asset_manager.load_role(voice_name.lower())
        
        # 生成完整音频
        full_audio = global_context.engine.generate_with_feature(
            text.strip(),
            feature,
            language=language
        )
        
        # 转换为字节流
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, full_audio, 24000, format='WAV')
        audio_bytes = audio_buffer.getvalue()
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=tts_output.wav"}
        )
        
    except Exception as e:
        logger.error(f"❌ 批量生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

# 错误处理中间件
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"🚨 API异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    # 开发模式运行
    uvicorn.run(
        "stream_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )