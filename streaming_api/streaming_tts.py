#!/usr/bin/env python3
"""
CineCast 流式 TTS API (Streaming TTS API)
实现实时音频流生成，支持网页端动态切换音色。
"""

import asyncio
import io
import logging
from typing import AsyncGenerator, Optional

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modules.mlx_tts_engine import MLXRenderEngine
from modules.asset_manager import AssetManager
from modules.rhythm_manager import RhythmManager
from modules.role_manager import RoleManager

logger = logging.getLogger(__name__)

# 全局状态管理
class GlobalVoiceState:
    def __init__(self):
        self.current_voice_config = {
            "role": "default",
            "feature": None,
            "engine": None
        }
        self.asset_manager = AssetManager()
        self.rhythm_manager = RhythmManager()
    
    async def initialize_engine(self):
        """初始化 TTS 引擎"""
        if self.current_voice_config["engine"] is None:
            self.current_voice_config["engine"] = MLXRenderEngine()
            logger.info("🚀 TTS 引擎已初始化")
    
    async def set_voice_by_role(self, role_name: str):
        """通过音色库设置音色"""
        try:
            feature = RoleManager.load_voice_feature(role_name, "./voices")
            self.current_voice_config["feature"] = feature
            self.current_voice_config["role"] = role_name
            logger.info(f"🔊 音色已设置为: {role_name}")
            return {"status": "success", "role": role_name}
        except Exception as e:
            logger.error(f"❌ 设置音色失败: {e}")
            raise HTTPException(status_code=400, detail=f"音色设置失败: {str(e)}")
    
    async def set_voice_by_upload(self, audio_bytes: bytes):
        """通过上传音频设置克隆音色"""
        try:
            # TODO: 实现音频特征提取逻辑
            # 这里需要调用 MLX 引擎的特征提取功能
            feature = self._extract_feature_from_bytes(audio_bytes)
            self.current_voice_config["feature"] = feature
            self.current_voice_config["role"] = "uploaded_clone"
            logger.info("🔊 克隆音色已设置")
            return {"status": "success", "role": "uploaded_clone"}
        except Exception as e:
            logger.error(f"❌ 克隆音色设置失败: {e}")
            raise HTTPException(status_code=400, detail=f"音色克隆失败: {str(e)}")
    
    def _extract_feature_from_bytes(self, audio_bytes: bytes):
        """从音频字节中提取特征"""
        if self.current_voice_config["engine"] is None:
            raise RuntimeError("TTS 引擎尚未初始化")
            
        import tempfile
        from pydub import AudioSegment
        
        # 将上传的字节流转为 24kHz 的 numpy 数组
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            audio_segment = AudioSegment.from_file(tmp_path)
            audio_segment = audio_segment.set_frame_rate(24000).set_channels(1)
            samples = np.array(audio_segment.get_array_of_samples())
            
            # 归一化处理
            if audio_segment.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:
                samples = samples.astype(np.float32) / 2147483648.0
                
            # 调用 MLX 引擎的提取逻辑
            return self.current_voice_config["engine"].extract_voice_feature(samples)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    async def stream_tts(self, text: str, language: str = "zh") -> AsyncGenerator[bytes, None]:
        """流式 TTS 生成"""
        if self.current_voice_config["engine"] is None:
            await self.initialize_engine()
        
        engine = self.current_voice_config["engine"]
        feature = self.current_voice_config["feature"]
        
        # 按句子分割文本
        sentences = [s["text"] for s in self.rhythm_manager.process_text_with_metadata(text)]
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            try:
                # 使用当前音色配置进行推理
                if feature is not None:
                    # 克隆模式
                    audio_array, sample_rate = engine.generate_voice_clone(sentence, feature)
                else:
                    # 默认模式
                    audio_array, sample_rate = engine._run_base(sentence)
                
                # 转换为 MP3 字节流（解决WAV头部冗余问题）
                mp3_bytes = self._numpy_to_mp3_bytes(audio_array, sample_rate)
                yield mp3_bytes
                
                # 显式清理 Metal 缓存（针对 Mac mini 内存优化）
                mx.metal.clear_cache()
                
            except Exception as e:
                logger.error(f"❌ TTS 生成失败: {e}")
                continue

    def _numpy_to_mp3_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """将 numpy 数组转换为 MP3 字节流（解决WAV头部冗余问题）"""
        try:
            from pydub import AudioSegment
            
            # 确保是 16-bit PCM 格式
            if audio_array.dtype != np.int16:
                audio_array = (audio_array * 32767).astype(np.int16)
            
            # 使用 pydub 转换为 MP3，避免WAV头部重复问题
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit
                channels=1       # mono
            )
            
            # 导出为 MP3 字节流，不带 ID3 标签以减少开销
            mp3_buffer = io.BytesIO()
            audio_segment.export(
                mp3_buffer,
                format="mp3",
                parameters=["-write_xing", "0"]  # 禁用 Xing header 减少头部信息
            )
            return mp3_buffer.getvalue()
            
        except ImportError:
            logger.error("pydub 未安装，无法生成 MP3 流")
            raise
        except Exception as e:
            logger.error(f"音频格式转换失败: {e}")
            raise

# FastAPI 应用实例
app = FastAPI(title="CineCast Streaming TTS API", version="1.0.0")

# 全局状态实例
voice_state = GlobalVoiceState()

# 请求模型
class TTSRequest(BaseModel):
    text: str
    language: str = "zh"

# API 路由
@app.post("/set_voice/role")
async def set_voice_role(role_name: str = Form(...)):
    """设置音色库中的音色"""
    return await voice_state.set_voice_by_role(role_name)

@app.post("/set_voice/upload")
async def set_voice_upload(file: UploadFile = File(...)):
    """上传音频文件设置克隆音色"""
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="请上传音频文件")
    
    audio_bytes = await file.read()
    return await voice_state.set_voice_by_upload(audio_bytes)

@app.post("/tts/stream")
async def stream_tts(request: TTSRequest):
    """流式 TTS 生成接口"""
    return StreamingResponse(
        voice_state.stream_tts(request.text, request.language),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "CineCast Streaming TTS"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
