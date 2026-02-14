#!/usr/bin/env python3
"""
CineCast MLX底层渲染引擎
阶段二：纯净干音渲染 (Dry Voice Rendering)
只负责将文本变成 WAV 文件，绝不维护状态
基于qwentts项目的成熟实现
"""

import gc
import io
import os
import re
import numpy as np
import soundfile as sf
import mlx.core as mx
from mlx_audio.tts.utils import load_model
from pydub import AudioSegment
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MLXRenderEngine:
    def __init__(self, model_path="./models/Qwen3-TTS-MLX-0.6B"):
        """
        初始化MLX纯净干音渲染引擎
        
        Args:
            model_path: Qwen3-TTS-MLX模型路径
        """
        logger.info("🚀 启动 MLX 纯净干音渲染引擎...")
        try:
            self.model = load_model(model_path)
            self.sample_rate = 22050
            self.max_chars = 60  # 微切片安全长度上限
            logger.info("✅ MLX渲染引擎初始化成功")
        except Exception as e:
            logger.error(f"❌ MLX渲染引擎初始化失败: {e}")
            raise
    
    def render_dry_chunk(self, content: str, voice_cfg: dict, save_path: str) -> bool:
        """
        只负责将文本变成 WAV 文件，绝不维护状态
        🌟 断点续传核心：已存在则直接跳过！
        """
        if os.path.exists(save_path):
            logger.debug(f"⏭️  文件已存在，跳过渲染: {save_path}")
            return True # 🌟 断点续传核心：已存在则直接跳过！
            
        try:
            logger.debug(f"🎵 渲染干音: {content[:50]}... -> {save_path}")
            
            # MLX 极速推理
            results = list(self.model.generate(
                text=content,
                ref_audio=voice_cfg["audio"],
                ref_text=voice_cfg["text"]
            ))
            
            audio_array = results[0].audio
            mx.eval(audio_array) # 强制执行
            audio_data = np.array(audio_array)
            
            # 直接写入磁盘，绝不在内存中积压
            sf.write(save_path, audio_data, self.sample_rate, format='WAV')
            logger.debug(f"✅ 干音渲染完成: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 干音渲染失败 [{content[:10]}...]: {e}")
            return False
            
        finally:
            # 清理内存
            if 'results' in locals(): del results
            if 'audio_array' in locals(): del audio_array
            if 'audio_data' in locals(): del audio_data
            mx.metal.clear_cache()
            gc.collect()
    
    def render_unit(self, content: str, voice_cfg: Dict) -> AudioSegment:
        """
        渲染单个剧本单元为AudioSegment（兼容旧接口）

        与 render_dry_chunk 的区别:
        - render_dry_chunk: 三段式架构推荐方法，直接写入磁盘WAV文件，
          支持断点续传，内存占用更低
        - render_unit: 旧接口，返回内存中的AudioSegment对象，
          适用于需要直接操作音频数据的场景

        Args:
            content: 要渲染的文本内容
            voice_cfg: 音色配置字典，包含 audio, text, speed 等字段

        Returns:
            AudioSegment: 渲染后的音频片段，失败时返回空AudioSegment
        """
        logger.warning("⚠️  使用旧接口render_unit，建议迁移到render_dry_chunk")
        try:
            chunks = self._micro_chunk(content)
            unit_audio = AudioSegment.empty()

            for chunk in chunks:
                if not chunk.strip():
                    continue

                results = list(self.model.generate(
                    text=chunk,
                    ref_audio=voice_cfg["audio"],
                    ref_text=voice_cfg["text"]
                ))

                audio_array = results[0].audio
                mx.eval(audio_array)
                audio_data = np.array(audio_array)

                buffer = io.BytesIO()
                sf.write(buffer, audio_data, self.sample_rate, format='WAV')
                buffer.seek(0)
                segment = AudioSegment.from_file(buffer, format="wav")

                speed_factor = voice_cfg.get("speed", 1.0)
                if speed_factor != 1.0:
                    new_frame_rate = int(segment.frame_rate * speed_factor)
                    segment = segment._spawn(segment.raw_data, overrides={
                        "frame_rate": new_frame_rate
                    }).set_frame_rate(self.sample_rate)

                unit_audio += segment

                del results, audio_array, audio_data
                mx.metal.clear_cache()
                gc.collect()

            return unit_audio
        except Exception as e:
            logger.error(f"❌ render_unit失败: {e}")
            return AudioSegment.empty()
    
    def _micro_chunk(self, text: str) -> list:
        """
        多级微切片算法
        确保每个片段都不超过安全长度限制
        """
        if not text.strip():
            return []
        
        # 第一级：按句号/换行符粗切分
        raw_sentences = re.split(r'([。！？；\n])', text)
        sub_chunks = []
        
        # 拼接标点与句子
        temp_sentence = ""
        for part in raw_sentences:
            if not part.strip():
                continue
            if re.match(r'^[。！？；\n]$', part.strip()):
                temp_sentence += part
                sub_chunks.append(temp_sentence)
                temp_sentence = ""
            else:
                if temp_sentence:
                    sub_chunks.append(temp_sentence)
                temp_sentence = part
        if temp_sentence:
            sub_chunks.append(temp_sentence)
        
        # 第二级：强制细分超长句
        fine_chunks = []
        for sentence in sub_chunks:
            if len(sentence) > self.max_chars:
                # 按逗号或顿号进一步肢解
                comma_parts = re.split(r'([，、：])', sentence)
                temp_comma = ""
                for cp in comma_parts:
                    if re.match(r'^[，、：]$', cp.strip()):
                        temp_comma += cp
                        fine_chunks.append(temp_comma)
                        temp_comma = ""
                    else:
                        temp_comma += cp
                        if len(temp_comma) >= self.max_chars:
                            fine_chunks.append(temp_comma)
                            temp_comma = ""
                if temp_comma:
                    fine_chunks.append(temp_comma)
            else:
                fine_chunks.append(sentence)
        
        # 第三级：智能回填合并过短片段
        final_chunks = []
        current_chunk = ""
        for fc in fine_chunks:
            fc = fc.strip()
            if not fc:
                continue
            if len(current_chunk) + len(fc) <= self.max_chars:
                current_chunk += " " + fc if current_chunk else fc
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                current_chunk = fc
        if current_chunk:
            final_chunks.append(current_chunk.strip())
        
        return [chunk for chunk in final_chunks if chunk.strip()]

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 注意：这里需要确保模型路径正确
    try:
        engine = MLXRenderEngine()
        
        # 测试音色配置
        test_voice_cfg = {
            "audio": "reference_for_production.wav",
            "text": "测试参考文本",
            "speed": 1.0
        }
        
        # 测试渲染
        test_content = "这是一个测试文本，用来验证MLX渲染引擎是否正常工作。"
        audio_result = engine.render_unit(test_content, test_voice_cfg)
        
        print(f"✅ 渲染成功，音频时长: {len(audio_result)/1000:.2f}秒")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")