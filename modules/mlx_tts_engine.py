#!/usr/bin/env python3
"""
CineCast MLX底层渲染引擎
集成微切片与动态静音补偿，专注于极致稳定、不崩内存的音频生成
基于qwentts项目的成熟实现
"""

import gc
import io
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
        初始化MLX渲染引擎
        
        Args:
            model_path: Qwen3-TTS-MLX模型路径
        """
        logger.info("🚀 初始化MLX渲染引擎...")
        try:
            self.model = load_model(model_path)
            self.sample_rate = 22050
            self.max_chars = 60  # 微切片安全红线
            logger.info("✅ MLX渲染引擎初始化成功")
        except Exception as e:
            logger.error(f"❌ MLX渲染引擎初始化失败: {e}")
            raise
    
    def _get_dynamic_pause(self, chunk_text: str) -> int:
        """
        句级动态静音补偿
        根据标点符号自动添加适当停顿
        """
        if chunk_text.endswith(('。', '！', '？', '.', '!', '?')):
            return 600  # 句号长停顿
        elif chunk_text.endswith(('；', ';')):
            return 400  # 分号中等停顿
        elif chunk_text.endswith(('，', '、', ',', '：', ':')):
            return 250  # 逗号短停顿
        else:
            return 100  # 其他极短停顿
    
    def render_unit(self, content: str, voice_cfg: Dict) -> AudioSegment:
        """
        渲染单个剧本单元（增强版：动态语速与音高控制）
        
        Args:
            content: 待渲染的文本内容
            voice_cfg: 音色配置字典
            
        Returns:
            AudioSegment: 渲染完成的音频片段
        """
        logger.debug(f"🎵 渲染单元: {content[:50]}...")
        
        # 1. 微切片处理
        chunks = self._micro_chunk(content)
        logger.debug(f"🔪 切分为 {len(chunks)} 个片段")
        
        unit_audio = AudioSegment.empty()
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            try:
                logger.debug(f"🔄 处理片段 {i+1}/{len(chunks)}: {len(chunk)}字符")
                
                # 1. MLX 极速推理
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
                
                # 🌟 2. 电影级语速与音调控制 (Dynamic Speed & Pitch)
                speed_factor = voice_cfg.get("speed", 1.0)
                if speed_factor != 1.0:
                    # 通过改变采样率实现物理降速/加速
                    # 速度 < 1.0: 语速变慢，音高变低，适合大标题的"一字一顿"、"严肃沉稳"
                    # 速度 > 1.0: 语速变快，音高变高，适合年轻角色的欢快对白
                    new_frame_rate = int(segment.frame_rate * speed_factor)
                    segment = segment._spawn(segment.raw_data, overrides={
                        "frame_rate": new_frame_rate
                    }).set_frame_rate(self.sample_rate) # 重采样回标准频率，防止拼接报错
                
                unit_audio += segment
                
                # 🌟 3. 动态标点停顿
                pause_duration = self._get_dynamic_pause(chunk)
                # 如果配置中要求"一字一顿"(速度极慢)，我们人为增加标点停顿的长度
                if speed_factor <= 0.85:
                    pause_duration = int(pause_duration * 1.5)
                    
                unit_audio += AudioSegment.silent(duration=pause_duration)
                
                logger.debug(f"✅ 片段 {i+1} 处理完成")
                
            except Exception as e:
                logger.error(f"❌ 片段处理失败: {e}")
                # 添加错误提示音（可选）
                unit_audio += AudioSegment.silent(duration=1000)
            finally:
                # 清理内存
                if 'results' in locals():
                    del results
                if 'audio_array' in locals():
                    del audio_array
                mx.metal.clear_cache()
                gc.collect()
        
        logger.debug(f"🎵 单元渲染完成，总时长: {len(unit_audio)/1000:.2f}秒")
        return unit_audio
    
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