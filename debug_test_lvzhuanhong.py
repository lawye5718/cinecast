#!/usr/bin/env python3
"""
吕转红受贿案EPUB深度调试测试脚本
详细记录terminal信息，排查音频文件为空的问题
"""

import os
import sys
import json
import logging
from pathlib import Path

# 设置详细的日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_test.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def detailed_test_lvzhuanhong():
    """详细的吕转红测试，包含完整的调试信息"""
    
    print("=" * 80)
    print("🔍 吕转红受贿案EPUB深度调试测试")
    print("=" * 80)
    
    # 1. 环境检查
    print("\n🔧 第一步：环境检查")
    print("-" * 40)
    
    # 检查必要的依赖
    try:
        import mlx.core as mx
        print(f"✅ MLX版本: {mx.__version__ if hasattr(mx, '__version__') else '未知'}")
    except ImportError as e:
        print(f"❌ MLX导入失败: {e}")
        return False
    
    try:
        import soundfile as sf
        print(f"✅ SoundFile版本: {sf.__version__}")
    except ImportError as e:
        print(f"❌ SoundFile导入失败: {e}")
        return False
    
    # 检查模型路径
    model_path = "../qwentts/models/Qwen3-TTS-MLX-0.6B"
    if os.path.exists(model_path):
        print(f"✅ 模型路径存在: {model_path}")
    else:
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 2. 组件初始化
    print("\n🚀 第二步：组件初始化")
    print("-" * 40)
    
    try:
        from alexandria.local_llm_client import LocalLLMClient
        from alexandria.local_tts_engine import LocalTTSEngine
        
        # 加载配置
        config = {
            "llm": {
                "provider": "ollama",
                "model": "qwen14b-pro",
                "host": "http://localhost:11434",
                "api_url": "http://localhost:11434/api/chat",
                "temperature": 0.0,
                "num_ctx": 8192
            },
            "tts": {
                "mode": "local",
                "model_path": model_path,
                "device": "metal",
                "compile_codec": False,
                "language": "Chinese"
            }
        }
        
        print("📝 初始化本地LLM客户端...")
        llm_client = LocalLLMClient(config)
        
        print("📝 初始化本地TTS引擎...")
        tts_engine = LocalTTSEngine(config)
        
        # 健康检查
        print("\n🏥 健康检查结果:")
        ollama_ok = llm_client._check_connection()
        tts_ok = tts_engine.is_available()
        
        print(f"  Ollama连接: {'✅ 正常' if ollama_ok else '❌ 异常'}")
        print(f"  TTS引擎: {'✅ 可用' if tts_ok else '❌ 不可用'}")
        
        if not (ollama_ok and tts_ok):
            print("❌ 系统组件初始化失败")
            return False
            
    except Exception as e:
        print(f"❌ 组件初始化失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    # 3. 文件处理
    print("\n📄 第三步：EPUB文件处理")
    print("-" * 40)
    
    epub_path = "./yuan, sophocles/吕转红受贿罪二审刑事裁定书/吕转红受贿罪二审刑事裁定书 - sophocles yuan.epub"
    
    if not os.path.exists(epub_path):
        print(f"❌ EPUB文件不存在: {epub_path}")
        return False
    
    print(f"📍 处理文件: {epub_path}")
    print(f"📊 文件大小: {os.path.getsize(epub_path) / 1024 / 1024:.2f} MB")
    
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        
        print("📖 解析EPUB文件...")
        book = epub.read_epub(epub_path)
        
        # 提取文本内容
        all_text = []
        item_count = 0
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            item_count += 1
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()
            clean_text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
            if len(clean_text) > 50:  # 降低过滤阈值
                all_text.append(clean_text)
        
        print(f"📊 解析统计: 共处理 {item_count} 个项目，提取 {len(all_text)} 个有效文本块")
        
        if not all_text:
            print("❌ 未能提取到有效文本内容")
            return False
            
        full_text = '\n\n'.join(all_text)
        print(f"✅ 成功提取文本，总字符数: {len(full_text)}")
        
        # 使用非常短的测试文本进行调试
        test_text = "测试音频生成功能。"
        print(f"📝 使用调试文本进行测试: '{test_text}' (长度: {len(test_text)})")
        
    except Exception as e:
        print(f"❌ EPUB处理失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    # 4. 剧本生成测试
    print("\n🎭 第四步：剧本生成测试")
    print("-" * 40)
    
    try:
        print("🧠 调用本地Qwen14B-Pro生成剧本...")
        script = llm_client.generate_script(test_text)
        
        if not script:
            print("❌ 剧本生成返回空结果")
            return False
            
        print(f"✅ 剧本生成成功，共 {len(script)} 个片段")
        
        for i, item in enumerate(script):
            print(f"  片段 {i+1}: [{item['type']}] {item['speaker']}: {item['content'][:50]}...")
            
    except Exception as e:
        print(f"❌ 剧本生成失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    # 5. 音频渲染深度测试
    print("\n🎵 第五步：音频渲染深度测试")
    print("-" * 40)
    
    output_dir = "./debug_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    render_success = 0
    
    for i, item in enumerate(script[:1]):  # 只测试第一个片段
        print(f"\n🔊 渲染片段 {i+1}:")
        print(f"   类型: {item['type']}")
        print(f"   说话者: {item['speaker']}")
        print(f"   内容: {item['content']}")
        print(f"   情感: {item.get('emotion', '未知')}")
        
        wav_path = os.path.join(output_dir, f"debug_fragment_{i:03d}_{item['type']}.wav")
        voice_config = {
            "speaker": item["speaker"],
            "gender": item["gender"]
        }
        
        print(f"   输出路径: {wav_path}")
        print(f"   音色配置: {voice_config}")
        
        try:
            # 调用渲染方法
            success = tts_engine.render_dry_chunk(
                item["content"], 
                voice_config, 
                wav_path, 
                item.get("emotion", "平静")
            )
            
            if success and os.path.exists(wav_path):
                file_size = os.path.getsize(wav_path)
                print(f"   ✅ 渲染成功，文件大小: {file_size} bytes")
                
                # 检查音频文件内容
                if file_size > 44:  # WAV文件头至少44字节
                    import soundfile as sf
                    try:
                        audio_data, sample_rate = sf.read(wav_path)
                        print(f"   📊 音频信息: 采样率={sample_rate}Hz, 长度={len(audio_data)}样本, 持续时间={len(audio_data)/sample_rate:.2f}秒")
                        print(f"   📊 数据范围: min={audio_data.min():.6f}, max={audio_data.max():.6f}, mean={audio_data.mean():.6f}")
                        
                        # 判断是否为静音
                        if audio_data.max() == 0.0 and audio_data.min() == 0.0:
                            print("   ⚠️ 检测到静音文件（全零数据）")
                        elif abs(audio_data.max() - audio_data.min()) < 0.001:
                            print("   ⚠️ 检测到几乎静音的文件（动态范围极小）")
                        else:
                            print("   ✅ 检测到有效音频信号")
                            render_success += 1
                            
                    except Exception as sf_error:
                        print(f"   ❌ 音频文件读取失败: {sf_error}")
                else:
                    print("   ❌ 文件过小，可能是空文件")
            else:
                print("   ❌ 渲染失败或文件未生成")
                
        except Exception as render_error:
            print(f"   ❌ 渲染过程异常: {render_error}")
            import traceback
            print(f"   详细错误: {traceback.format_exc()}")
    
    # 6. 测试总结
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    
    print(f"📊 关键指标:")
    print(f"  - 环境检查: {'通过' if (ollama_ok and tts_ok) else '失败'}")
    print(f"  - 文件处理: {'通过' if len(full_text) > 0 else '失败'}")
    print(f"  - 剧本生成: {'通过' if len(script) > 0 else '失败'}")
    print(f"  - 音频渲染: {render_success}/1 成功")
    
    if render_success > 0:
        print("\n🎉 测试成功！音频文件包含有效内容。")
        return True
    else:
        print("\n❌ 测试失败！音频文件为空或无效。")
        return False

def main():
    """主函数"""
    print("开始吕转红受贿案EPUB深度调试测试...")
    
    try:
        success = detailed_test_lvzhuanhong()
        
        if success:
            print("\n" + "=" * 80)
            print("✅ 深度调试测试通过！")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ 深度调试测试失败！")
            print("=" * 80)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生未预期错误: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()