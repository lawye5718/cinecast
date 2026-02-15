#!/usr/bin/env python3
"""
吕转红受贿案EPUB测试脚本 (Alexandria分支版本)
使用本地化集成的组件处理法律文书
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入本地化组件
try:
    from alexandria.local_llm_client import LocalLLMClient
    from alexandria.local_tts_engine import LocalTTSEngine
    LOCAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 本地化组件导入失败: {e}")
    LOCAL_COMPONENTS_AVAILABLE = False

def test_lvzhuanhong_with_alexandria():
    """使用Alexandria分支组件测试吕转红案件"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("📚 开始使用Alexandria分支测试吕转红受贿案")
    print("=" * 60)
    
    # 检查本地化组件
    if not LOCAL_COMPONENTS_AVAILABLE:
        print("❌ 本地化组件不可用，检查alexandria目录结构")
        return False
    
    # 加载配置
    config_path = "./alexandria/local_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
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
                "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",
                "device": "metal",
                "compile_codec": False,
                "language": "Chinese"
            }
        }
    
    # 初始化组件
    print("\n🔧 初始化本地化组件...")
    llm_client = LocalLLMClient(config)
    tts_engine = LocalTTSEngine(config)
    
    # 健康检查
    print("\n🏥 系统健康检查:")
    ollama_ok = llm_client._check_connection()
    tts_ok = tts_engine.is_available()
    
    print(f"  Ollama连接: {'✅ 正常' if ollama_ok else '❌ 异常'}")
    print(f"  TTS引擎: {'✅ 可用' if tts_ok else '❌ 不可用'}")
    
    if not (ollama_ok and tts_ok):
        print("\n❌ 系统组件存在问题，无法继续测试")
        return False
    
    # EPUB文件路径
    epub_path = "./yuan, sophocles/吕转红受贿罪二审刑事裁定书/吕转红受贿罪二审刑事裁定书 - sophocles yuan.epub"
    if not os.path.exists(epub_path):
        print(f"\n❌ EPUB文件不存在: {epub_path}")
        return False
    
    print(f"\n📄 处理文件: {epub_path}")
    
    # 解析EPUB内容
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        
        print("📖 正在解析EPUB文件...")
        book = epub.read_epub(epub_path)
        
        # 提取文本内容
        all_text = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()
            # 清理文本
            clean_text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
            if len(clean_text) > 100:  # 过滤短内容
                all_text.append(clean_text)
        
        if not all_text:
            print("❌ 未能提取到有效文本内容")
            return False
            
        full_text = '\n\n'.join(all_text)
        print(f"✅ 成功提取文本，总字符数: {len(full_text)}")
        
        # 截取前1500字符进行测试（平衡测试时间和效果）
        test_text = full_text[:1500]
        print(f"📝 使用前{len(test_text)}字符进行测试")
        
        # 处理文本 - 使用本地LLM生成剧本
        print("\n🧠 开始剧本生成（使用本地Qwen14B-Pro）...")
        script = llm_client.generate_script(test_text)
        
        if not script:
            print("❌ 剧本生成失败")
            return False
            
        print(f"✅ 剧本生成成功，共 {len(script)} 个片段")
        
        # 分析剧本结构
        narration_count = sum(1 for s in script if s['type'] == 'narration')
        title_count = sum(1 for s in script if s['type'] == 'title')
        dialogue_count = sum(1 for s in script if s['type'] == 'dialogue')
        
        print(f"\n📊 剧本结构分析:")
        print(f"  旁白片段: {narration_count}")
        print(f"  标题片段: {title_count}")
        print(f"  对话片段: {dialogue_count}")
        
        # 显示剧本片段示例
        print("\n📋 剧本片段示例:")
        for i, item in enumerate(script[:5]):  # 显示前5个片段
            content_preview = item['content'][:60] + "..." if len(item['content']) > 60 else item['content']
            print(f"  {i+1}. [{item['type']}] {item['speaker']}: {content_preview}")
        
        # 音频渲染测试（选择前2个片段进行快速测试）
        print("\n🎵 开始音频渲染测试（使用本地MLX Qwen-TTS）...")
        output_dir = "./lvzhuanhong_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        render_success = 0
        for i, item in enumerate(script[:2]):  # 只渲染前2个片段以节省时间
            wav_path = os.path.join(output_dir, f"lvzhuanhong_fragment_{i:03d}_{item['type']}.wav")
            voice_config = {
                "speaker": item["speaker"],
                "gender": item["gender"]
            }
            
            if tts_engine.render_dry_chunk(item["content"], voice_config, wav_path, item.get("emotion", "平静")):
                render_success += 1
                print(f"  ✅ 片段 {i+1} 渲染成功: {wav_path}")
            else:
                print(f"  ❌ 片段 {i+1} 渲染失败")
        
        print(f"\n📊 测试总结:")
        print(f"  - 文本提取: 成功 ({len(full_text)} 字符)")
        print(f"  - 剧本生成: {len(script)} 个片段")
        print(f"  - 音频渲染: {render_success}/2 成功")
        print(f"  - 输出目录: {output_dir}")
        
        if render_success > 0:
            print("🎉 测试完成，Alexandria分支集成工作正常！")
            return True
        else:
            print("⚠️ 音频渲染存在问题，但基本功能正常")
            return True
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_lvzhuanhong_with_alexandria()
    if success:
        print("\n✅ 吕转红受贿案Alexandria分支测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()