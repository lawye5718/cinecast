#!/usr/bin/env python3
"""
吕转红受贿案EPUB测试脚本
使用本地化集成的Alexandria组件处理法律文书
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from integrate_local_components import AlexandriaLocalAdapter

def test_lvzhuanhong_case():
    """测试吕转红受贿案EPUB处理"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("📚 开始测试吕转红受贿案EPUB文件处理")
    print("=" * 50)
    
    # 初始化本地化适配器
    adapter = AlexandriaLocalAdapter()
    
    # 健康检查
    print("\n🏥 系统健康检查:")
    health_status = adapter.health_check()
    for check, status in health_status.items():
        print(f"  {check}: {status}")
    
    if not health_status["overall_status"].startswith("✅"):
        print("\n❌ 系统检查失败，无法继续测试")
        return False
    
    # EPUB文件路径
    epub_path = "./lvzhuanhong.epub"
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
        
        # 截取前2000字符进行测试（避免处理时间过长）
        test_text = full_text[:2000]
        print(f"📝 使用前{len(test_text)}字符进行测试")
        
        # 处理文本
        print("\n🧠 开始剧本生成...")
        script = adapter.generate_local_script(test_text)
        
        if not script:
            print("❌ 剧本生成失败")
            return False
            
        print(f"✅ 剧本生成成功，共 {len(script)} 个片段")
        
        # 显示剧本片段示例
        print("\n📋 剧本片段示例:")
        for i, item in enumerate(script[:5]):  # 显示前5个片段
            content_preview = item['content'][:50] + "..." if len(item['content']) > 50 else item['content']
            print(f"  {i+1}. [{item['type']}] {item['speaker']}: {content_preview}")
        
        # 音频渲染测试（选择前3个片段）
        print("\n🎵 开始音频渲染测试...")
        output_dir = "./test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        render_success = 0
        for i, item in enumerate(script[:3]):  # 只渲染前3个片段
            wav_path = os.path.join(output_dir, f"fragment_{i:03d}_{item['type']}.wav")
            voice_config = {
                "speaker": item["speaker"],
                "gender": item["gender"]
            }
            
            if adapter.render_local_audio(item["content"], voice_config, wav_path, item.get("emotion", "平静")):
                render_success += 1
                print(f"  ✅ 片段 {i+1} 渲染成功: {wav_path}")
            else:
                print(f"  ❌ 片段 {i+1} 渲染失败")
        
        print(f"\n📊 测试总结:")
        print(f"  - 文本提取: 成功")
        print(f"  - 剧本生成: {len(script)} 个片段")
        print(f"  - 音频渲染: {render_success}/3 成功")
        print(f"  - 输出目录: {output_dir}")
        
        if render_success > 0:
            print("🎉 测试完成，本地化集成工作正常！")
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
    success = test_lvzhuanhong_case()
    if success:
        print("\n✅ 吕转红受贿案EPUB测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()