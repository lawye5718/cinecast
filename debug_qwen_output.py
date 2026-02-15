#!/usr/bin/env python3
"""
调试Qwen14B-Pro输出格式问题
检查为什么会出现JSON解析失败
"""

import json
import requests
import logging
from datetime import datetime

# 设置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qwen_direct_api():
    """直接测试Qwen14B-Pro API输出"""
    
    print("=" * 60)
    print("🔍 Qwen14B-Pro API输出格式调试")
    print("=" * 60)
    
    # 使用测试文本
    test_text = """
    第一章
    
    夜晚的港口总是显得格外神秘。老渔夫坐在岸边，凝视着远方的海面。
    
    "你相信命运吗？"老渔夫突然问道。
    
    年轻的助手沉默了一会儿，然后回答："我相信努力。"
    """
    
    # 构建API请求
    api_url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen14b-pro",
        "messages": [
            {
                "role": "system",
                "content": """
你是一位顶级的有声书导演兼数据清洗专家，负责将原始小说文本转换为标准化的录音剧本。
你必须严格遵守以下四大纪律，任何违反都将导致系统崩溃：

【一、 绝对忠实原则（Iron Rule）】
- 必须 100% 逐字保留原文内容！
- 严禁任何形式的概括、改写、缩写、续写或润色！
- 严禁自行添加原文中不存在的台词或动作描写！

【二、 字符净化原则】
- 剔除所有不可发音的特殊符号（如 Emoji表情、Markdown标记 * _ ~ #、制表符 \t、不可见控制字符）。
- 仅保留基础标点符号（，。！？：；、“”‘’（））。
- 数字、英文字母允许保留，但禁止出现复杂的数学公式符号。

【三、 粒度拆分原则】
- 必须将"对白"和"旁白/动作描写"严格剥离为独立的对象！
- 例如原文："你好，"老渔夫笑着说。
  必须拆分为两个对象：1. 角色对白("你好，") 2. 旁白描述("老渔夫笑着说。")

【四、 JSON 格式规范】
必须且只能输出合法的 JSON 数组，禁止任何解释性前言或后缀（如"好的，以下是..."），禁止输出 Markdown 代码块标记（```json）。
数组元素字段要求：
- "type": 仅限 "title"(章节名), "subtitle"(小标题), "narration"(旁白), "dialogue"(对白)。
- "speaker": 对白填具体的角色名（需根据上下文推断并保持全书统一）；旁白和标题统一填 "narrator"。
- "gender": 仅限 "male"、"female" 或 "unknown"。对白请推测性别；旁白固定为 "male"。
- "emotion": 情感标签（如"平静"、"激动"、"沧桑/叹息"、"愤怒"、"悲伤"等），用于未来语音合成的情感控制。
- "content": 纯净的文本内容。如果 type 是 "dialogue"，必须去掉最外层的引号（如""或""）。

【输出格式示例（One-Shot）】
[
  {
    "type": "narration",
    "speaker": "narrator",
    "gender": "male",
    "emotion": "平静",
    "content": "夜幕降临，港口的灯火开始闪烁。"
  },
  {
    "type": "dialogue",
    "speaker": "老渔夫",
    "gender": "male",
    "emotion": "沧桑/叹息",
    "content": "你相信命运吗？"
  },
  {
    "type": "narration",
    "speaker": "narrator",
    "gender": "male",
    "emotion": "平静",
    "content": "老渔夫说道。"
  }
]
"""
            },
            {
                "role": "user", 
                "content": f"请严格按照规范，将以下文本拆解为纯净的 JSON 剧本（绝不改写原意）：\n\n{test_text}"
            }
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": 8192,
            "temperature": 0.0,
            "top_p": 0.1
        }
    }
    
    print(f"\n📤 发送API请求...")
    print(f"📊 请求大小: {len(json.dumps(payload))} 字符")
    
    try:
        # 发送请求
        response = requests.post(api_url, json=payload, timeout=120)
        
        print(f"\n📥 收到响应:")
        print(f"📊 状态码: {response.status_code}")
        print(f"📊 响应大小: {len(response.text)} 字符")
        
        if response.status_code != 200:
            print(f"❌ HTTP错误: {response.status_code}")
            print(f"📝 错误内容: {response.text}")
            return False
            
        # 解析响应
        response_data = response.json()
        raw_content = response_data.get('message', {}).get('content', '')
        
        print(f"\n📄 原始响应内容:")
        print("-" * 40)
        print(raw_content)
        print("-" * 40)
        
        # 尝试JSON解析
        print(f"\n🧪 JSON解析测试:")
        
        # 1. 直接解析
        try:
            parsed_json = json.loads(raw_content)
            print("✅ 直接JSON解析成功")
            print(f"📊 解析结果类型: {type(parsed_json)}")
            if isinstance(parsed_json, list):
                print(f"📊 数组长度: {len(parsed_json)}")
                print("📊 前3个元素预览:")
                for i, item in enumerate(parsed_json[:3]):
                    print(f"  {i+1}. {item}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ 直接JSON解析失败: {e}")
            
        # 2. 清理Markdown标记后解析
        import re
        cleaned_content = re.sub(r'^```(?:json)?\s*', '', raw_content.strip(), flags=re.IGNORECASE)
        cleaned_content = re.sub(r'\s*```$', '', cleaned_content.strip())
        
        if cleaned_content != raw_content:
            print(f"🧹 清理Markdown标记后内容长度: {len(cleaned_content)} 字符")
            try:
                parsed_json = json.loads(cleaned_content)
                print("✅ 清理后JSON解析成功")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ 清理后JSON解析失败: {e}")
        
        # 3. 查找JSON数组模式
        print(f"\n🔍 尝试正则匹配JSON数组...")
        array_match = re.search(r'\[[\s\S]*\]', raw_content)
        if array_match:
            array_content = array_match.group()
            print(f"📊 找到可能的JSON数组，长度: {len(array_content)} 字符")
            try:
                parsed_json = json.loads(array_content)
                print("✅ 正则提取JSON解析成功")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ 正则提取JSON解析失败: {e}")
        else:
            print("❌ 未找到JSON数组模式")
            
        # 4. 正则降级方案测试
        print(f"\n🔄 测试正则降级方案...")
        fallback_result = test_regex_fallback(raw_content)
        if fallback_result:
            print("✅ 正则降级方案成功")
            return True
        else:
            print("❌ 正则降级方案也失败")
            
        return False
        
    except Exception as e:
        print(f"❌ API调用异常: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_regex_fallback(content):
    """测试正则降级方案"""
    import re
    
    # 使用local_llm_client中的正则模式
    pattern = re.compile(
        r'\{\s*'
        r'"(?:type)"\s*:\s*"([^"]*)"\s*,\s*'
        r'"(?:speaker)"\s*:\s*"([^"]*)"\s*,\s*'
        r'"(?:gender)"\s*:\s*"([^"]*)"\s*,\s*'
        r'"(?:emotion|instruct)"\s*:\s*"([^"]*)"\s*,\s*'
        r'"(?:content)"\s*:\s*"([^"]*)"',
        re.DOTALL,
    )
    
    entries = []
    for m in pattern.finditer(content):
        entries.append({
            "type": m.group(1) or "narration",
            "speaker": m.group(2) or "narrator",
            "gender": m.group(3) or "unknown",
            "emotion": m.group(4) or "平静",
            "content": m.group(5) or "",
        })
    
    print(f"📊 正则匹配结果: 找到 {len(entries)} 个条目")
    for i, entry in enumerate(entries[:3]):
        print(f"  {i+1}. [{entry['type']}] {entry['speaker']}: {entry['content'][:30]}...")
    
    return len(entries) > 0

def main():
    """主函数"""
    print("开始Qwen14B-Pro输出格式调试...")
    
    success = test_qwen_direct_api()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Qwen14B-Pro输出格式正常")
    else:
        print("❌ Qwen14B-Pro输出格式存在问题")
    print("=" * 60)

if __name__ == "__main__":
    main()