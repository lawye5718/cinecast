#!/usr/bin/env python3
"""
CineCast专用大模型测试
模拟实际生产环境中的提示词和场景
"""

import requests
import json
import time
from datetime import datetime

def test_cinecast_production_prompts():
    """测试CineCast生产环境使用的实际提示词"""
    print("=" * 80)
    print("🎬 CineCast生产环境大模型测试")
    print("=" * 80)
    
    base_url = "http://localhost:11434"
    model_name = "qwen14b-pro"
    
    # CineCast实际使用的系统提示词
    cinecast_system_prompt = """你是一个高精度的有声书剧本转换接口。
任务：将输入文本逐句解析为 JSON 数组格式。
核心规则：
1. 物理对齐：原文的每一句、每一段必须对应数组中的一个对象。严禁合并，严禁删减。
2. 根节点约束：输出结果必须是一个标准的 JSON 数组（即以 `[` 开头）。严禁输出 `{"data": [...]}` 这种格式。
3. 字段要求：每个对象必须包含 type, speaker, gender, emotion, content 字段。
4. 角色一致性：speaker 必须根据上下文推断。
5. 情绪约束：仅限 [平静, 激动, 悲伤, 愤怒, 惊讶, 疑惑]。"""
    
    test_cases = [
        {
            "name": "简单旁白测试",
            "text": "这是一个测试句子。",
            "expected_structure": "narration"
        },
        {
            "name": "对话测试",
            "text": '"你好，"他说。"今天天气不错。"',
            "expected_structure": "dialogue+narration"
        },
        {
            "name": "复杂叙事测试",
            "text": """老渔夫坐在海边，望着远方的大海。"这条船还能撑多久？"他喃喃自语道。海风吹过，带来一丝咸腥的味道。""",
            "expected_structure": "mixed"
        },
        {
            "name": "长文本测试",
            "text": """春天来了，大地苏醒。柳絮飞舞，桃花盛开。小鸟在枝头欢快地歌唱，仿佛在庆祝这个美好的季节。微风轻拂，带来阵阵花香。人们脱下厚重的冬衣，换上轻便的春装，走出家门享受温暖的阳光。""",
            "expected_structure": "long_narration"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📍 测试 {i}/{len(test_cases)}: {test_case['name']}")
        print("-" * 60)
        
        # 构造完整的用户提示词（模拟CineCast实际使用）
        user_prompt = f"""【指令：将以下文本转换为平铺的 JSON 数组，严禁最外层使用字典】

待处理原文：
{test_case['text']}"""
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": cinecast_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": 8192,
                "temperature": 0,
                "top_p": 0.1,
                "num_predict": 2048
            }
        }
        
        print(f"📤 发送提示词长度: {len(user_prompt)} 字符")
        start_time = time.time()
        
        try:
            response = requests.post(f"{base_url}/api/chat", json=payload, timeout=180)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                content = result_data.get('message', {}).get('content', '')
                
                print(f"✅ 请求成功 | 响应时间: {elapsed_time:.2f}秒")
                print(f"📝 原始响应长度: {len(content)} 字符")
                
                # 详细分析响应内容
                analysis = analyze_response(content, test_case)
                print_analysis(analysis)
                
                # 记录结果
                test_result = {
                    'test_name': test_case['name'],
                    'input_text': test_case['text'],
                    'response_time': elapsed_time,
                    'success': analysis['valid_json'] and analysis['correct_format'],
                    'analysis': analysis
                }
                results.append(test_result)
                
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                results.append({
                    'test_name': test_case['name'],
                    'input_text': test_case['text'],
                    'response_time': elapsed_time,
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                })
                
        except requests.Timeout:
            elapsed_time = time.time() - start_time
            print(f"⏰ 请求超时 (>{elapsed_time:.2f}秒)")
            results.append({
                'test_name': test_case['name'],
                'input_text': test_case['text'],
                'response_time': elapsed_time,
                'success': False,
                'error': 'timeout'
            })
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"💥 请求异常: {e}")
            results.append({
                'test_name': test_case['name'],
                'input_text': test_case['text'],
                'response_time': elapsed_time,
                'success': False,
                'error': str(e)
            })
    
    # 生成测试报告
    generate_report(results)

def analyze_response(content: str, test_case: dict) -> dict:
    """分析响应内容"""
    analysis = {
        'valid_json': False,
        'correct_format': False,
        'has_required_fields': False,
        'array_structure': False,
        'element_count': 0,
        'parsed_data': None,
        'issues': []
    }
    
    # 预处理内容
    clean_content = content.strip()
    if clean_content.startswith('```json'):
        clean_content = clean_content[7:]
    if clean_content.endswith('```'):
        clean_content = clean_content[:-3]
    clean_content = clean_content.strip()
    
    # 尝试JSON解析
    try:
        parsed_data = json.loads(clean_content)
        analysis['valid_json'] = True
        analysis['parsed_data'] = parsed_data
        
        # 检查是否为数组
        if isinstance(parsed_data, list):
            analysis['array_structure'] = True
            analysis['element_count'] = len(parsed_data)
            
            # 检查必要字段
            required_fields = ['type', 'speaker', 'content']
            if parsed_data:
                all_have_fields = all(
                    all(field in item for field in required_fields) 
                    for item in parsed_data 
                    if isinstance(item, dict)
                )
                analysis['has_required_fields'] = all_have_fields and len(parsed_data) > 0
            
            # 检查格式正确性
            analysis['correct_format'] = (
                analysis['array_structure'] and 
                analysis['has_required_fields']
            )
            
        else:
            analysis['issues'].append("响应不是数组格式")
            
    except json.JSONDecodeError as e:
        analysis['issues'].append(f"JSON解析失败: {e}")
    
    return analysis

def print_analysis(analysis: dict):
    """打印分析结果"""
    print(f"   JSON有效性: {'✅' if analysis['valid_json'] else '❌'}")
    print(f"   数组结构: {'✅' if analysis['array_structure'] else '❌'}")
    print(f"   必要字段: {'✅' if analysis['has_required_fields'] else '❌'}")
    print(f"   元素数量: {analysis['element_count']}")
    
    if analysis['issues']:
        print(f"   问题: {', '.join(analysis['issues'])}")
    
    if analysis['parsed_data'] and isinstance(analysis['parsed_data'], list):
        print("   解析结果预览:")
        for i, item in enumerate(analysis['parsed_data'][:3]):  # 只显示前3个
            if isinstance(item, dict):
                preview = {k: v for k, v in item.items() if k in ['type', 'speaker', 'content']}
                print(f"     [{i+1}] {preview}")

def generate_report(results: list):
    """生成测试报告"""
    print("\n" + "=" * 80)
    print("📊 测试报告")
    print("=" * 80)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests
    
    print(f"📈 总体统计:")
    print(f"   总测试数: {total_tests}")
    print(f"   成功: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
    print(f"   失败: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    
    response_times = [r['response_time'] for r in results]
    print(f"   平均响应时间: {sum(response_times)/len(response_times):.2f}秒")
    print(f"   最快响应: {min(response_times):.2f}秒")
    print(f"   最慢响应: {max(response_times):.2f}秒")
    
    print(f"\n📋 详细结果:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"   {result['test_name']}: {status} ({result['response_time']:.2f}秒)")
        if not result['success'] and 'error' in result:
            print(f"     错误: {result['error']}")
    
    # 保存报告
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'average_response_time': sum(response_times)/len(response_times)
        },
        'detailed_results': results
    }
    
    with open('cinecast_model_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试报告已保存到: cinecast_model_test_report.json")

if __name__ == "__main__":
    test_cinecast_production_prompts()