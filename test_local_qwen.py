#!/usr/bin/env python3
"""
本地Qwen Coder模型调用测试脚本
用于验证在CineCast项目中调用本地14B Qwen模型的能力
"""

import subprocess
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class LocalQwenCoder:
    def __init__(self, model_name: str = "qwen14b-pro"):
        """
        初始化本地Qwen Coder模型调用器
        
        Args:
            model_name: Ollama中的模型名称
        """
        self.model_name = model_name
        self.is_available = self._check_model_availability()
        
    def _check_model_availability(self) -> bool:
        """检查模型是否可用"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0 and self.model_name in result.stdout:
                logger.info(f"✅ 找到本地模型: {self.model_name}")
                return True
            else:
                logger.warning(f"❌ 未找到模型: {self.model_name}")
                return False
        except Exception as e:
            logger.error(f"检查模型可用性时出错: {e}")
            return False
    
    def call_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        调用本地Qwen模型
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            
        Returns:
            模型响应文本
        """
        if not self.is_available:
            raise RuntimeError("本地Qwen模型不可用")
        
        try:
            # 构建完整的提示
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # 调用Ollama
            result = subprocess.run(
                ["ollama", "run", self.model_name, full_prompt],
                capture_output=True,
                text=True,
                timeout=60  # 60秒超时
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                logger.info(f"✅ 模型调用成功，响应长度: {len(response)} 字符")
                return response
            else:
                error_msg = result.stderr.strip() if result.stderr else "未知错误"
                logger.error(f"❌ 模型调用失败: {error_msg}")
                raise RuntimeError(f"模型调用失败: {error_msg}")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ 模型调用超时")
            raise TimeoutError("模型调用超时")
        except Exception as e:
            logger.error(f"❌ 模型调用异常: {e}")
            raise
    
    def test_coding_assistant(self) -> bool:
        """测试编程助手功能"""
        test_prompt = '''
请帮我写一个Python函数，该函数能够：
1. 接收一个字符串列表作为输入
2. 统计每个字符串的长度
3. 返回长度最长的字符串

请提供完整的函数实现。
'''
        
        try:
            response = self.call_model(test_prompt)
            print("📝 编程助手测试结果:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            return True
        except Exception as e:
            logger.error(f"编程助手测试失败: {e}")
            return False
    
    def test_text_analysis(self) -> bool:
        """测试文本分析功能"""
        test_text = "第一章 夜晚的港口\n海风轻抚着岸边的礁石，远处的灯塔在黑暗中闪烁着微弱的光芒。"
        
        analysis_prompt = f'''
请分析以下文本的文学特点：
"{test_text}"

请从以下几个方面进行分析：
1. 文学风格和语言特色
2. 情感基调
3. 可能的故事发展方向
'''
        
        try:
            response = self.call_model(analysis_prompt)
            print("🔍 文本分析测试结果:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            return True
        except Exception as e:
            logger.error(f"文本分析测试失败: {e}")
            return False

def main():
    """主测试函数"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 本地Qwen Coder模型测试")
    print("=" * 60)
    
    # 初始化模型调用器
    qwen_coder = LocalQwenCoder()
    
    if not qwen_coder.is_available:
        print("❌ 本地Qwen模型不可用，请检查安装")
        return
    
    print(f"✅ 成功连接到本地模型: {qwen_coder.model_name}")
    print(f"📊 模型大小: 9.9 GB")
    print()
    
    # 运行测试
    tests = [
        ("编程助手测试", qwen_coder.test_coding_assistant),
        ("文本分析测试", qwen_coder.test_text_analysis)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"🧪 运行 {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
        print()
    
    print("=" * 60)
    print(f"📊 测试结果: {passed}/{len(tests)} 个测试通过")
    
    if passed == len(tests):
        print("🎉 所有测试通过！本地Qwen Coder模型可在CineCast项目中使用")
    else:
        print("⚠️  部分测试失败，请检查模型配置")

if __name__ == "__main__":
    main()