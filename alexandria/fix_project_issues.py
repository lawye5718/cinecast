#!/usr/bin/env python3
"""
Alexandria项目综合修复脚本
应用CineCast中验证的成功实现，修复音频生成和LLM处理问题
"""

import os
import sys
import json
import threading
import time
import requests
from pathlib import Path

# 添加项目路径
project_root = Path("/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook")
sys.path.insert(0, str(project_root))

def fix_audio_generation_issues():
    """修复音频生成问题"""
    print("🔧 修复音频生成问题...")
    
    # 修复project.py中的音频生成逻辑
    project_py_path = project_root / "app" / "project.py"
    
    with open(project_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复音频文件检查逻辑，确保检查文件大小
    content = content.replace(
        'if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:',
        'if not os.path.exists(temp_path):\n                print(f"DEBUG: Temp file does not exist: {temp_path}")\n                self._update_chunk_fields(index, status="error")\n                return False, "Generated audio file does not exist"\n            elif os.path.getsize(temp_path) <= 44:  # WAV文件头至少44字节\n                print(f"DEBUG: Temp file too small: {temp_path}, size: {os.path.getsize(temp_path)})\n                self._update_chunk_fields(index, status="error")\n                return False, "Generated audio file is too small (< 44 bytes)"'
    )
    
    # 修复音频保存逻辑，确保正确保存数据
    content = content.replace(
        'sf.write(output_path, audio_array, sample_rate)',
        '# 确保音频数据是正确的numpy数组格式\n            if not isinstance(audio_array, np.ndarray):\n                audio_array = np.array(audio_array)\n            if audio_array.ndim > 1:\n                audio_array = audio_array.flatten()\n            \n            # 验证音频数据\n            if audio_array.size == 0:\n                print(f"ERROR: Generated audio array is empty for: {output_path}")\n                return False, "Generated audio array is empty"\n            \n            # 保存音频文件\n            sf.write(output_path, audio_array, sample_rate)\n            print(f"DEBUG: Audio saved to {output_path}, size: {os.path.getsize(output_path)} bytes")'
    )
    
    # 保存修改
    with open(project_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ project.py 中的音频生成逻辑已修复")

def implement_serial_llm_processing():
    """实现串行LLM处理"""
    print("🔄 实现串行LLM处理以避免内存冲突...")
    
    # 创建串行LLM客户端
    serial_llm_client_content = '''import threading
import time
import requests
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SerialLocalLLMClient:
    """串行本地LLM客户端 - 确保一次只处理一个请求以避免内存冲突"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("llm", {})
        self.api_url = self.config.get("api_url", "http://localhost:11434/api/chat")
        self.model_name = self.config.get("model", "qwen14b-pro")
        self.temperature = self.config.get("temperature", 0.0)
        self.num_ctx = self.config.get("num_ctx", 8192)
        
        # 串行锁，确保一次只处理一个请求
        self._request_lock = threading.Lock()
        
        # 验证连接
        self._verify_connection()
    
    def _verify_connection(self):
        """验证与本地LLM服务的连接"""
        try:
            # 测试请求
            test_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 512}
            }
            
            response = requests.post(self.api_url, json=test_payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ 本地LLM服务连接正常: {self.model_name}")
            else:
                logger.warning(f"⚠️ 本地LLM服务响应异常: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ 无法连接到本地LLM服务: {e}")
    
    def generate_script(self, text_chunk: str, context: str = "") -> List[Dict]:
        """串行生成脚本 - 一次只处理一个请求"""
        with self._request_lock:  # 串行执行
            logger.info(f"🔒 串行锁已获取，开始处理LLM请求...")
            start_time = time.time()
            
            try:
                result = self._generate_script_internal(text_chunk, context)
                end_time = time.time()
                logger.info(f"✅ LLM请求处理完成，耗时: {end_time - start_time:.2f}秒")
                return result
            except Exception as e:
                end_time = time.time()
                logger.error(f"❌ LLM请求处理失败，耗时: {end_time - start_time:.2f}秒, 错误: {e}")
                raise
            finally:
                logger.info("🔓 串行锁已释放")
    
    def _generate_script_internal(self, text_chunk: str, context: str = "") -> List[Dict]:
        """内部生成方法 - 基于CineCast中验证的实现"""
        system_prompt = """
你是一位顶级的有声书导演兼数据清洗专家，负责将原始小说文本转换为标准化的录音剧本。
你必须严格遵守以下四大纪律，任何违反都将导致系统崩溃：

【一、 绝对忠实原则（Iron Rule）】
- 必须 100% 逐字保留原文内容！
- 严禁任何形式的概括、改写、缩写、续写或润色！
- 严禁自行添加原文中不存在的台词或动作描写！
- 严禁在 content 中保留归属标签（如"他说"、"她叫道"），归属信息只能出现在 speaker 字段！

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

        # 构建用户提示
        user_prompt = f"请严格按照规范，将以下文本拆解为纯净的 JSON 剧本（绝不改写原意）：\\n\\n{text_chunk}"
        if context:
            user_prompt = f"上下文信息：{context}\\n\\n{user_prompt}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "top_p": 0.1
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            content = response.json().get('message', {}).get('content', '[]')

            # 清理Markdown代码块
            import re
            content = re.sub(r'^```(?:json)?\\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\\s*```$', '', content.strip())

            # 解析JSON
            script = json.loads(content)

            # 验证并修复数据结构
            if isinstance(script, list):
                return self._validate_script_elements(script)
            elif isinstance(script, dict):
                # 处理包装格式
                for value in script.values():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        return self._validate_script_elements(value)

            # 降级到正则解析
            logger.warning("⚠️ JSON解析失败，使用正则降级方案")
            return self._fallback_regex_parse(text_chunk)

        except Exception as e:
            logger.error(f"❌ LLM剧本生成失败: {e}")
            return self._fallback_regex_parse(text_chunk)
    
    def _validate_script_elements(self, script: List[Dict]) -> List[Dict]:
        """验证并修复脚本元素"""
        required_fields = ['type', 'speaker', 'content']
        validated_script = []

        for i, element in enumerate(script):
            if not isinstance(element, dict):
                logger.warning(f"⚠️ 脚本元素 {i} 不是字典类型，跳过: {element}")
                continue

            fixed_element = element.copy()

            # 补充缺失字段
            for field in required_fields:
                if field not in fixed_element:
                    if field == 'type':
                        fixed_element['type'] = 'narration'
                    elif field == 'speaker':
                        fixed_element['speaker'] = 'narrator'
                    elif field == 'content':
                        fixed_element['content'] = ''
                    logger.warning(f"⚠️ 补充缺失字段 '{field}'")

            # 确保其他必需字段
            if 'gender' not in fixed_element:
                fixed_element['gender'] = 'unknown'
            if 'emotion' not in fixed_element:
                fixed_element['emotion'] = '平静'

            validated_script.append(fixed_element)

        return validated_script

    def _fallback_regex_parse(self, text: str) -> List[Dict]:
        """正则降级解析方案"""
        import re

        units = []
        lines = text.split('\\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测标题
            if self._is_title(line):
                units.append({
                    "type": "title",
                    "speaker": "narrator",
                    "gender": "unknown",
                    "emotion": "平静",
                    "content": line
                })
            # 检测对话
            elif self._is_dialogue(line):
                speaker, content = self._extract_dialogue_components(line)
                gender = self._predict_gender(speaker)
                units.append({
                    "type": "dialogue",
                    "speaker": speaker,
                    "gender": gender,
                    "emotion": "平静",
                    "content": content
                })
            # 默认为旁白
            else:
                units.append({
                    "type": "narration",
                    "speaker": "narrator",
                    "gender": "unknown",
                    "emotion": "平静",
                    "content": line
                })

        return units

    def _is_title(self, text: str) -> bool:
        """判断是否为标题"""
        import re
        if len(text) < 30 and re.search(r'[第章节卷部集]', text):
            return True
        if text.isupper() and len(text) < 50:
            return True
        return False

    def _is_dialogue(self, text: str) -> bool:
        """判断是否为对话"""
        return ('"' in text or '"' in text or
                '""' in text or '""' in text)

    def _extract_dialogue_components(self, line: str) -> tuple:
        """提取对话组件"""
        import re
        # 简单的对话提取逻辑
        match = re.search(r'^(.*?)[""""](.*)["""]', line)
        if match:
            speaker = match.group(1).strip().rstrip(':：')
            content = match.group(2).strip()
            return speaker if speaker else "未知角色", content
        return "未知角色", line

    def _predict_gender(self, speaker_name: str) -> str:
        """简单性别预测"""
        female_indicators = ['女士', '小姐', '夫人', '妈妈', '姐姐', '妹妹', '女儿']
        male_indicators = ['先生', '少爷', '老爷', '爸爸', '哥哥', '弟弟', '儿子']

        for indicator in female_indicators:
            if indicator in speaker_name:
                return "female"
        for indicator in male_indicators:
            if indicator in speaker_name:
                return "male"

        return "unknown"

# 兼容性函数
def create_serial_local_llm_client(config: Dict[str, Any]):
    """创建串行本地LLM客户端实例"""
    return SerialLocalLLMClient(config)
'''

    # 写入串行LLM客户端文件
    with open(project_root / "serial_local_llm_client.py", 'w', encoding='utf-8') as f:
        f.write(serial_llm_client_content)
    
    print("✅ 串行LLM客户端已创建")

def update_config_for_local_models():
    """更新配置以使用本地已验证的模型"""
    print("⚙️ 更新配置以使用本地已验证的模型...")
    
    config_path = project_root / "config.json"
    
    # 读取现有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 更新LLM配置为本地已验证的qwen14b-pro模型
    config["llm"] = {
        "api_url": "http://localhost:11434/api/chat",
        "model": "qwen14b-pro",  # 使用cinecast中验证的模型
        "temperature": 0.0,
        "num_ctx": 8192
    }
    
    # 更新TTS配置为本地MLX Qwen模型
    config["tts"] = {
        "mode": "local",  # 使用本地模式
        "device": "auto",
        "language": "Chinese",
        "compile_codec": False
    }
    
    # 保存更新后的配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ 配置文件已更新为本地已验证模型")

def create_single_chat_contact_discovery():
    """创建单聊联系人发现功能"""
    print("👤 创建单聊联系人发现功能...")
    
    discovery_script = '''#!/usr/bin/env python3
"""
钉钉单聊联系人发现工具
基于CineCast中验证的实现
用于获取用户ID以便后续单聊消息发送
"""

import asyncio
import os
import json
import logging
from typing import Dict, Any
import threading
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DingTalkContactDiscovery:
    """钉钉联系人发现器"""
    
    def __init__(self, storage_file="dingtalk_contacts.json"):
        self.storage_file = storage_file
        self.contacts = self._load_contacts()
        self.discovered_users = set()  # 避免重复记录
        
    def _load_contacts(self) -> Dict[str, Any]:
        """加载已发现的联系人"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载联系人文件失败: {e}")
                return {}
        return {}
    
    def _save_contacts(self):
        """保存联系人信息"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.contacts, f, ensure_ascii=False, indent=2)
            logger.info(f"联系人信息已保存到: {self.storage_file}")
        except Exception as e:
            logger.error(f"保存联系人文件失败: {e}")
    
    def record_contact(self, user_info: Dict[str, Any]):
        """记录联系人信息"""
        user_id = user_info.get('user_id') or user_info.get('sender_user_id')
        if not user_id:
            logger.warning("用户信息中缺少用户ID，无法记录")
            return False
        
        # 避免重复记录
        if user_id in self.discovered_users:
            logger.debug(f"用户 {user_id} 已记录，跳过")
            return True
        
        # 生成唯一标识符
        unique_id = user_info.get('union_id', user_id)
        
        contact_info = {
            "user_id": user_id,
            "union_id": user_info.get('union_id', ''),
            "nick_name": user_info.get('nick_name', user_info.get('sender_nick', 'Unknown')),
            "avatar_url": user_info.get('avatar_url', ''),
            "department": user_info.get('department', ''),
            "position": user_info.get('position', ''),
            "first_contact_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "last_contact_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "contact_count": 1
        }
        
        # 更新或添加联系人
        if unique_id in self.contacts:
            # 更新现有联系人信息
            existing = self.contacts[unique_id]
            existing.update(contact_info)
            existing['last_contact_time'] = contact_info['last_contact_time']
            existing['contact_count'] += 1
        else:
            # 添加新联系人
            self.contacts[unique_id] = contact_info
        
        self.discovered_users.add(user_id)
        self._save_contacts()
        
        logger.info(f"✅ 联系人已记录: {contact_info['nick_name']} (ID: {user_id[:8]}...)")
        return True
    
    def get_contact_by_id(self, user_id: str) -> Dict[str, Any]:
        """根据用户ID获取联系人信息"""
        for contact_id, contact_info in self.contacts.items():
            if contact_info.get('user_id') == user_id:
                return contact_info
        return {}
    
    def get_all_contacts(self) -> Dict[str, Any]:
        """获取所有联系人"""
        return self.contacts
    
    def add_manual_contact(self, user_id: str, nick_name: str, **kwargs) -> bool:
        """手动添加联系人"""
        contact_info = {
            "user_id": user_id,
            "union_id": kwargs.get('union_id', ''),
            "nick_name": nick_name,
            "avatar_url": kwargs.get('avatar_url', ''),
            "department": kwargs.get('department', ''),
            "position": kwargs.get('position', ''),
            "first_contact_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "last_contact_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "contact_count": 1,
            "manually_added": True
        }
        
        unique_id = kwargs.get('union_id', user_id)
        self.contacts[unique_id] = contact_info
        self.discovered_users.add(user_id)
        self._save_contacts()
        
        logger.info(f"✅ 手动联系人已添加: {nick_name} (ID: {user_id})")
        return True
    
    def export_contacts(self, export_path: str = "dingtalk_contacts_export.json"):
        """导出联系人列表"""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.contacts, f, ensure_ascii=False, indent=2)
            logger.info(f"联系人已导出到: {export_path}")
            return True
        except Exception as e:
            logger.error(f"导出联系人失败: {e}")
            return False

def setup_single_chat_contacts():
    """设置单聊联系人发现功能"""
    print("🔧 设置钉钉单聊联系人发现功能...")
    
    # 创建发现器实例
    discovery = DingTalkContactDiscovery()
    
    # 创建联系人配置模板
    contacts_config_template = {
        "single_chat_recipients": [],
        "auto_discovery_enabled": True,
        "discovery_storage_file": "dingtalk_contacts.json",
        "last_discovery_time": None,
        "total_discovered_contacts": len(discovery.get_all_contacts())
    }
    
    # 保存配置模板
    with open("single_chat_contacts_config.json", "w", encoding="utf-8") as f:
        json.dump(contacts_config_template, f, ensure_ascii=False, indent=2)
    
    print("✅ 单聊联系人发现功能已设置")
    print("💡 使用说明:")
    print("   1. 启动钉钉机器人监听服务")
    print("   2. 让目标用户向机器人发送消息")
    print("   3. 系统将自动记录用户ID到dingtalk_contacts.json")
    print("   4. 使用这些ID进行单聊消息发送")
    
    return discovery

if __name__ == "__main__":
    discovery = setup_single_chat_contacts()
    print(f"📋 已发现联系人数量: {len(discovery.get_all_contacts())}")
'''
    
    # 写入发现脚本
    with open(project_root / "dingtalk_contact_discovery.py", 'w', encoding='utf-8') as f:
        f.write(discovery_script)
    
    print("✅ 单聊联系人发现脚本已创建")

def update_requirements():
    """更新依赖要求"""
    print("📦 更新依赖要求...")
    
    requirements_content = """# Alexandria Audiobook Generator 依赖
# Python 3.12+ 版本

# 核心依赖
numpy>=1.24.0
pandas>=2.0.0
requests>=2.31.0
pydub>=0.25.1
soundfile>=0.12.0
librosa>=0.10.0

# MLX 依赖 (用于本地Qwen-TTS)
mlx>=0.15.0
mlx-lm>=0.15.0
mlx-audio>=0.1.0

# Web框架
fastapi>=0.104.0
uvicorn>=0.24.0
pydub>=0.25.1

# 配置管理
pyyaml>=6.0
python-dotenv>=1.0.0

# 工具库
tqdm>=4.66.0
click>=8.1.0
tenacity>=8.2.0

# 开发工具
pytest>=7.4.0
black>=23.0.0
mypy>=1.7.0
"""
    
    with open(project_root / "requirements.txt", 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    print("✅ 依赖要求已更新")

def main():
    """主修复函数"""
    print("🚀 Alexandria项目综合修复开始")
    print("="*60)
    
    # 执行所有修复
    fix_audio_generation_issues()
    implement_serial_llm_processing()
    update_config_for_local_models()
    create_single_chat_contact_discovery()
    update_requirements()
    
    print("="*60)
    print("✅ 所有修复已完成！")
    print("\n📋 修复内容总结:")
    print("   1. 修复了音频生成问题（0字节WAV文件）")
    print("   2. 实现了串行LLM处理以避免内存冲突")
    print("   3. 更新配置使用本地已验证模型")
    print("   4. 创建了单聊联系人发现功能")
    print("   5. 更新了项目依赖")
    
    print("\n💡 下一步操作:")
    print("   1. 安装更新的依赖: pip3 install -r requirements.txt")
    print("   2. 确保本地Ollama服务运行: ollama serve")
    print("   3. 确保qwen14b-pro模型已下载: ollama pull qwen14b-pro")
    print("   4. 运行项目测试修复效果")
    
    print("\n🎉 Alexandria项目修复完成！")

if __name__ == "__main__":
    main()