#!/usr/bin/env python3
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
