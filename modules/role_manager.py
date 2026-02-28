#!/usr/bin/env python3
"""
CineCast 音色资产管理器 (Role Manager)
管理音色特征的持久化存储，支持 NPZ 格式的角色音色库。
实现 Voice Cards 跨项目复用。
"""

import json
import os
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RoleManager:
    """音色库管理器。

    管理角色名与音色特征文件的映射关系。
    使用 JSON 存储元数据，使用 NPZ 存储特征向量。
    """

    def __init__(self, roles_dir: str = "./voices"):
        """初始化角色管理器。

        Args:
            roles_dir: 角色音色库目录路径
        """
        self.roles_dir = roles_dir
        os.makedirs(roles_dir, exist_ok=True)

    @staticmethod
    def save_voice_feature(feature_dict: Dict[str, np.ndarray],
                           role_name: str,
                           roles_dir: str = "./voices",
                           metadata: Optional[Dict] = None):
        """持久化角色音色特征到 NPZ 文件。

        Args:
            feature_dict: 特征向量字典（键为特征名，值为 numpy 数组）
            role_name: 角色名称
            roles_dir: 保存目录
            metadata: 可选的元数据字典（描述、语言等）
        """
        os.makedirs(roles_dir, exist_ok=True)
        npz_path = os.path.join(roles_dir, f"{role_name}.npz")
        np.savez(npz_path, **feature_dict)
        logger.info(f"💾 角色 '{role_name}' 特征已保存: {npz_path}")

        # 保存元数据 JSON
        if metadata:
            meta_path = os.path.join(roles_dir, f"{role_name}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"📋 角色 '{role_name}' 元数据已保存: {meta_path}")

    @staticmethod
    def load_voice_feature(role_name: str,
                           roles_dir: str = "./voices") -> Optional[Dict[str, np.ndarray]]:
        """加载单个角色的音色特征。

        Args:
            role_name: 角色名称
            roles_dir: 音色库目录

        Returns:
            特征向量字典，或 None（文件不存在时）
        """
        npz_path = os.path.join(roles_dir, f"{role_name}.npz")
        if not os.path.exists(npz_path):
            logger.warning(f"⚠️ 角色 '{role_name}' 特征文件不存在: {npz_path}")
            return None
        data = np.load(npz_path, allow_pickle=False)
        feature = dict(data)
        data.close()
        logger.info(f"🎤 已加载角色 '{role_name}' 特征: {list(feature.keys())}")
        return feature

    @staticmethod
    def load_voice_metadata(role_name: str,
                            roles_dir: str = "./voices") -> Optional[Dict]:
        """加载角色的元数据。

        Args:
            role_name: 角色名称
            roles_dir: 音色库目录

        Returns:
            元数据字典，或 None
        """
        meta_path = os.path.join(roles_dir, f"{role_name}.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_role_bank(self, role_names: Optional[List[str]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """加载多角色音色库。

        Args:
            role_names: 要加载的角色名列表。为 None 时自动扫描目录中所有 .npz 文件。

        Returns:
            角色音色库字典 {角色名: 特征字典}
        """
        bank = {}

        if role_names is None:
            # 自动扫描目录
            if not os.path.exists(self.roles_dir):
                logger.warning(f"⚠️ 角色库目录不存在: {self.roles_dir}")
                return bank
            role_names = []
            for f in os.listdir(self.roles_dir):
                if f.endswith(".npz"):
                    role_names.append(os.path.splitext(f)[0])

        for name in role_names:
            feature = self.load_voice_feature(name, self.roles_dir)
            if feature is not None:
                bank[name] = feature

        logger.info(f"📚 角色库加载完成: {len(bank)} 个角色 ({list(bank.keys())})")
        return bank

    def list_roles(self) -> List[Dict]:
        """列出音色库中所有可用角色。

        Returns:
            角色信息列表，每个元素包含 name, has_metadata, feature_keys
        """
        roles = []
        if not os.path.exists(self.roles_dir):
            return roles

        for f in os.listdir(self.roles_dir):
            if f.endswith(".npz"):
                name = os.path.splitext(f)[0]
                metadata = self.load_voice_metadata(name, self.roles_dir)
                feature = self.load_voice_feature(name, self.roles_dir)
                roles.append({
                    "name": name,
                    "has_metadata": metadata is not None,
                    "metadata": metadata,
                    "feature_keys": list(feature.keys()) if feature else [],
                })
        return roles

    def delete_role(self, role_name: str) -> bool:
        """删除指定角色的音色数据。

        Args:
            role_name: 角色名称

        Returns:
            是否成功删除
        """
        deleted = False
        for ext in (".npz", ".json"):
            path = os.path.join(self.roles_dir, f"{role_name}{ext}")
            if os.path.exists(path):
                os.remove(path)
                deleted = True
                logger.info(f"🗑️ 已删除: {path}")

        if not deleted:
            logger.warning(f"⚠️ 角色 '{role_name}' 不存在")
        return deleted

    def export_voice_card(self, role_name: str, export_dir: str) -> Optional[str]:
        """导出角色 Voice Card（NPZ + JSON 元数据打包）。

        Args:
            role_name: 角色名称
            export_dir: 导出目录

        Returns:
            导出文件路径，或 None
        """
        os.makedirs(export_dir, exist_ok=True)

        npz_src = os.path.join(self.roles_dir, f"{role_name}.npz")
        if not os.path.exists(npz_src):
            logger.warning(f"⚠️ 角色 '{role_name}' NPZ 文件不存在")
            return None

        # 复制 NPZ
        import shutil
        npz_dst = os.path.join(export_dir, f"{role_name}.npz")
        shutil.copy2(npz_src, npz_dst)

        # 复制元数据（如果有）
        meta_src = os.path.join(self.roles_dir, f"{role_name}.json")
        if os.path.exists(meta_src):
            meta_dst = os.path.join(export_dir, f"{role_name}.json")
            shutil.copy2(meta_src, meta_dst)

        logger.info(f"📦 Voice Card 已导出: {export_dir}/{role_name}")
        return npz_dst

    def import_voice_card(self, card_path: str) -> Optional[str]:
        """导入 Voice Card 到角色库。

        Args:
            card_path: Voice Card 的 NPZ 文件路径

        Returns:
            角色名称，或 None
        """
        if not os.path.exists(card_path) or not card_path.endswith(".npz"):
            logger.warning(f"⚠️ 无效的 Voice Card 路径: {card_path}")
            return None

        import shutil
        role_name = os.path.splitext(os.path.basename(card_path))[0]
        dst = os.path.join(self.roles_dir, f"{role_name}.npz")
        shutil.copy2(card_path, dst)

        # 尝试导入配套的 JSON 元数据
        meta_src = card_path.replace(".npz", ".json")
        if os.path.exists(meta_src):
            meta_dst = os.path.join(self.roles_dir, f"{role_name}.json")
            shutil.copy2(meta_src, meta_dst)

        logger.info(f"📥 Voice Card 已导入: {role_name}")
        return role_name
