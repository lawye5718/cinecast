#!/usr/bin/env python3
"""
CineCast 流式API服务启动脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """启动流式API服务"""
    try:
        import uvicorn
        from modules.stream_api import app
        
        print("🚀 启动 CineCast 流式 TTS API 服务...")
        print("📍 服务地址: http://localhost:8000")
        print("📊 API文档: http://localhost:8000/docs")
        print("🏥 健康检查: http://localhost:8000/health")
        print("🎤 音色列表: http://localhost:8000/voices")
        print("⏹️  按 Ctrl+C 停止服务")
        print("-" * 50)
        
        # 启动服务
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # 生产环境关闭热重载
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()