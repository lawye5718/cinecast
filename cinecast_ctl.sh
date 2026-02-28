#!/bin/bash

# CineCast服务管理快捷脚本
# 使用方法: ./cinecast_ctl.sh {start|stop|restart|status}

SCRIPT_PATH="/Users/yuanliang/superstar/superstar3.1/projects/cinecast/start_cinecast_daemon.sh"

case "$1" in
    start)
        echo "启动CineCast服务..."
        "$SCRIPT_PATH" start
        ;;
    stop)
        echo "停止CineCast服务..."
        "$SCRIPT_PATH" stop
        ;;
    restart)
        echo "重启CineCast服务..."
        "$SCRIPT_PATH" restart
        ;;
    status)
        echo "检查CineCast服务状态..."
        "$SCRIPT_PATH" status
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        echo ""
        echo "或者使用系统命令:"
        echo "  启动: launchctl start com.cinecast.daemon"
        echo "  停止: launchctl stop com.cinecast.daemon"
        echo "  重启: launchctl kickstart -k gui/$(id -u)/com.cinecast.daemon"
        echo "  状态: launchctl list | grep cinecast"
        exit 1
        ;;
esac