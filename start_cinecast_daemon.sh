#!/bin/bash

# CineCast TTS服务开机自启动脚本
# 作者: Yuan Liang
# 日期: 2026-02-28

# 设置工作目录
WORK_DIR="/Users/yuanliang/superstar/superstar3.1/projects/cinecast"
LOG_FILE="/Users/yuanliang/superstar/superstar3.1/projects/cinecast/logs/cinecast_daemon.log"
PID_FILE="/Users/yuanliang/superstar/superstar3.1/projects/cinecast/cinecast.pid"

# 创建日志目录
mkdir -p "$(dirname "$LOG_FILE")"

# 日志函数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# 检查服务是否已在运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# 启动CineCast服务
start_cinecast() {
    log "开始启动CineCast TTS服务"
    
    cd "$WORK_DIR"
    
    # 激活虚拟环境
    source cinecast_venv/bin/activate
    
    # 启动服务并记录PID
    nohup python stream_api_production.py > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"
    
    log "CineCast服务已启动，PID: $pid"
    
    # 等待服务启动
    sleep 5
    
    # 检查服务状态
    if curl -s http://localhost:8888/health > /dev/null 2>&1; then
        log "CineCast服务启动成功"
        return 0
    else
        log "CineCast服务启动失败"
        return 1
    fi
}

# 启动Cloudflare Tunnel
start_cloudflared() {
    log "开始启动Cloudflare Tunnel"
    
    # 检查cloudflared是否已在运行
    if pgrep -f "cloudflared tunnel run mac-mini-tunnel" > /dev/null 2>&1; then
        log "Cloudflare Tunnel已在运行"
        return 0
    fi
    
    # 启动Cloudflare Tunnel
    nohup cloudflared tunnel run mac-mini-tunnel > "/Users/yuanliang/superstar/superstar3.1/projects/cinecast/logs/cloudflared.log" 2>&1 &
    local cf_pid=$!
    
    log "Cloudflare Tunnel已启动，PID: $cf_pid"
    
    # 等待隧道建立
    sleep 10
    
    # 检查隧道状态
    if curl -s https://qwentts.damingxing.vip/health > /dev/null 2>&1; then
        log "Cloudflare Tunnel连接成功"
        return 0
    else
        log "Cloudflare Tunnel连接失败"
        return 1
    fi
}

# 停止服务
stop_services() {
    log "停止CineCast服务"
    
    # 停止CineCast
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid"
            rm -f "$PID_FILE"
            log "CineCast服务已停止"
        fi
    fi
    
    # 停止Cloudflare Tunnel
    pkill -f "cloudflared tunnel run mac-mini-tunnel"
    log "Cloudflare Tunnel已停止"
}

# 主程序
case "$1" in
    start)
        log "=== 启动CineCast守护进程 ==="
        
        if is_running; then
            log "CineCast服务已在运行"
            exit 0
        fi
        
        # 启动服务
        if start_cinecast && start_cloudflared; then
            log "所有服务启动成功"
            exit 0
        else
            log "服务启动失败"
            exit 1
        fi
        ;;
    
    stop)
        log "=== 停止CineCast守护进程 ==="
        stop_services
        ;;
    
    restart)
        log "=== 重启CineCast守护进程 ==="
        stop_services
        sleep 3
        if start_cinecast && start_cloudflared; then
            log "服务重启成功"
            exit 0
        else
            log "服务重启失败"
            exit 1
        fi
        ;;
    
    status)
        if is_running; then
            echo "CineCast服务: 运行中 (PID: $(cat "$PID_FILE"))"
        else
            echo "CineCast服务: 已停止"
        fi
        
        if pgrep -f "cloudflared tunnel run mac-mini-tunnel" > /dev/null 2>&1; then
            echo "Cloudflare Tunnel: 运行中"
        else
            echo "Cloudflare Tunnel: 已停止"
        fi
        
        # 检查公网访问
        if curl -s https://qwentts.damingxing.vip/health > /dev/null 2>&1; then
            echo "公网访问: 正常"
        else
            echo "公网访问: 异常"
        fi
        ;;
    
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac