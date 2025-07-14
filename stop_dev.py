#!/usr/bin/env python3
"""
Script để dừng tất cả development services
"""

import os
import sys
import signal
import subprocess
import logging
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_processes_by_port(port):
    """Tìm process đang sử dụng port cụ thể"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def find_processes_by_name(name_patterns):
    """Tìm process theo tên"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            for pattern in name_patterns:
                if pattern in cmdline:
                    processes.append(proc)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def stop_process(proc, reason=""):
    """Dừng process gracefully"""
    try:
        logger.info(f"🛑 Stopping {proc.name()} (PID: {proc.pid}) {reason}")
        proc.terminate()
        
        # Đợi process dừng
        try:
            proc.wait(timeout=10)
            logger.info(f"✅ {proc.name()} stopped gracefully")
            return True
        except psutil.TimeoutExpired:
            logger.warning(f"⚠️ {proc.name()} didn't stop gracefully, killing...")
            proc.kill()
            proc.wait()
            logger.info(f"✅ {proc.name()} killed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error stopping {proc.name()}: {e}")
        return False

def stop_services():
    """Dừng tất cả development services"""
    logger.info("🛑 Stopping all development services...")
    
    # Tìm và dừng processes theo port
    ports_to_check = [8000, 8001]  # API Gateway và Model Server
    
    for port in ports_to_check:
        processes = find_processes_by_port(port)
        for proc in processes:
            stop_process(proc, f"(port {port})")
    
    # Tìm và dừng processes theo tên
    name_patterns = [
        "model_server.py",
        "api_gateway/main.py",
        "uvicorn",
        "run_dev.py",
        "run_individual_services.py"
    ]
    
    processes = find_processes_by_name(name_patterns)
    for proc in processes:
        stop_process(proc, "(by name)")
    
    logger.info("✅ All development services stopped")

def check_running_services():
    """Kiểm tra services đang chạy"""
    logger.info("🔍 Checking running services...")
    
    # Kiểm tra ports
    for port in [8000, 8001]:
        processes = find_processes_by_port(port)
        if processes:
            logger.info(f"⚠️ Port {port} is in use by:")
            for proc in processes:
                logger.info(f"  - {proc.name()} (PID: {proc.pid})")
        else:
            logger.info(f"✅ Port {port} is free")
    
    # Kiểm tra processes theo tên
    name_patterns = [
        "model_server.py",
        "api_gateway/main.py",
        "uvicorn"
    ]
    
    processes = find_processes_by_name(name_patterns)
    if processes:
        logger.info("⚠️ Found development processes:")
        for proc in processes:
            logger.info(f"  - {proc.name()} (PID: {proc.pid})")
    else:
        logger.info("✅ No development processes found")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Stop development services")
    parser.add_argument("--check", action="store_true", 
                       help="Check running services without stopping")
    
    args = parser.parse_args()
    
    if args.check:
        check_running_services()
    else:
        stop_services()

if __name__ == "__main__":
    import argparse
    main() 