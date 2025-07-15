import torch
import psutil
import time
import threading
from typing import Optional

class MemoryMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Bắt đầu monitoring memory"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Memory monitoring started...")
        
    def stop_monitoring(self):
        """Dừng monitoring memory"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("🛑 Memory monitoring stopped.")
        
    def _monitor_loop(self):
        """Loop monitoring memory"""
        while self.monitoring:
            try:
                # GPU Memory
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free = total - reserved
                    
                    print(f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free / {total:.2f}GB total")
                
                # CPU Memory
                cpu_memory = psutil.virtual_memory()
                print(f"CPU: {cpu_memory.percent}% used ({cpu_memory.used/1024**3:.1f}GB / {cpu_memory.total/1024**3:.1f}GB)")
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error in memory monitoring: {e}")
                time.sleep(self.interval)
                
    def get_memory_info(self):
        """Lấy thông tin memory hiện tại"""
        info = {}
        
        if torch.cuda.is_available():
            info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
            info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info['gpu_free'] = info['gpu_total'] - info['gpu_reserved']
            
        cpu_memory = psutil.virtual_memory()
        info['cpu_percent'] = cpu_memory.percent
        info['cpu_used'] = cpu_memory.used / 1024**3
        info['cpu_total'] = cpu_memory.total / 1024**3
        
        return info

def clear_memory():
    """Clear memory và cache"""
    import gc
    
    # Clear Python garbage collector
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    print("🧹 Memory cleared!")

def optimize_memory_settings():
    """Thiết lập các environment variables để tối ưu memory"""
    import os
    
    # PyTorch memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Reduce memory fragmentation
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    
    print("⚙️ Memory optimization settings applied!")

# Sử dụng example
if __name__ == "__main__":
    # Thiết lập optimization
    optimize_memory_settings()
    
    # Tạo monitor
    monitor = MemoryMonitor(interval=2.0)
    
    # Bắt đầu monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate some work
        print("Doing some work...")
        time.sleep(10)
        
        # Clear memory
        clear_memory()
        
        # Get current memory info
        info = monitor.get_memory_info()
        print(f"Current memory info: {info}")
        
    finally:
        # Dừng monitoring
        monitor.stop_monitoring() 