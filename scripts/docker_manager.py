#!/usr/bin/env python3
"""
Docker Manager Script
Quản lý Docker images và containers để tránh bị xóa
"""

import subprocess
import json
import os
import sys
from datetime import datetime

class DockerManager:
    def __init__(self):
        self.project_name = "mental-health-chatbot"
        
    def run_command(self, command, capture_output=True):
        """Chạy lệnh shell và trả về kết quả"""
        try:
            result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
            return result
        except Exception as e:
            print(f"Lỗi khi chạy lệnh: {e}")
            return None
    
    def get_images(self):
        """Lấy danh sách Docker images"""
        result = self.run_command("docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}'")
        if result and result.returncode == 0:
            return result.stdout
        return None
    
    def save_image(self, image_name, tag="latest"):
        """Lưu Docker image thành file tar"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_name}_{tag}_{timestamp}.tar"
        
        print(f"Đang lưu image {image_name}:{tag} thành {filename}...")
        result = self.run_command(f"docker save {image_name}:{tag} -o {filename}")
        
        if result and result.returncode == 0:
            print(f"✅ Đã lưu image thành công: {filename}")
            return filename
        else:
            print(f"❌ Lỗi khi lưu image: {image_name}:{tag}")
            return None
    
    def load_image(self, filename):
        """Load Docker image từ file tar"""
        print(f"Đang load image từ {filename}...")
        result = self.run_command(f"docker load -i {filename}")
        
        if result and result.returncode == 0:
            print(f"✅ Đã load image thành công từ {filename}")
            return True
        else:
            print(f"❌ Lỗi khi load image từ {filename}")
            return False
    
    def tag_image(self, image_id, repository, tag="latest"):
        """Tag Docker image"""
        result = self.run_command(f"docker tag {image_id} {repository}:{tag}")
        if result and result.returncode == 0:
            print(f"✅ Đã tag image {image_id} thành {repository}:{tag}")
            return True
        else:
            print(f"❌ Lỗi khi tag image")
            return False
    
    def build_with_cache(self):
        """Build images với cache và tag rõ ràng"""
        print("🔨 Đang build Docker images với cache...")
        
        # Build backend
        backend_result = self.run_command(
            f"docker build -t {self.project_name}-backend:latest ."
        )
        
        # Build frontend
        frontend_result = self.run_command(
            f"docker build -f Dockerfile.frontend -t {self.project_name}-frontend:latest ."
        )
        
        if backend_result and backend_result.returncode == 0:
            print("✅ Backend image đã được build thành công")
        else:
            print("❌ Lỗi khi build backend image")
            
        if frontend_result and frontend_result.returncode == 0:
            print("✅ Frontend image đã được build thành công")
        else:
            print("❌ Lỗi khi build frontend image")
    
    def backup_images(self):
        """Backup tất cả images của project"""
        print("💾 Đang backup Docker images...")
        
        images_to_backup = [
            f"{self.project_name}-backend:latest",
            f"{self.project_name}-frontend:latest"
        ]
        
        backup_files = []
        for image in images_to_backup:
            backup_file = self.save_image(image.split(':')[0], image.split(':')[1])
            if backup_file:
                backup_files.append(backup_file)
        
        if backup_files:
            print(f"✅ Đã backup {len(backup_files)} images")
            return backup_files
        else:
            print("❌ Không có images nào được backup")
            return []
    
    def restore_images(self, backup_dir="."):
        """Restore images từ backup files"""
        print("🔄 Đang restore Docker images...")
        
        # Tìm tất cả file .tar trong thư mục hiện tại
        result = self.run_command(f"find {backup_dir} -name '*.tar' -type f")
        if result and result.returncode == 0:
            tar_files = result.stdout.strip().split('\n')
            
            for tar_file in tar_files:
                if tar_file:
                    self.load_image(tar_file)
        
        print("✅ Hoàn thành restore images")
    
    def list_containers(self):
        """Liệt kê tất cả containers"""
        result = self.run_command("docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'")
        if result and result.returncode == 0:
            return result.stdout
        return None
    
    def stop_and_remove_containers(self):
        """Dừng và xóa tất cả containers của project"""
        print("🛑 Đang dừng và xóa containers...")
        
        # Dừng containers
        self.run_command(f"docker-compose down")
        
        # Xóa containers cũ
        result = self.run_command("docker ps -a -q")
        if result and result.returncode == 0 and result.stdout.strip():
            self.run_command("docker rm -f $(docker ps -a -q)")
        
        print("✅ Đã dừng và xóa containers")
    
    def start_services(self):
        """Khởi động services với docker-compose"""
        print("🚀 Đang khởi động services...")
        
        # Build và start
        result = self.run_command("docker-compose up --build -d")
        
        if result and result.returncode == 0:
            print("✅ Services đã được khởi động thành công")
            print("📊 Backend: http://localhost:8000")
            print("📊 Frontend: http://localhost:7860")
        else:
            print("❌ Lỗi khi khởi động services")
    
    def show_status(self):
        """Hiển thị trạng thái hiện tại"""
        print("📊 === TRẠNG THÁI DOCKER ===")
        
        print("\n🐳 Docker Images:")
        images = self.get_images()
        if images:
            print(images)
        
        print("\n📦 Docker Containers:")
        containers = self.list_containers()
        if containers:
            print(containers)
        
        print("\n💾 Backup Files:")
        result = self.run_command("find . -name '*.tar' -type f")
        if result and result.returncode == 0 and result.stdout.strip():
            print(result.stdout)
        else:
            print("Không có file backup nào")

def main():
    manager = DockerManager()
    
    if len(sys.argv) < 2:
        print("""
🔧 Docker Manager - Mental Health Chatbot

Cách sử dụng:
  python scripts/docker_manager.py [command]

Commands:
  build     - Build images với cache
  backup    - Backup tất cả images
  restore   - Restore images từ backup
  start     - Khởi động services
  stop      - Dừng và xóa containers
  status    - Hiển thị trạng thái
  list      - Liệt kê images và containers
        """)
        return
    
    command = sys.argv[1]
    
    if command == "build":
        manager.build_with_cache()
    elif command == "backup":
        manager.backup_images()
    elif command == "restore":
        manager.restore_images()
    elif command == "start":
        manager.start_services()
    elif command == "stop":
        manager.stop_and_remove_containers()
    elif command == "status":
        manager.show_status()
    elif command == "list":
        print("🐳 Docker Images:")
        images = manager.get_images()
        if images:
            print(images)
        
        print("\n📦 Docker Containers:")
        containers = manager.list_containers()
        if containers:
            print(containers)
    else:
        print(f"❌ Command không hợp lệ: {command}")

if __name__ == "__main__":
    main() 