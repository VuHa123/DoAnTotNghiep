#!/bin/bash

# Docker Manager Script cho Mental Health Chatbot
# Giúp tránh Docker images bị xóa khi mở Docker Desktop

PROJECT_NAME="mental-health-chatbot"
BACKEND_IMAGE="${PROJECT_NAME}-backend:latest"
FRONTEND_IMAGE="${PROJECT_NAME}-frontend:latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker không chạy. Vui lòng khởi động Docker Desktop trước."
        exit 1
    fi
    print_status "Docker đang chạy"
}

# Function to build images with proper tagging
build_images() {
    print_header "BUILDING DOCKER IMAGES"
    check_docker
    
    print_status "Building backend image..."
    docker build -t $BACKEND_IMAGE .
    
    if [ $? -eq 0 ]; then
        print_status "✅ Backend image built successfully"
    else
        print_error "❌ Failed to build backend image"
        exit 1
    fi
    
    print_status "Building frontend image..."
    docker build -f Dockerfile.frontend -t $FRONTEND_IMAGE .
    
    if [ $? -eq 0 ]; then
        print_status "✅ Frontend image built successfully"
    else
        print_error "❌ Failed to build frontend image"
        exit 1
    fi
    
    print_status "All images built successfully!"
}

# Function to save images to tar files
backup_images() {
    print_header "BACKING UP DOCKER IMAGES"
    check_docker
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    print_status "Saving backend image..."
    docker save $BACKEND_IMAGE -o "backup_${PROJECT_NAME}_backend_${timestamp}.tar"
    
    print_status "Saving frontend image..."
    docker save $FRONTEND_IMAGE -o "backup_${PROJECT_NAME}_frontend_${timestamp}.tar"
    
    print_status "✅ Images backed up successfully!"
    ls -la backup_${PROJECT_NAME}_*.tar
}

# Function to load images from tar files
restore_images() {
    print_header "RESTORING DOCKER IMAGES"
    check_docker
    
    backup_files=$(find . -name "backup_${PROJECT_NAME}_*.tar" -type f)
    
    if [ -z "$backup_files" ]; then
        print_warning "No backup files found!"
        return
    fi
    
    for file in $backup_files; do
        print_status "Loading image from $file..."
        docker load -i "$file"
    done
    
    print_status "✅ Images restored successfully!"
}

# Function to start services
start_services() {
    print_header "STARTING SERVICES"
    check_docker
    
    print_status "Starting services with docker-compose..."
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        print_status "✅ Services started successfully!"
        print_status "Backend: http://localhost:8000"
        print_status "Frontend: http://localhost:7860"
    else
        print_error "❌ Failed to start services"
        exit 1
    fi
}

# Function to stop services
stop_services() {
    print_header "STOPPING SERVICES"
    check_docker
    
    print_status "Stopping services..."
    docker-compose down
    
    print_status "✅ Services stopped successfully!"
}

# Function to show status
show_status() {
    print_header "DOCKER STATUS"
    check_docker
    
    echo ""
    print_status "Docker Images:"
    docker images | grep $PROJECT_NAME || echo "No project images found"
    
    echo ""
    print_status "Docker Containers:"
    docker ps -a | grep $PROJECT_NAME || echo "No project containers found"
    
    echo ""
    print_status "Backup Files:"
    find . -name "backup_${PROJECT_NAME}_*.tar" -type f 2>/dev/null || echo "No backup files found"
}

# Function to clean up
cleanup() {
    print_header "CLEANING UP"
    check_docker
    
    print_warning "This will remove all containers and images for this project!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping and removing containers..."
        docker-compose down --rmi all --volumes --remove-orphans
        
        print_status "Removing images..."
        docker rmi $BACKEND_IMAGE $FRONTEND_IMAGE 2>/dev/null || true
        
        print_status "✅ Cleanup completed!"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo "🔧 Docker Manager - Mental Health Chatbot"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     - Build Docker images with proper tagging"
    echo "  backup    - Save images to tar files"
    echo "  restore   - Load images from tar files"
    echo "  start     - Start services with docker-compose"
    echo "  stop      - Stop services"
    echo "  status    - Show current status"
    echo "  cleanup   - Remove all containers and images"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build    # Build images"
    echo "  $0 backup   # Backup images"
    echo "  $0 start    # Start services"
    echo "  $0 status   # Show status"
}

# Main script logic
case "${1:-help}" in
    build)
        build_images
        ;;
    backup)
        backup_images
        ;;
    restore)
        restore_images
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 