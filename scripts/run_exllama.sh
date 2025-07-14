#!/bin/bash

# ExLlama GPTQ Workflow Script
# Sử dụng: ./run_exllama.sh [setup|convert|test|interactive|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Virtual environment not detected. Please activate your virtual environment first."
        print_status "Example: source .venv/bin/activate"
        exit 1
    fi
    print_success "Virtual environment detected: $VIRTUAL_ENV"
}

# Check CUDA availability
check_cuda() {
    print_status "Checking CUDA availability..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || {
        print_error "PyTorch not installed or CUDA not available"
        exit 1
    }
}

# Setup environment
setup_environment() {
    print_status "Setting up ExLlama environment..."
    
    # Update requirements
    python scripts/update_requirements.py
    
    # Install ExLlama
    python scripts/setup_exllama.py
    
    print_success "Environment setup completed!"
}

# Convert model
convert_model() {
    print_status "Converting model to GPTQ format..."
    
    # Check if LoRA model exists
    if [[ ! -d "models/weights/chatbot_finetuned_nf4" ]]; then
        print_error "LoRA model not found at models/weights/chatbot_finetuned_nf4"
        print_status "Please ensure you have completed fine-tuning first"
        exit 1
    fi
    
    # Convert model
    python scripts/convert_to_gptq.py \
        --base_model "meta-llama/Llama-3.2-1B-Instruct" \
        --lora_path "models/weights/chatbot_finetuned_nf4" \
        --output_path "models/weights/chatbot_gptq" \
        --bits 4 \
        --group_size 128
    
    print_success "Model conversion completed!"
}

# Test inference
test_inference() {
    print_status "Testing inference..."
    
    # Check if GPTQ model exists
    if [[ ! -d "models/weights/chatbot_gptq" ]]; then
        print_error "GPTQ model not found at models/weights/chatbot_gptq"
        print_status "Please run convert step first: ./run_exllama.sh convert"
        exit 1
    fi
    
    # Test inference
    python scripts/exllama_inference.py \
        --model_path "models/weights/chatbot_gptq" \
        --test
    
    print_success "Inference test completed!"
}

# Interactive chat
interactive_chat() {
    print_status "Starting interactive chat..."
    
    # Check if GPTQ model exists
    if [[ ! -d "models/weights/chatbot_gptq" ]]; then
        print_error "GPTQ model not found at models/weights/chatbot_gptq"
        print_status "Please run convert step first: ./run_exllama.sh convert"
        exit 1
    fi
    
    # Start interactive chat
    python scripts/exllama_inference.py \
        --model_path "models/weights/chatbot_gptq" \
        --interactive
}

# Run all steps
run_all() {
    print_status "Running complete ExLlama GPTQ workflow..."
    
    setup_environment
    convert_model
    test_inference
    
    print_success "All steps completed successfully!"
    print_status "You can now run interactive chat with: ./run_exllama.sh interactive"
}

# Show usage
show_usage() {
    echo "ExLlama GPTQ Workflow Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup       - Setup ExLlama environment and dependencies"
    echo "  convert     - Convert fine-tuned LoRA model to GPTQ format"
    echo "  test        - Test inference with GPTQ model"
    echo "  interactive - Start interactive chat"
    echo "  all         - Run all steps (setup + convert + test)"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 convert"
    echo "  $0 test"
    echo "  $0 interactive"
    echo "  $0 all"
    echo ""
    echo "Note: Make sure your virtual environment is activated before running this script."
}

# Main script
main() {
    # Check virtual environment
    check_venv
    
    # Check CUDA
    check_cuda
    
    # Parse command line arguments
    case "${1:-}" in
        "setup")
            setup_environment
            ;;
        "convert")
            convert_model
            ;;
        "test")
            test_inference
            ;;
        "interactive")
            interactive_chat
            ;;
        "all")
            run_all
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 