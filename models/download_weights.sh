#!/bin/bash
# ==============================================================================
# Hugging Face Model Weights Auto-Download Script
# Project: [Your Project Name] - AI-Driven De Novo Antibody Design
# ==============================================================================

# 👇 [1] Insert your Hugging Face repository ID here
# Format: "username/repo_name", e.g., "BioAI-Lab/RFdiffusion-Antibody-LoRA"
HF_REPO_ID="BowenDuGroup/merged_base_ckpt_finetuned.pt"

# 👇 [2] Insert your actual checkpoint file name
FILE_NAME="merged_base_ckpt_finetuned.pt"

# 👇 [3] Target directory (Based on standard structure, save to code/)
TARGET_DIR=""

# ==============================================================================

# Construct the official Hugging Face direct download URL
DOWNLOAD_URL="https://huggingface.co/${HF_REPO_ID}/resolve/main/${FILE_NAME}"

echo ""
echo "========================================================"
echo "🚀 Preparing to fetch model weights..."
echo "========================================================"
echo "🎯 Target Repo: https://huggingface.co/${HF_REPO_ID}"
echo "📦 Target File: ${FILE_NAME}"
echo "📂 Output Dir : ./${TARGET_DIR}/"
echo "========================================================"
echo ""

# Ensure the target directory exists
mkdir -p ${TARGET_DIR}

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo "❌ Error: 'wget' is not installed. Please install it first."
    exit 1
fi

echo "⏳ Downloading (Resume capability enabled)..."
# Core download command: -c enables resume, -O specifies output path
wget -c -O "${TARGET_DIR}/${FILE_NAME}" "${DOWNLOAD_URL}"

# Verify download status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "✅ Download completed successfully!"
    echo "📍 Weights saved to: ./${TARGET_DIR}/${FILE_NAME}"
    echo "💡 You can now run: cd code && python run_design.py"
    echo "========================================================"
else
    echo ""
    echo "========================================================"
    echo "❌ Download failed!"
    echo "Please check:"
    echo "1. Your internet connection to Hugging Face."
    echo "2. Whether HF_REPO_ID and FILE_NAME are correct."
    echo "3. If the repository is Private (Requires HF token)."
    echo "========================================================"
    # Remove potentially corrupted incomplete files
    rm -f "${TARGET_DIR}/${FILE_NAME}"
    exit 1
fi
