#!/bin/bash
# Multi-Model Orchestration - Model Setup Script
# Pulls recommended models for the orchestration

echo "🎭 Multi-Model Orchestration - Model Setup"
echo "=============================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo "Please install from: https://ollama.ai"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollama is not running!"
    echo "Start it with: ollama serve"
    exit 1
fi

echo "✅ Ollama is running"
echo ""

# Array of recommended models
declare -a RECOMMENDED_MODELS=(
    "llama3.1:8b"
    "mistral:7b"
    "phi3:medium"
    "gemma2:9b"
    "qwen2:7b"
)

# Array of optional advanced models
declare -a ADVANCED_MODELS=(
    "llama3.1:70b"
    "mixtral:8x7b"
    "codellama:13b"
)

echo "📦 Recommended Models (for basic council):"
for model in "${RECOMMENDED_MODELS[@]}"; do
    echo "  • $model"
done
echo ""

echo "🌟 Advanced Models (optional, require more resources):"
for model in "${ADVANCED_MODELS[@]}"; do
    echo "  • $model"
done
echo ""

# Ask user what to install
read -p "Install recommended models? (y/n): " install_recommended
echo ""

if [ "$install_recommended" = "y" ] || [ "$install_recommended" = "Y" ]; then
    echo "📥 Pulling recommended models..."
    echo ""
    
    for model in "${RECOMMENDED_MODELS[@]}"; do
        echo "⏳ Pulling $model..."
        ollama pull $model
        
        if [ $? -eq 0 ]; then
            echo "✅ $model installed successfully"
        else
            echo "❌ Failed to install $model"
        fi
        echo ""
    done
fi

# Ask about advanced models
read -p "Install advanced models? (requires significant disk space) (y/n): " install_advanced
echo ""

if [ "$install_advanced" = "y" ] || [ "$install_advanced" = "Y" ]; then
    echo "📥 Pulling advanced models (this may take a while)..."
    echo ""
    
    for model in "${ADVANCED_MODELS[@]}"; do
        echo "⏳ Pulling $model..."
        ollama pull $model
        
        if [ $? -eq 0 ]; then
            echo "✅ $model installed successfully"
        else
            echo "❌ Failed to install $model"
        fi
        echo ""
    done
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Installed models:"
ollama list
echo ""
echo "🚀 You can now start the council with:"
echo "   python main.py"
