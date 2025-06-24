#!/bin/bash

echo "🚀 Walker Advanced Q-Learning System Startup"
echo "============================================="

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Set MLflow configuration
export MLFLOW_TRACKING_URI=http://localhost:5002

# Check system status
echo "🔍 System Status Check:"
echo "  ✅ Virtual environment activated"
echo "  ✅ MLflow URI: $MLFLOW_TRACKING_URI"

# Test core imports
echo "  🧪 Testing core imports..."
python3 -c "from src.agents.learning_manager import LearningManager, LearningApproach; print('    ✅ Learning Manager')" 2>/dev/null || echo "    ❌ Learning Manager"
python3 -c "from src.agents.ecosystem_interface import EcosystemInterface; print('    ✅ Ecosystem Interface')" 2>/dev/null || echo "    ❌ Ecosystem Interface"
python3 -c "import torch; print('    ✅ PyTorch (Deep Q-Learning available)')" 2>/dev/null || echo "    ⚠️  PyTorch not available (Deep Q-Learning disabled)"

echo ""
echo "🎯 Available Learning Approaches:"
echo "  🔤 Basic Q-Learning - Simple tabular (144 states)"
echo "  ⚡ Enhanced Q-Learning - Advanced tabular with experience replay"  
echo "  🍃 Survival Q-Learning - Ecosystem-aware (40,960 states)"
echo "  🧠 Deep Q-Learning - Neural network with CPU/GPU support"

echo ""
echo "🎮 Usage Options:"
echo "  1. Run full test suite:        python test_flexible_learning.py"
echo "  2. Test GPU deep learning:     python test_gpu_deep_learning.py"
echo "  3. Start training environment: python train_robots_web_visual.py"
echo "  4. Access web interface:       http://localhost:8080"

echo ""
echo "📚 Documentation: FLEXIBLE_LEARNING_GUIDE.md"
echo ""

# Offer to start training
read -p "🚀 Start training environment now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎯 Starting advanced Q-learning training environment..."
    python train_robots_web_visual.py
else
    echo "✅ System ready! Use the commands above to get started."
fi 