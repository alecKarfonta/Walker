# Walker - AI Robot Training Platform

A sophisticated reinforcement learning environment for training physics-based robots using evolutionary Q-learning algorithms. Walker provides a real-time simulation where AI agents learn to navigate, crawl, and walk through adaptive reinforcement learning techniques.

🌐 **Live Demo**: [https://mlapi.us/walker](https://mlapi.us/walker)

## 🤖 Overview

Walker is an experimental platform for developing and testing Q-Learning control policies across diverse robotic systems within a physics-based simulation environment. The platform offers real-time manipulation of learning parameters, robot characteristics, and environmental physics, enabling researchers to observe immediate effects of parameter modifications on robot behavior and learning outcomes.

### Key Features

- **🧠 Multi-Layered Q-Learning**: Progressive complexity from basic (3D) to deep learning (15D continuous)
- **⚡ Real-time Physics Simulation**: Built on Pymunk physics engine for realistic interactions
- **🎮 Interactive Web Interface**: Live training visualization and parameter adjustment
- **🧬 Evolutionary Algorithms**: Population-based optimization for accelerated learning
- **📊 Comprehensive Monitoring**: MLflow tracking, Prometheus metrics, Grafana dashboards
- **🎯 Multiple Robot Types**: Crawling crates, legged walkers, and custom configurations
- **🔧 GPU Acceleration**: NVIDIA GPU support for deep learning models

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Prerequisites**:
   - Docker (20.10+) and Docker Compose (2.0+)
   - 8GB+ RAM, 10GB+ disk space
   - NVIDIA GPU (optional, for deep learning)

2. **Deploy the complete stack**:
   ```bash
   ./start-walker-stack.sh
   ```

3. **Access the interfaces**:
   - **Main Training Interface**: http://localhost:7777
   - **MLflow Tracking**: http://localhost:5555
   - **Grafana Dashboards**: http://localhost:3333 (admin/walker123)

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run basic physics test**:
   ```bash
   python run.py
   ```

3. **Start training interface**:
   ```bash
   python train_robots_web_visual.py
   ```

## 🏗️ Architecture

### Q-Learning Systems

Walker implements multiple Q-learning approaches with increasing sophistication:

#### 1. **Basic Q-Learning (3D State Space)**
- Simple movement patterns
- Joint position and velocity tracking
- Foundation for more complex behaviors

#### 2. **Enhanced Q-Learning (7D State Space)**  
- Adaptive learning rates based on visit counts
- Exploration bonuses for under-explored actions
- Position, velocity, and orientation awareness

#### 3. **Survival Q-Learning (8D State Space)**
- Energy management and efficiency optimization
- Long-term strategy development
- Environmental adaptation

#### 4. **Deep Q-Learning (15D Continuous)**
- Neural network-based value approximation
- High-dimensional state representation
- Complex behavior emergence

### Robot Types

- **Crawling Crates**: Single-arm robots with two joints optimized for forward movement
- **Legged Walkers**: Dual-arm robots designed for complex locomotion patterns
- **Custom Configurations**: Flexible body factory for experimental designs

### Physics Engine

Built on **Pymunk** (Python port of Chipmunk Physics):
- Realistic collision detection and response
- Adjustable physics parameters (timestep, gravity, friction)
- Dynamic terrain and obstacle generation
- Real-time body manipulation and visualization

## 📊 Monitoring & Analytics

### MLflow Integration
- Experiment tracking and comparison
- Parameter logging and visualization  
- Model artifact management
- Automated metric collection

### Prometheus & Grafana
- Real-time performance metrics
- Training progress visualization
- System resource monitoring
- Custom dashboard creation

## 🎮 Interactive Controls

### Web Interface Features
- **Robot Selection**: Click to select and control individual robots
- **Parameter Adjustment**: Real-time learning rate, exploration, and physics tweaking
- **Training Control**: Pause, reset, and goal modification for selected robots
- **Visualization**: Camera following, zoom controls, and debug overlays

### Keyboard Controls
- **Arrow Keys**: Manual robot control (when learning disabled)
- **Space**: Send selected robot to home position
- **H**: Toggle motor hold/release
- **L**: Toggle learning on/off

## 📁 Project Structure

```
Walker/
├── src/                    # Source code
│   ├── agents/            # Q-learning implementations
│   ├── physics/           # Pymunk physics engine
│   ├── rendering/         # Pygame visualization
│   ├── population/        # Evolutionary algorithms
│   ├── ui/               # Web interface components
│   └── utils/            # Utility functions
├── legacy/               # Original Java/LibGDX implementation
├── experiments/          # Training data and results
├── evaluation_exports/   # Performance analysis
├── config/              # Configuration files
├── templates/           # Web interface templates
├── tests/               # Unit and integration tests
├── docker-compose.yml   # Multi-service deployment
├── Dockerfile          # Container configuration
└── requirements.txt    # Python dependencies
```

## 🔬 Research Applications

Walker is designed for researchers and developers working on:

- **Reinforcement Learning**: Testing new Q-learning variants and optimizations
- **Robotics Simulation**: Physics-based robot behavior modeling
- **Evolutionary Algorithms**: Population-based optimization strategies
- **Multi-Agent Systems**: Competitive and cooperative robot interactions
- **Parameter Optimization**: Automated hyperparameter tuning

## 🐛 Known Issues & Development Status

### Current Status: 🟡 **Functional with Ongoing Improvements**

**Working Features**:
- ✅ Core physics simulation and rendering
- ✅ Basic and Enhanced Q-learning algorithms
- ✅ Web interface and real-time visualization
- ✅ Docker deployment and monitoring stack
- ✅ Multi-robot training and evaluation

**In Development**:
- 🔄 Deep Q-learning stability improvements
- 🔄 Advanced evolutionary algorithm integration
- 🔄 Enhanced robot body types and configurations
- 🔄 Performance optimizations for large populations

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

- Q-learning algorithm improvements and variants
- New robot body designs and configurations
- Performance optimizations and GPU utilization
- Web interface enhancements and new visualizations
- Documentation and tutorial creation

## 📈 Performance

- **Training Speed**: 100-1000+ steps/second depending on configuration
- **Population Size**: Support for 50+ concurrent robots
- **GPU Acceleration**: 5-10x speedup for deep learning models
- **Memory Usage**: ~2-4GB for typical training sessions

## 🔗 Links

- **Live Demo**: [https://mlapi.us/walker](https://mlapi.us/walker)
- **Documentation**: See individual `.md` files for detailed guides
- **Original Java Implementation**: See `legacy/` directory

## 📝 License

[License information to be added]

---

**Walker** - Where AI learns to move, one step at a time. 🚶‍♂️🤖 