# Walker Python

A Python recreation of the Walker project - a simulated environment for training populations of based robots using reinforcement learning evolutionary algorithms.

## Features

- **Physics Simulation**: Realistic 2D physics using Box2d
- **Reinforcement Learning**: Various implementations of Q-Learning
- **Evolutionary Algorithm**: Population-based evolution with genetic operators
- **Real-time Visualization**: Pygame-based rendering and debugging
- **Interactive GUI**: Parameter adjustment and simulation control

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd walker_python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the physics simulation test:
```bash
python run.py
```

## Project Structure

```
walker_python/
├── src/
│   ├── config/          # Configuration and settings
│   ├── physics/         # Physics engine (Pymunk)
│   ├── agents/          # Reinforcement learning agents
│   ├── population/      # Population management
│   ├── rendering/       # Graphics and visualization
│   ├── ui/             # User interface
│   ├── utils/          # Utility functions
│   └── assets/         # Game assets
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── run.py             # Main entry point
```
