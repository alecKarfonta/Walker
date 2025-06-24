# Walker Python

A Python recreation of the Walker project - a reinforcement learning environment for training physics-based robots using evolutionary algorithms.

## Features

- **Physics Simulation**: Realistic 2D physics using Pymunk (Python port of Chipmunk)
- **Reinforcement Learning**: Q-learning agents with discrete state/action spaces
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

## Development Status

### Phase 1: Core Physics Engine ✅
- [x] Physics world setup with Pymunk
- [x] Body factory for creating different body types
- [x] Contact handler for collision detection
- [x] Ground terrain generation
- [x] Basic physics simulation

### Phase 2: Agent System (In Progress)
- [ ] Base agent architecture
- [ ] Q-learning implementation
- [ ] Basic robot design
- [ ] Learning algorithm testing

### Phase 3: Population Management (Planned)
- [ ] Population controller
- [ ] Evolution algorithms
- [ ] Selection strategies

### Phase 4: Rendering System (Planned)
- [ ] Basic rendering pipeline
- [ ] Camera system
- [ ] Debug visualization

### Phase 5: User Interface (Planned)
- [ ] GUI framework setup
- [ ] Control windows
- [ ] Interactive elements

## Technology Stack

- **Python 3.9+**: Main programming language
- **Pymunk**: 2D physics engine (Python port of Chipmunk)
- **Pygame**: Graphics and input handling
- **NumPy**: Numerical computations
- **Pandas**: Data handling

## License

[Add your license here] 