# 🤖 Walker Robot Training

Watch robots learn to crawl in real-time! This system combines physics simulation, Q-learning, and evolutionary algorithms to train crawling robots.

## 🚀 Quick Start

### 1. Test the System First
```bash
# Make sure you're in the walker_python directory
cd walker_python

# Test that everything works
python3 test_training.py
```

### 2. Run the Full Training Visualization
```bash
# Start the training with real-time visualization
python3 train_robots.py
```

## 🎮 Controls

Once the training window opens:

- **SPACE**: Pause/Resume training
- **R**: Reset the entire population
- **D**: Toggle debug information
- **1**: Slow training speed
- **2**: Normal training speed  
- **3**: Fast training speed
- **ESC**: Quit training

## 📊 What You'll See

### Main Window
- **8 robots** arranged in a row
- **Real-time physics simulation** with Pymunk
- **Green dots** = robots moving forward
- **Red dots** = robots moving backward
- **Arm angles** displayed for each robot

### UI Panel (Right Side)
- **Generation counter** - tracks evolution progress
- **Episode counter** - tracks training episodes
- **Best fitness** - best performing robot's score
- **Average fitness** - population average
- **Fitness history graph** - learning progress over time

## 🔬 How It Works

### 1. Physics Simulation
- Each robot is a **crawling crate** with two articulated arms
- **Pymunk physics engine** provides realistic 2D physics
- Robots can only move by **actuating their arms** (no wheels!)

### 2. Learning Process
- Robots start with **random actions** (exploration)
- **Fitness is measured** by forward progress
- **Evolution occurs** every 10 episodes
- **Best robots survive** and pass on their "genes"

### 3. Evolution
- **Population size**: 8 robots
- **Elite preservation**: 2 best robots survive
- **Mutation**: Random changes to robot parameters
- **Crossover**: Combining traits from parent robots

## 🧪 Testing Commands

### Basic Functionality Test
```bash
python3 test_training.py
```
This tests:
- ✅ Agent creation and physics
- ✅ Basic movement and fitness
- ✅ Population management
- ✅ Training system setup

### Individual Component Tests
```bash
# Test physics and agents
python3 -m pytest tests/test_crawling_crate_integration.py -v

# Test evolution system
python3 -m pytest tests/test_population.py -v

# Test Q-learning
python3 -m pytest tests/test_agents.py -v
```

### Simple Visualization
```bash
# Watch a single robot (if you have pygame)
python3 visualize_crate.py
```

## 📈 Expected Behavior

### Initial Phase (Episodes 1-50)
- Robots will move **randomly** and mostly **backward**
- **Low fitness scores** (0.0 - 0.5)
- Some robots may **fall over** or **flip**

### Learning Phase (Episodes 50-200)
- Robots start to **stabilize** and stay upright
- **Fitness improves** gradually
- Some robots discover **basic crawling patterns**

### Advanced Phase (Episodes 200+)
- Robots develop **coordinated arm movements**
- **Significant forward progress** (fitness > 2.0)
- **Consistent crawling behavior**

## 🔧 Troubleshooting

### "Command 'python' not found"
```bash
# Use python3 instead
python3 train_robots.py
```

### "pygame not found"
```bash
# Install pygame
pip install pygame
```

### "pymunk not found"
```bash
# Install pymunk
pip install pymunk
```

### Window doesn't open
```bash
# Check if you have a display
echo $DISPLAY

# For remote SSH, you might need X11 forwarding
ssh -X username@hostname
```

### Performance issues
```bash
# Reduce population size in train_robots.py
# Change population_size=8 to population_size=4
```

## 🎯 Success Criteria

A successful training run should show:

1. **Robots staying upright** (not falling over)
2. **Gradual fitness improvement** over time
3. **Forward movement** (green dots appearing)
4. **Coordinated arm movements** (not random flailing)
5. **Fitness scores > 1.0** after 100+ episodes

## 🔬 Advanced Usage

### Custom Training Parameters
Edit `train_robots.py` to modify:
- **Population size**: Change `population_size=8`
- **Training speed**: Modify `training_speed`
- **Evolution frequency**: Change `if self.episode % 10 == 0`
- **Fitness evaluation**: Modify `evaluate_agent()` function

### Save/Load Trained Robots
```python
# Save best robot (future feature)
best_robot.save("best_crawler.pkl")

# Load trained robot (future feature)
robot = CrawlingCrate.load("best_crawler.pkl")
```

## 📚 Technical Details

### Robot Architecture
- **Body**: 4x2 crate with high friction
- **Arms**: Two articulated segments with motors
- **Sensors**: Position, velocity, angle, contact
- **Actuators**: Two motor joints (left/right arms)

### Learning Algorithm
- **Q-learning** with discrete state/action spaces
- **State space**: 7 dimensions (position, velocity, angles, contacts)
- **Action space**: 8 discrete arm movement patterns
- **Reward function**: Forward progress + stability bonuses

### Evolution Strategy
- **Tournament selection** for parent selection
- **Uniform crossover** for Q-table combination
- **Gaussian mutation** for parameter changes
- **Elitism** to preserve best performers

## 🎉 Have Fun!

Watch your robots evolve from clumsy boxes to skilled crawlers! The training process demonstrates:

- **Emergent behavior** from simple rules
- **Evolutionary optimization** in action
- **Reinforcement learning** through trial and error
- **Physics-based robotics** simulation

Remember: **The untrained robots start with random behavior** - that's normal! Watch them learn and improve over time. 