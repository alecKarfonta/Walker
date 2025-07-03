# Robot Size Mutations During Respawning

## Overview

The robot size mutation system allows for gradual evolution of robot body part dimensions during respawning while preserving their learned Q-network weights. This provides the benefits of evolutionary adaptation without losing valuable training progress.

## Key Features

- **🧬 Size-Only Mutations**: Only body part dimensions change, structural features remain intact
- **🧠 Q-Network Preservation**: Neural network weights and learning progress are maintained
- **⚖️ Bounded Variations**: All mutations are constrained to reasonable biological ranges
- **🎲 Probabilistic Application**: Each parameter has a chance to mutate, creating diversity
- **🔄 Respawn Integration**: Automatically applied during robot respawning events

## Implementation

### Core Mutation Method

The heart of the system is the `mutate_sizes_only()` method in `PhysicalParameters`:

```python
def mutate_sizes_only(self, mutation_rate: float = 0.15) -> 'PhysicalParameters':
    """
    Create a mutated copy with only body part size changes for respawning.
    Preserves all complex structural features while allowing size variations.
    """
    mutated = deepcopy(self)
    
    # Body size mutations (±20% variation)
    if random.random() < mutation_rate:
        mutated.body_width = self._mutate_bounded(self.body_width, 0.2, 0.8, 2.5)
    
    # Overall scale mutation (affects everything proportionally)
    if random.random() < mutation_rate:
        mutated.overall_scale = self._mutate_bounded(self.overall_scale, 0.15, 0.7, 1.5)
    
    # Additional size parameters...
    return mutated.validate_and_repair()
```

### Mutable Parameters

The following robot dimensions can be mutated during respawning:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `body_width` | 0.8m - 2.5m | Main body width |
| `body_height` | 0.4m - 1.2m | Main body height |
| `overall_scale` | 0.7x - 1.5x | Proportional scaling factor |
| `arm_length` | 0.6m - 1.8m | Upper arm segment length |
| `arm_width` | 0.12m - 0.35m | Upper arm segment width |
| `wrist_length` | 0.6m - 1.8m | Lower arm segment length |
| `wrist_width` | 0.12m - 0.35m | Lower arm segment width |
| `wheel_radius` | 0.3m - 0.8m | Wheel/foot radius |
| `leg_spread` | 1.2m - 3.0m | Distance between wheels |

### Preserved Features

The following structural features remain unchanged:

- Number of arms/limbs
- Number of segments per limb
- Joint configuration and types
- Body shape (rectangle, oval, etc.)
- Locomotion type (crawler, walker, etc.)
- Specialized features (tails, modules, etc.)

## Integration Points

### Robot Memory Pool

The `RobotMemoryPool` applies size mutations during robot reuse:

```python
def acquire_robot(self, position, physical_params=None, apply_size_mutations=True):
    if self.available_robots:
        robot = self.available_robots.popleft()
        self._reset_robot(robot, position, physical_params, apply_size_mutations)
        # Q-network weights preserved automatically
```

### Training Environment

The main training loop uses size mutations for replacement agents:

```python
def _create_replacement_agent(self):
    if self.robot_memory_pool:
        new_agent = self.robot_memory_pool.acquire_robot(
            position=spawn_position,
            physical_params=random_params,
            apply_size_mutations=True  # Enable mutations
        )
```

## Mutation Algorithm

### Bounded Mutation Function

```python
def _mutate_bounded(self, value: float, mutation_strength: float, 
                   min_val: float, max_val: float) -> float:
    """
    Mutate a value with Gaussian-like noise, keeping it within bounds.
    """
    # Bidirectional mutation (can increase or decrease)
    sign = 1.0 if random.random() > 0.5 else -1.0
    mutation_magnitude = mutation_strength * value * random.random()
    mutated_value = value + sign * mutation_magnitude
    
    # Clamp to valid range
    return np.clip(mutated_value, min_val, max_val)
```

### Mutation Parameters

- **Mutation Rate**: 15% chance per parameter (configurable)
- **Mutation Strength**: 15-20% of current value (parameter-specific)
- **Direction**: Bidirectional (can grow or shrink)
- **Distribution**: Uniform random within strength bounds

## Benefits

### Evolutionary Diversity

- Robots gradually explore different body proportions
- Natural selection favors effective size combinations
- Population maintains morphological diversity over time

### Learning Preservation

- Valuable Q-network training is never lost
- Robots can build upon previous learning experiences
- Faster convergence than full resets

### Biological Realism

- Mimics natural growth variations in offspring
- Size changes are gradual and bounded
- Maintains functional robot designs

## Example Usage

### Direct Mutation

```python
from src.agents.physical_parameters import PhysicalParameters

# Create base parameters
params = PhysicalParameters()
params.body_width = 1.5
params.overall_scale = 1.0

# Apply size mutations
mutated_params = params.mutate_sizes_only(mutation_rate=0.15)

print(f"Original: {params.body_width:.2f}m × {params.overall_scale:.2f}x")
print(f"Mutated:  {mutated_params.body_width:.2f}m × {mutated_params.overall_scale:.2f}x")
```

### Respawning with Mutations

```python
# Robot dies and needs replacement
dead_robot = some_dead_robot
memory_pool.return_robot(dead_robot)

# Respawn with size mutations applied
new_robot = memory_pool.acquire_robot(
    position=(x, y),
    apply_size_mutations=True
)

# Q-network weights from dead_robot are preserved in new_robot
# Body dimensions have slight variations for diversity
```

## Configuration

### Mutation Rate Tuning

- **Low (0.05-0.10)**: Conservative size changes, stable populations
- **Medium (0.10-0.20)**: Balanced exploration and stability  
- **High (0.20-0.30)**: Aggressive size exploration, more diversity

### Parameter-Specific Tuning

```python
# Fine-tune individual parameter mutation strengths
mutated.body_width = self._mutate_bounded(
    self.body_width, 
    0.15,  # 15% mutation strength
    0.8,   # Minimum width
    2.5    # Maximum width
)
```

## Testing

Run the demonstration:

```bash
python examples/size_mutation_demo.py
```

Expected output shows:
- Original robot parameters
- Size mutations applied during respawning
- Q-network weight preservation verification
- Multiple mutation iterations with size tracking

## Future Enhancements

### Adaptive Mutation Rates

- Increase mutation rate when population diversity is low
- Decrease when robots are performing well
- Environmental pressure-based adjustments

### Correlated Mutations

- Link related parameters (e.g., arm length and width)
- Maintain proportional relationships
- Biologically-inspired constraint satisfaction

### Performance-Based Mutations

- Bias mutations toward beneficial size combinations
- Learn which dimensions correlate with success
- Implement directed evolution based on fitness

## Summary

The robot size mutation system provides a sophisticated yet simple approach to evolutionary robotics that:

1. **Preserves Learning**: Q-network weights are never lost
2. **Enables Evolution**: Body dimensions gradually adapt over time  
3. **Maintains Stability**: All changes are bounded and validated
4. **Promotes Diversity**: Each respawn introduces subtle variations
5. **Scales Efficiently**: Low computational overhead during respawning

This system bridges the gap between learning and evolution, allowing robots to both improve their decision-making through experience and adapt their physical form through selection pressure. 