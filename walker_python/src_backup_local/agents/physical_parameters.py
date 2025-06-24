"""
Physical Parameters for Evolutionary Crawling Robots.

This module defines the physical characteristics that can be evolved,
inspired by the Java CrawlingCrate implementation with comprehensive mutation support.
"""

import numpy as np
import random
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class PhysicalParameters:
    """
    Comprehensive physical parameters for crawling robots.
    All parameters can be evolved through mutation and crossover.
    """
    
    # Body parameters (inspired by Java CrawlingCrate)
    body_width: float = 1.5
    body_height: float = 0.75
    body_density: float = 4.0
    body_friction: float = 0.9
    body_restitution: float = 0.1
    body_linear_damping: float = 0.05
    body_angular_damping: float = 0.05
    
    # Arm parameters (extensive like Java implementation)
    arm_length: float = 1.0  # Upper arm length
    arm_width: float = 0.2   # Upper arm width
    wrist_length: float = 1.0  # Lower arm length
    wrist_width: float = 0.2   # Lower arm width
    arm_density: float = 0.1
    arm_friction: float = 0.5
    arm_restitution: float = 0.1
    
    # Joint parameters
    arm_torque: float = 150.0
    wrist_torque: float = 150.0
    arm_speed: float = 3.0
    wrist_speed: float = 3.0
    shoulder_lower_limit: float = -np.pi/2  # -90 degrees
    shoulder_upper_limit: float = np.pi/2   # +90 degrees
    elbow_lower_limit: float = 0.0          # 0 degrees
    elbow_upper_limit: float = 3*np.pi/4    # 135 degrees
    
    # Wheel parameters
    wheel_radius: float = 0.5
    wheel_density: float = 8.0
    wheel_friction: float = 0.9
    wheel_restitution: float = 0.2
    leg_spread: float = 2.0  # Distance between wheels
    suspension: float = 0.75  # Height of wheels below body
    ride_height: float = 0.0  # Additional height offset
    
    # Learning and behavior parameters (from Java BasicAgent)
    learning_rate: float = 0.005
    min_learning_rate: float = 0.001
    max_learning_rate: float = 0.05
    epsilon: float = 0.3
    min_epsilon: float = 0.01
    max_epsilon: float = 0.6
    discount_factor: float = 0.9
    exploration_bonus: float = 0.15
    impatience: float = 0.002
    
    # Goal and reward weights (from Java implementation)
    speed_value_weight: float = 0.06
    acceleration_value_weight: float = 0.04
    position_weight: float = 0.01
    stability_weight: float = 0.03
    average_speed_value_weight: float = 0.04
    
    # Motor control parameters
    motor_torque: float = 150.0
    motor_speed: float = 3.0
    action_interval: int = 2
    learning_interval: int = 30
    
    # Advanced parameters
    precision: float = 1.0  # State discretization precision
    update_timer: float = 0.1
    mutation_rate: float = 0.01
    
    def mutate(self, mutation_rate: float = 0.1) -> 'PhysicalParameters':
        """
        Create a mutated copy of these parameters.
        Inspired by the comprehensive mutation in Java CrawlingCrate.
        
        Args:
            mutation_rate: Base mutation rate for all parameters
            
        Returns:
            New PhysicalParameters with mutations applied
        """
        mutated = deepcopy(self)
        
        # Body parameter mutations
        if random.random() < mutation_rate:
            mutated.body_width = self._mutate_bounded(
                self.body_width, 0.5, 0.8, 3.0
            )
        if random.random() < mutation_rate:
            mutated.body_height = self._mutate_bounded(
                self.body_height, 0.4, 0.3, 1.5
            )
        if random.random() < mutation_rate:
            mutated.body_density = self._mutate_bounded(
                self.body_density, 0.5, 1.0, 8.0
            )
        if random.random() < mutation_rate:
            mutated.body_friction = self._mutate_bounded(
                self.body_friction, 0.3, 0.1, 2.0
            )
        
        # Arm parameter mutations (comprehensive like Java)
        if random.random() < mutation_rate:
            mutated.arm_length = self._mutate_bounded(
                self.arm_length, 0.4, 0.5, 2.0
            )
        if random.random() < mutation_rate:
            mutated.arm_width = self._mutate_bounded(
                self.arm_width, 0.3, 0.1, 0.5
            )
        if random.random() < mutation_rate:
            mutated.wrist_length = self._mutate_bounded(
                self.wrist_length, 0.4, 0.5, 2.0
            )
        if random.random() < mutation_rate:
            mutated.wrist_width = self._mutate_bounded(
                self.wrist_width, 0.3, 0.1, 0.5
            )
        
        # Wheel and suspension mutations (like Java)
        if random.random() < mutation_rate:
            mutated.wheel_radius = self._mutate_bounded(
                self.wheel_radius, 0.3, 0.2, 1.0
            )
        if random.random() < mutation_rate:
            mutated.leg_spread = self._mutate_bounded(
                self.leg_spread, 0.4, 1.0, 4.0
            )
        if random.random() < mutation_rate:
            mutated.suspension = self._mutate_bounded(
                self.suspension, 0.2, 0.3, 1.5
            )
        
        # Motor parameter mutations
        if random.random() < mutation_rate:
            mutated.motor_torque = self._mutate_bounded(
                self.motor_torque, 0.3, 50.0, 300.0
            )
        if random.random() < mutation_rate:
            mutated.motor_speed = self._mutate_bounded(
                self.motor_speed, 0.3, 1.0, 8.0
            )
        
        # Learning parameter mutations
        if random.random() < mutation_rate:
            mutated.learning_rate = self._mutate_bounded(
                self.learning_rate, 0.4, mutated.min_learning_rate, mutated.max_learning_rate
            )
        if random.random() < mutation_rate:
            mutated.epsilon = self._mutate_bounded(
                self.epsilon, 0.4, mutated.min_epsilon, mutated.max_epsilon
            )
        if random.random() < mutation_rate:
            mutated.discount_factor = self._mutate_bounded(
                self.discount_factor, 0.2, 0.5, 0.99
            )
        
        # Reward weight mutations (from Java implementation)
        if random.random() < mutation_rate:
            mutated.speed_value_weight = self._mutate_bounded(
                self.speed_value_weight, 0.5, 0.01, 0.2
            )
        if random.random() < mutation_rate:
            mutated.acceleration_value_weight = self._mutate_bounded(
                self.acceleration_value_weight, 0.5, 0.01, 0.15
            )
        if random.random() < mutation_rate:
            mutated.stability_weight = self._mutate_bounded(
                self.stability_weight, 0.5, 0.005, 0.1
            )
        
        return mutated
    
    def _mutate_bounded(self, value: float, mutation_strength: float, 
                       min_val: float, max_val: float) -> float:
        """
        Mutate a value with Gaussian noise, keeping it within bounds.
        Uses the same approach as Java CrawlingCrate mutation.
        
        Args:
            value: Current value
            mutation_strength: Relative strength of mutation (0-1)
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Mutated value within bounds
        """
        # Use bidirectional mutation like Java implementation
        sign = 1.0 if random.random() > 0.5 else -1.0
        mutation_magnitude = mutation_strength * value * random.random()
        mutated_value = value + sign * mutation_magnitude
        
        return np.clip(mutated_value, min_val, max_val)
    
    def crossover(self, other: 'PhysicalParameters', 
                 crossover_rate: float = 0.5) -> 'PhysicalParameters':
        """
        Create offspring by crossing over with another parameter set.
        
        Args:
            other: Other parent's parameters
            crossover_rate: Probability of taking each parameter from this parent
            
        Returns:
            New PhysicalParameters combining both parents
        """
        child = PhysicalParameters()
        
        # Crossover each parameter
        for field_name in self.__dataclass_fields__:
            if random.random() < crossover_rate:
                setattr(child, field_name, getattr(self, field_name))
            else:
                setattr(child, field_name, getattr(other, field_name))
        
        return child
    
    def average_with(self, other: 'PhysicalParameters', 
                    weight: float = 0.5) -> 'PhysicalParameters':
        """
        Create a parameter set by averaging with another set.
        
        Args:
            other: Other parameter set to average with
            weight: Weight for this parameter set (0.5 = equal averaging)
            
        Returns:
            New PhysicalParameters with averaged values
        """
        averaged = PhysicalParameters()
        
        for field_name in self.__dataclass_fields__:
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)
            
            if isinstance(self_val, (int, float)):
                averaged_val = self_val * weight + other_val * (1 - weight)
                setattr(averaged, field_name, averaged_val)
            else:
                # For non-numeric fields, randomly choose
                chosen_val = self_val if random.random() < weight else other_val
                setattr(averaged, field_name, chosen_val)
        
        return averaged
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """
        Get metrics that represent the diversity of this parameter set.
        Used for maintaining population diversity.
        
        Returns:
            Dictionary of diversity metrics
        """
        return {
            'body_size': self.body_width * self.body_height,
            'arm_length_ratio': self.arm_length / self.wrist_length,
            'wheel_body_ratio': self.wheel_radius / self.body_height,
            'motor_power': self.motor_torque * self.motor_speed,
            'learning_aggressiveness': self.learning_rate / self.epsilon,
            'stability_focus': self.stability_weight / self.speed_value_weight,
            'suspension_ratio': self.suspension / self.body_height,
        }
    
    def validate_and_repair(self) -> 'PhysicalParameters':
        """
        Ensure all parameters are within valid ranges and repair if needed.
        
        Returns:
            Valid parameter set (may be modified)
        """
        repaired = deepcopy(self)
        
        # Ensure minimum viable sizes
        repaired.body_width = max(0.5, repaired.body_width)
        repaired.body_height = max(0.3, repaired.body_height)
        repaired.arm_length = max(0.3, repaired.arm_length)
        repaired.wrist_length = max(0.3, repaired.wrist_length)
        repaired.wheel_radius = max(0.1, repaired.wheel_radius)
        
        # Ensure positive values where needed
        repaired.body_density = max(0.1, repaired.body_density)
        repaired.motor_torque = max(10.0, repaired.motor_torque)
        repaired.motor_speed = max(0.5, repaired.motor_speed)
        
        # Ensure learning rates are reasonable
        repaired.learning_rate = np.clip(repaired.learning_rate, 0.001, 0.1)
        repaired.epsilon = np.clip(repaired.epsilon, 0.01, 0.8)
        repaired.discount_factor = np.clip(repaired.discount_factor, 0.1, 0.99)
        
        return repaired
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicalParameters':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def random_parameters(cls, base_params: Optional['PhysicalParameters'] = None) -> 'PhysicalParameters':
        """
        Generate random parameters, optionally based on a template.
        
        Args:
            base_params: Base parameters to vary from (None for completely random)
            
        Returns:
            New random PhysicalParameters
        """
        if base_params is None:
            base_params = cls()  # Use defaults
        
        # Create a heavily mutated version
        random_params = base_params.mutate(mutation_rate=0.8)
        return random_params.validate_and_repair()


class PhysicalParameterSpace:
    """
    Manages the space of possible physical parameters and provides
    utilities for evolution and diversity maintenance.
    """
    
    def __init__(self):
        self.parameter_ranges = self._define_parameter_ranges()
        self.diversity_history: List[Dict[str, float]] = []
    
    def _define_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Define the valid ranges for each parameter."""
        return {
            'body_width': (0.8, 3.0),
            'body_height': (0.3, 1.5),
            'arm_length': (0.5, 2.5),
            'wrist_length': (0.5, 2.5),
            'wheel_radius': (0.2, 1.2),
            'motor_torque': (50.0, 400.0),
            'motor_speed': (1.0, 10.0),
            'learning_rate': (0.001, 0.1),
            'epsilon': (0.01, 0.8),
        }
    
    def calculate_population_diversity(self, 
                                     parameter_sets: List[PhysicalParameters]) -> float:
        """
        Calculate the diversity of a population of parameter sets.
        
        Args:
            parameter_sets: List of parameter sets to analyze
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(parameter_sets) < 2:
            return 0.0
        
        # Get diversity metrics for all parameter sets
        all_metrics = [params.get_diversity_metrics() for params in parameter_sets]
        
        # Calculate variance across each metric
        diversity_scores = []
        for metric_name in all_metrics[0].keys():
            values = [metrics[metric_name] for metrics in all_metrics]
            if len(set(values)) > 1:  # Avoid division by zero
                variance = np.var(values)
                mean_val = np.mean(values)
                normalized_variance = variance / (mean_val + 1e-8)  # Normalize by mean
                diversity_scores.append(normalized_variance)
        
        return float(np.mean(diversity_scores)) if diversity_scores else 0.0
    
    def maintain_diversity(self, 
                          parameter_sets: List[PhysicalParameters],
                          target_diversity: float = 0.3) -> List[PhysicalParameters]:
        """
        Ensure population maintains minimum diversity by introducing variation.
        
        Args:
            parameter_sets: Current population parameter sets
            target_diversity: Minimum desired diversity level
            
        Returns:
            Modified parameter sets with enhanced diversity
        """
        current_diversity = self.calculate_population_diversity(parameter_sets)
        
        if current_diversity >= target_diversity:
            return parameter_sets  # Already diverse enough
        
        # Add diversity by mutating some individuals
        enhanced_sets = parameter_sets.copy()
        num_to_diversify = max(1, len(parameter_sets) // 4)  # Diversify 25%
        
        # Choose least diverse individuals (those closest to population center)
        indices_to_diversify = random.sample(range(len(enhanced_sets)), num_to_diversify)
        
        for idx in indices_to_diversify:
            # Apply stronger mutation to increase diversity
            enhanced_sets[idx] = enhanced_sets[idx].mutate(mutation_rate=0.4)
            enhanced_sets[idx] = enhanced_sets[idx].validate_and_repair()
        
        return enhanced_sets 