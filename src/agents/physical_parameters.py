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
    Designed to simulate the incredible diversity of animal evolution.
    """
    
    # Body parameters (inspired by Java CrawlingCrate)
    body_width: float = 1.5
    body_height: float = 0.75
    body_density: float = 4.0
    body_friction: float = 0.9
    body_restitution: float = 0.1
    body_linear_damping: float = 0.05
    body_angular_damping: float = 0.05
    
    # NEW: Body shape and size evolution (like real animals!)
    body_shape: str = "rectangle"  # "rectangle", "oval", "triangle", "trapezoid", "diamond"
    body_taper: float = 1.0       # How much the body tapers (1.0 = no taper, 0.5 = half width at back)
    body_curve: float = 0.0       # Curvature of body sides (0.0 = straight, 1.0 = very curved)
    overall_scale: float = 1.0    # Overall size multiplier (0.5 = tiny, 2.0 = giant)
    body_aspect_ratio: float = 2.0 # Length to width ratio (like different animal proportions)
    
    # NEW: Arm attachment and configuration (massive biological diversity!)
    arm_attachment_x: float = 0.0     # Where along body length arm attaches (-1.0 to 1.0)
    arm_attachment_y: float = 0.5     # How high on body arm attaches (0.0 = bottom, 1.0 = top)
    num_arms: int = 1                 # Number of arms (1-6, like insects, spiders, etc.)
    arm_symmetry: float = 1.0         # Symmetry between left/right arms (1.0 = identical, 0.0 = completely different)
    arm_angle_offset: float = 0.0     # Base angle offset for arm positioning
    
    # NEW: Variable limb segments (like different joint configurations!)
    segments_per_limb: int = 2        # Number of segments per limb (2-5, like different animals)
    segment_length_ratios: List[float] = field(default_factory=lambda: [1.0, 1.0])  # Relative lengths of each segment
    segment_width_ratios: List[float] = field(default_factory=lambda: [1.0, 0.8])   # Relative widths of each segment
    joint_flexibility_per_segment: List[float] = field(default_factory=lambda: [1.0, 1.0])  # Flexibility of each joint
    
    # NEW: Limb specialization (like different animal locomotion)
    limb_specialization: str = "general"  # "general", "digging", "climbing", "swimming", "grasping"
    arm_flexibility: float = 1.0          # How flexible the arm is (0.5 = rigid, 2.0 = very flexible)
    joint_stiffness: float = 1.0          # Joint resistance (0.5 = loose, 2.0 = stiff)
    
    # Arm parameters (extensive like Java implementation)
    arm_length: float = 1.0  # Upper arm length
    arm_width: float = 0.2   # Upper arm width
    wrist_length: float = 1.0  # Lower arm length
    wrist_width: float = 0.2   # Lower arm width
    arm_density: float = 0.1
    arm_friction: float = 0.5
    arm_restitution: float = 0.1
    
    # Joint parameters - now dynamic based on limb configuration
    arm_torque: float = 150.0
    wrist_torque: float = 150.0
    arm_speed: float = 3.0
    wrist_speed: float = 3.0
    shoulder_lower_limit: float = -np.pi/2  # -90 degrees
    shoulder_upper_limit: float = np.pi/2   # +90 degrees
    elbow_lower_limit: float = 0.0          # 0 degrees
    elbow_upper_limit: float = 3*np.pi/4    # 135 degrees
    
    # NEW: Dynamic joint parameters for variable segments
    joint_torques: List[float] = field(default_factory=lambda: [150.0, 100.0])  # Torque for each joint
    joint_speeds: List[float] = field(default_factory=lambda: [3.0, 3.0])       # Speed for each joint
    joint_lower_limits: List[float] = field(default_factory=lambda: [-np.pi/2, 0.0])     # Lower angle limits
    joint_upper_limits: List[float] = field(default_factory=lambda: [np.pi/2, 3*np.pi/4]) # Upper angle limits
    
    # Wheel parameters
    wheel_radius: float = 0.5
    wheel_density: float = 8.0
    wheel_friction: float = 0.9
    wheel_restitution: float = 0.2
    leg_spread: float = 2.0  # Distance between wheels
    suspension: float = 0.75  # Height of wheels below body
    ride_height: float = 0.0  # Additional height offset
    
    # NEW: Wheel/leg evolution (like different animal feet!)
    wheel_shape: str = "circle"       # "circle", "oval", "star", "bumpy" (like paws, hooves, etc.)
    num_wheels: int = 2               # Number of wheels/legs (2-8, like different animals)
    wheel_asymmetry: float = 0.0      # Left/right wheel size difference
    leg_angle: float = 0.0            # Angle of leg attachment (splayed out vs straight down)
    wheel_size_variation: float = 0.0 # Variation in wheel sizes (0.0 = all same, 1.0 = very different)
    
    # NEW: Action space configuration for dynamic control
    action_combination_style: str = "independent"  # "independent", "paired", "sequential", "coordinated"
    max_simultaneous_joints: int = 2   # Maximum joints that can move simultaneously
    joint_priority_weights: List[float] = field(default_factory=lambda: [1.0, 0.8])  # Priority of each joint
    
    # NEW: Locomotion specialization (crawlers vs walkers vs rollers)
    locomotion_type: str = "crawler"  # "crawler", "walker", "roller", "jumper", "hybrid"
    ground_contact_area: float = 1.0  # How much surface area touches ground
    stability_preference: float = 0.5 # Trade-off between speed and stability
    
    # NEW: Appendages and extras (like tails, fins, etc.)
    has_tail: bool = False            # Whether robot has a tail
    tail_length: float = 1.0          # Length of tail if present
    tail_flexibility: float = 1.0     # How flexible the tail is
    appendage_count: int = 0          # Additional small appendages (0-3)
    
    # NEW: Material and structural properties (like bone vs cartilage)
    body_rigidity: float = 1.0        # How rigid the body structure is (0.5 = flexible, 2.0 = very rigid)
    weight_distribution: str = "center"  # "center", "front", "back", "low", "high"
    structural_reinforcement: float = 1.0  # Extra structural strength (like thick bones)
    
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
        
        # NEW: Body shape and size mutations (animal-like evolution!)
        if random.random() < mutation_rate:
            body_shapes = ["rectangle", "oval", "triangle", "trapezoid", "diamond"]
            mutated.body_shape = random.choice(body_shapes)
        if random.random() < mutation_rate:
            mutated.body_taper = self._mutate_bounded(
                self.body_taper, 0.3, 0.3, 1.5
            )
        if random.random() < mutation_rate:
            mutated.body_curve = self._mutate_bounded(
                self.body_curve, 0.4, 0.0, 1.0
            )
        if random.random() < mutation_rate:
            mutated.overall_scale = self._mutate_bounded(
                self.overall_scale, 0.3, 0.4, 2.5  # Huge size variation!
            )
        if random.random() < mutation_rate:
            mutated.body_aspect_ratio = self._mutate_bounded(
                self.body_aspect_ratio, 0.4, 1.0, 4.0
            )
        
        # NEW: Arm attachment mutations (where arms attach to body)
        if random.random() < mutation_rate:
            mutated.arm_attachment_x = self._mutate_bounded(
                self.arm_attachment_x, 0.5, -0.8, 0.8  # Along body length
            )
        if random.random() < mutation_rate:
            mutated.arm_attachment_y = self._mutate_bounded(
                self.arm_attachment_y, 0.3, 0.0, 1.0   # Height on body
            )
        if random.random() < mutation_rate:
            mutated.num_arms = max(1, min(6, self.num_arms + random.choice([-1, 0, 1])))
        if random.random() < mutation_rate:
            mutated.arm_symmetry = self._mutate_bounded(
                self.arm_symmetry, 0.3, 0.3, 1.0  # Asymmetrical evolution!
            )
        if random.random() < mutation_rate:
            mutated.arm_angle_offset = self._mutate_bounded(
                self.arm_angle_offset, 0.4, -np.pi/3, np.pi/3
            )
        
        # NEW: Variable limb segment mutations (revolutionary limb evolution!)
        if random.random() < mutation_rate:
            new_segments = max(2, min(5, self.segments_per_limb + random.choice([-1, 0, 1])))
            mutated.segments_per_limb = new_segments
            
            # Resize segment parameter arrays to match new segment count
            mutated.segment_length_ratios = self._resize_array(
                self.segment_length_ratios, new_segments, default_value=1.0
            )
            mutated.segment_width_ratios = self._resize_array(
                self.segment_width_ratios, new_segments, default_value=0.8
            )
            mutated.joint_flexibility_per_segment = self._resize_array(
                self.joint_flexibility_per_segment, new_segments, default_value=1.0
            )
            mutated.joint_torques = self._resize_array(
                self.joint_torques, new_segments, default_value=120.0
            )
            mutated.joint_speeds = self._resize_array(
                self.joint_speeds, new_segments, default_value=3.0
            )
            mutated.joint_lower_limits = self._resize_array(
                self.joint_lower_limits, new_segments, default_value=-np.pi/4
            )
            mutated.joint_upper_limits = self._resize_array(
                self.joint_upper_limits, new_segments, default_value=np.pi/2
            )
            mutated.joint_priority_weights = self._resize_array(
                self.joint_priority_weights, new_segments, default_value=0.8
            )
        
        # Mutate individual segment parameters
        if random.random() < mutation_rate:
            mutated.segment_length_ratios = [
                self._mutate_bounded(ratio, 0.3, 0.3, 2.0) for ratio in mutated.segment_length_ratios
            ]
        if random.random() < mutation_rate:
            mutated.segment_width_ratios = [
                self._mutate_bounded(ratio, 0.3, 0.2, 1.5) for ratio in mutated.segment_width_ratios
            ]
        if random.random() < mutation_rate:
            mutated.joint_flexibility_per_segment = [
                self._mutate_bounded(flex, 0.3, 0.2, 3.0) for flex in mutated.joint_flexibility_per_segment
            ]
        if random.random() < mutation_rate:
            mutated.joint_torques = [
                self._mutate_bounded(torque, 0.3, 30.0, 400.0) for torque in mutated.joint_torques
            ]
        if random.random() < mutation_rate:
            mutated.joint_speeds = [
                self._mutate_bounded(speed, 0.3, 0.5, 10.0) for speed in mutated.joint_speeds
            ]
        
        # NEW: Limb specialization mutations
        if random.random() < mutation_rate:
            specializations = ["general", "digging", "climbing", "swimming", "grasping"]
            mutated.limb_specialization = random.choice(specializations)
        if random.random() < mutation_rate:
            mutated.arm_flexibility = self._mutate_bounded(
                self.arm_flexibility, 0.3, 0.3, 3.0
            )
        if random.random() < mutation_rate:
            mutated.joint_stiffness = self._mutate_bounded(
                self.joint_stiffness, 0.3, 0.3, 3.0
            )
        
        # NEW: Wheel/leg mutations (feet evolution!)
        if random.random() < mutation_rate:
            wheel_shapes = ["circle", "oval", "star", "bumpy"]
            mutated.wheel_shape = random.choice(wheel_shapes)
        if random.random() < mutation_rate:
            mutated.num_wheels = max(2, min(8, self.num_wheels + random.choice([-1, 0, 1])))
        if random.random() < mutation_rate:
            mutated.wheel_asymmetry = self._mutate_bounded(
                self.wheel_asymmetry, 0.3, 0.0, 0.5
            )
        if random.random() < mutation_rate:
            mutated.leg_angle = self._mutate_bounded(
                self.leg_angle, 0.3, -np.pi/4, np.pi/4
            )
        if random.random() < mutation_rate:
            mutated.wheel_size_variation = self._mutate_bounded(
                self.wheel_size_variation, 0.3, 0.0, 1.0
            )
        
        # NEW: Action space mutations (control strategy evolution!)
        if random.random() < mutation_rate:
            action_styles = ["independent", "paired", "sequential", "coordinated"]
            mutated.action_combination_style = random.choice(action_styles)
        if random.random() < mutation_rate:
            mutated.max_simultaneous_joints = max(1, min(mutated.segments_per_limb, 
                self.max_simultaneous_joints + random.choice([-1, 0, 1])))
        if random.random() < mutation_rate:
            mutated.joint_priority_weights = [
                self._mutate_bounded(weight, 0.3, 0.1, 2.0) for weight in mutated.joint_priority_weights
            ]
        
        # NEW: Locomotion type mutations
        if random.random() < mutation_rate:
            locomotion_types = ["crawler", "walker", "roller", "jumper", "hybrid"]
            mutated.locomotion_type = random.choice(locomotion_types)
        if random.random() < mutation_rate:
            mutated.ground_contact_area = self._mutate_bounded(
                self.ground_contact_area, 0.3, 0.2, 3.0
            )
        if random.random() < mutation_rate:
            mutated.stability_preference = self._mutate_bounded(
                self.stability_preference, 0.3, 0.0, 1.0
            )
        
        # NEW: Appendage mutations (tails and extras!)
        if random.random() < mutation_rate:
            mutated.has_tail = not self.has_tail if random.random() < 0.3 else self.has_tail
        if random.random() < mutation_rate:
            mutated.tail_length = self._mutate_bounded(
                self.tail_length, 0.4, 0.3, 2.0
            )
        if random.random() < mutation_rate:
            mutated.tail_flexibility = self._mutate_bounded(
                self.tail_flexibility, 0.3, 0.3, 2.0
            )
        if random.random() < mutation_rate:
            mutated.appendage_count = max(0, min(3, self.appendage_count + random.choice([-1, 0, 1])))
        
        # NEW: Material property mutations
        if random.random() < mutation_rate:
            mutated.body_rigidity = self._mutate_bounded(
                self.body_rigidity, 0.3, 0.3, 3.0
            )
        if random.random() < mutation_rate:
            weight_distributions = ["center", "front", "back", "low", "high"]
            mutated.weight_distribution = random.choice(weight_distributions)
        if random.random() < mutation_rate:
            mutated.structural_reinforcement = self._mutate_bounded(
                self.structural_reinforcement, 0.3, 0.5, 2.0
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
    
    def _resize_array(self, original_array: List[float], new_size: int, default_value: float) -> List[float]:
        """
        Resize an array to a new size, preserving existing values and filling with defaults.
        
        Args:
            original_array: Original array to resize
            new_size: Target size for the array
            default_value: Value to use for new elements
            
        Returns:
            Resized array
        """
        result = original_array.copy()
        
        if len(result) > new_size:
            # Truncate if too large
            result = result[:new_size]
        elif len(result) < new_size:
            # Extend with default values if too small
            result.extend([default_value] * (new_size - len(result)))
        
        return result
    
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
        Used for maintaining population diversity across all evolutionary traits.
        
        Returns:
            Dictionary of diversity metrics
        """
        # Create hash values for categorical variables to include in diversity
        body_shape_hash = hash(self.body_shape) % 1000 / 1000.0
        limb_spec_hash = hash(self.limb_specialization) % 1000 / 1000.0
        wheel_shape_hash = hash(self.wheel_shape) % 1000 / 1000.0
        locomotion_hash = hash(self.locomotion_type) % 1000 / 1000.0
        weight_dist_hash = hash(self.weight_distribution) % 1000 / 1000.0
        
        return {
            # Original diversity metrics
            'body_size': self.body_width * self.body_height * self.overall_scale,
            'arm_length_ratio': self.arm_length / self.wrist_length,
            'wheel_body_ratio': self.wheel_radius / self.body_height,
            'motor_power': self.motor_torque * self.motor_speed,
            'learning_aggressiveness': self.learning_rate / self.epsilon,
            'stability_focus': self.stability_weight / self.speed_value_weight,
            'suspension_ratio': self.suspension / self.body_height,
            
            # NEW: Shape and form diversity (like different animal body plans!)
            'body_shape_variety': body_shape_hash,
            'body_proportions': self.body_aspect_ratio * self.body_taper,
            'size_scaling': self.overall_scale,
            'body_curvature': self.body_curve,
            
            # NEW: Limb configuration diversity (like arm placement evolution)
            'arm_positioning': abs(self.arm_attachment_x) + self.arm_attachment_y,
            'limb_count': self.num_arms + self.num_wheels,
            'asymmetry_factor': 2.0 - self.arm_symmetry,  # Higher when more asymmetric
            'limb_specialization_type': limb_spec_hash,
            'joint_characteristics': self.arm_flexibility * self.joint_stiffness,
            
            # NEW: Variable limb segment diversity (revolutionary articulation!)
            'segment_complexity': float(self.segments_per_limb * len(self.segment_length_ratios)),
            'segment_proportions': float(np.std(self.segment_length_ratios)) if self.segment_length_ratios else 0.0,
            'joint_torque_diversity': float(np.std(self.joint_torques)) if self.joint_torques else 0.0,
            'joint_priority_balance': float(np.mean(self.joint_priority_weights)) if self.joint_priority_weights else 1.0,
            'limb_articulation': float(self.segments_per_limb * np.mean(self.joint_flexibility_per_segment)) if self.joint_flexibility_per_segment else 1.0,
            
            # NEW: Action space and control diversity
            'action_style_hash': hash(self.action_combination_style) % 1000 / 1000.0,
            'control_complexity': self.max_simultaneous_joints / max(1, self.segments_per_limb),
            'wheel_variation': self.wheel_size_variation + self.wheel_asymmetry,
            
            # NEW: Locomotion diversity (different movement strategies)
            'wheel_configuration': wheel_shape_hash + self.wheel_asymmetry,
            'locomotion_strategy': locomotion_hash,
            'ground_interaction': self.ground_contact_area * self.stability_preference,
            'leg_positioning': abs(self.leg_angle) + self.leg_spread,
            
            # NEW: Structural diversity (like different animal builds)
            'appendage_complexity': (1.0 if self.has_tail else 0.0) + self.appendage_count * 0.3,
            'material_properties': self.body_rigidity * self.structural_reinforcement,
            'weight_balance': weight_dist_hash,
            'flexibility_spectrum': self.arm_flexibility + self.tail_flexibility,
        }
    
    def validate_and_repair(self) -> 'PhysicalParameters':
        """
        Ensure all parameters are within valid ranges and repair if needed.
        Validates both original and new evolutionary parameters.
        
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
        
        # NEW: Validate evolutionary parameters
        # Body shape and proportions
        valid_body_shapes = ["rectangle", "oval", "triangle", "trapezoid", "diamond"]
        if repaired.body_shape not in valid_body_shapes:
            repaired.body_shape = "rectangle"
        repaired.body_taper = np.clip(repaired.body_taper, 0.3, 1.5)
        repaired.body_curve = np.clip(repaired.body_curve, 0.0, 1.0)
        repaired.overall_scale = np.clip(repaired.overall_scale, 0.4, 2.5)
        repaired.body_aspect_ratio = np.clip(repaired.body_aspect_ratio, 1.0, 4.0)
        
        # Arm attachment and configuration
        repaired.arm_attachment_x = np.clip(repaired.arm_attachment_x, -0.8, 0.8)
        repaired.arm_attachment_y = np.clip(repaired.arm_attachment_y, 0.0, 1.0)
        repaired.num_arms = max(1, min(6, int(repaired.num_arms)))
        repaired.arm_symmetry = np.clip(repaired.arm_symmetry, 0.3, 1.0)
        repaired.arm_angle_offset = np.clip(repaired.arm_angle_offset, -np.pi/3, np.pi/3)
        
        # Variable limb segments validation
        repaired.segments_per_limb = max(2, min(5, int(repaired.segments_per_limb)))
        
        # Ensure segment arrays match the number of segments
        target_size = repaired.segments_per_limb
        repaired.segment_length_ratios = repaired._resize_array(
            repaired.segment_length_ratios, target_size, 1.0
        )
        repaired.segment_width_ratios = repaired._resize_array(
            repaired.segment_width_ratios, target_size, 0.8
        )
        repaired.joint_flexibility_per_segment = repaired._resize_array(
            repaired.joint_flexibility_per_segment, target_size, 1.0
        )
        repaired.joint_torques = repaired._resize_array(
            repaired.joint_torques, target_size, 120.0
        )
        repaired.joint_speeds = repaired._resize_array(
            repaired.joint_speeds, target_size, 3.0
        )
        repaired.joint_lower_limits = repaired._resize_array(
            repaired.joint_lower_limits, target_size, -np.pi/4
        )
        repaired.joint_upper_limits = repaired._resize_array(
            repaired.joint_upper_limits, target_size, np.pi/2
        )
        repaired.joint_priority_weights = repaired._resize_array(
            repaired.joint_priority_weights, target_size, 0.8
        )
        
        # Validate segment parameter values
        repaired.segment_length_ratios = [np.clip(ratio, 0.3, 2.0) for ratio in repaired.segment_length_ratios]
        repaired.segment_width_ratios = [np.clip(ratio, 0.2, 1.5) for ratio in repaired.segment_width_ratios]
        repaired.joint_flexibility_per_segment = [np.clip(flex, 0.2, 3.0) for flex in repaired.joint_flexibility_per_segment]
        repaired.joint_torques = [np.clip(torque, 30.0, 400.0) for torque in repaired.joint_torques]
        repaired.joint_speeds = [np.clip(speed, 0.5, 10.0) for speed in repaired.joint_speeds]
        repaired.joint_priority_weights = [np.clip(weight, 0.1, 2.0) for weight in repaired.joint_priority_weights]
        
        # Limb specialization
        valid_specializations = ["general", "digging", "climbing", "swimming", "grasping"]
        if repaired.limb_specialization not in valid_specializations:
            repaired.limb_specialization = "general"
        repaired.arm_flexibility = np.clip(repaired.arm_flexibility, 0.3, 3.0)
        repaired.joint_stiffness = np.clip(repaired.joint_stiffness, 0.3, 3.0)
        
        # Wheel/leg configuration
        valid_wheel_shapes = ["circle", "oval", "star", "bumpy"]
        if repaired.wheel_shape not in valid_wheel_shapes:
            repaired.wheel_shape = "circle"
        repaired.num_wheels = max(2, min(8, int(repaired.num_wheels)))
        repaired.wheel_asymmetry = np.clip(repaired.wheel_asymmetry, 0.0, 0.5)
        repaired.leg_angle = np.clip(repaired.leg_angle, -np.pi/4, np.pi/4)
        repaired.wheel_size_variation = np.clip(repaired.wheel_size_variation, 0.0, 1.0)
        
        # Action space configuration
        valid_action_styles = ["independent", "paired", "sequential", "coordinated"]
        if repaired.action_combination_style not in valid_action_styles:
            repaired.action_combination_style = "independent"
        repaired.max_simultaneous_joints = max(1, min(repaired.segments_per_limb, int(repaired.max_simultaneous_joints)))
        
        # Locomotion type
        valid_locomotion_types = ["crawler", "walker", "roller", "jumper", "hybrid"]
        if repaired.locomotion_type not in valid_locomotion_types:
            repaired.locomotion_type = "crawler"
        repaired.ground_contact_area = np.clip(repaired.ground_contact_area, 0.2, 3.0)
        repaired.stability_preference = np.clip(repaired.stability_preference, 0.0, 1.0)
        
        # Appendages
        repaired.tail_length = max(0.3, repaired.tail_length)
        repaired.tail_flexibility = np.clip(repaired.tail_flexibility, 0.3, 2.0)
        repaired.appendage_count = max(0, min(3, int(repaired.appendage_count)))
        
        # Material properties
        repaired.body_rigidity = np.clip(repaired.body_rigidity, 0.3, 3.0)
        valid_weight_distributions = ["center", "front", "back", "low", "high"]
        if repaired.weight_distribution not in valid_weight_distributions:
            repaired.weight_distribution = "center"
        repaired.structural_reinforcement = np.clip(repaired.structural_reinforcement, 0.5, 2.0)
        
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