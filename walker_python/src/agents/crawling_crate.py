"""
CrawlingCrate agent: articulated robot with a crate body and two arms for crawling locomotion.
"""

import Box2D as b2
import numpy as np
from typing import Tuple, List, Dict, Any
from .base_agent import BaseAgent

class CrawlingCrate(BaseAgent):
    """
    Articulated robot with a crate body, two wheels, and a single two-segment arm,
    all simulated using the Box2D physics engine.
    """
    def __init__(self, world: b2.b2World, agent_id: int, position: Tuple[float, float] = (0, 5), category_bits=0x0001, mask_bits=0xFFFF):
        super().__init__()
        self.world = world
        self.initial_position = position
        self.id = agent_id
        
        # Physics properties for collision filtering
        self.filter = b2.b2Filter(
            categoryBits=category_bits,
            maskBits=mask_bits
        )
        
        self._create_body()
        self.reset()

    def _create_body(self):
        # Body definition
        body_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody,
            position=self.initial_position,
            linearDamping=0.05,  # Reduced damping
            angularDamping=0.05  # Reduced damping
        )
        self.body = self.world.CreateBody(body_def)
        
        # Main chassis fixture
        chassis_shape = b2.b2PolygonShape(box=(1.5, 0.75))
        chassis_fixture = self.body.CreateFixture(shape=chassis_shape, density=2.0, friction=0.8)
        chassis_fixture.filterData = self.filter

        # Wheels
        self.wheels = []
        for dx in [-1, 1]:
            wheel_def = b2.b2BodyDef(
                type=b2.b2_dynamicBody,
                position=self.body.GetWorldPoint((dx, -0.75)),
                linearDamping=0.05,  # Reduced damping
                angularDamping=0.05  # Reduced damping
            )
            wheel = self.world.CreateBody(wheel_def)
            wheel_fixture = wheel.CreateFixture(shape=b2.b2CircleShape(radius=0.5), density=5.0, friction=0.9)
            wheel_fixture.filterData = self.filter
            self.wheels.append(wheel)

            # Revolute joint to attach wheel to chassis
            r_joint_def = b2.b2RevoluteJointDef(
                bodyA=self.body,
                bodyB=wheel,
                localAnchorA=(dx, -0.75),
                localAnchorB=(0, 0),
            )
            self.world.CreateJoint(r_joint_def)

        # Arm - removed collision filtering so arms can interact with ground
        # Upper arm
        upper_arm_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody, 
            position=self.body.GetWorldPoint((0,0.75)),
            linearDamping=0.05,  # Reduced damping
            angularDamping=0.05  # Reduced damping
        )
        self.upper_arm = self.world.CreateBody(upper_arm_def)
        upper_arm_fixture = self.upper_arm.CreateFixture(
            shape=b2.b2PolygonShape(box=(1.25, 0.1)), 
            density=0.5,  # Reduced density for lighter arms
            friction=0.5
        )
        upper_arm_fixture.filterData = self.filter

        # Lower arm
        lower_arm_def = b2.b2BodyDef(
            type=b2.b2_dynamicBody, 
            position=self.upper_arm.GetWorldPoint((1.25,0)),
            linearDamping=0.05,  # Reduced damping
            angularDamping=0.05  # Reduced damping
        )
        self.lower_arm = self.world.CreateBody(lower_arm_def)
        lower_arm_fixture = self.lower_arm.CreateFixture(
            shape=b2.b2PolygonShape(box=(1.25, 0.1)), 
            density=0.5,  # Reduced density for lighter arms
            friction=0.5  # Reduced friction to prevent sticking
        )
        lower_arm_fixture.filterData = self.filter
        
        # Arm joints - with limits for realistic range of motion
        shoulder_joint_def = b2.b2RevoluteJointDef(
            bodyA=self.body, 
            bodyB=self.upper_arm, 
            localAnchorA=(0,0.75), 
            localAnchorB=(-1.25, 0)
        )
        shoulder_joint_def.enableLimit = True
        shoulder_joint_def.lowerAngle = 0  # 0 degrees (arm pointing forward)
        shoulder_joint_def.upperAngle = 360   # +180 degrees (arm pointing backward)
        self.shoulder_joint = self.world.CreateJoint(shoulder_joint_def)
        
        elbow_joint_def = b2.b2RevoluteJointDef(
            bodyA=self.upper_arm, 
            bodyB=self.lower_arm, 
            localAnchorA=(1.25,0), 
            localAnchorB=(-1.25, 0)
        )
        elbow_joint_def.enableLimit = True
        elbow_joint_def.lowerAngle = 0           # 0 degrees (fully extended)
        elbow_joint_def.upperAngle = 3*np.pi/4    # +135 degrees
        self.elbow_joint = self.world.CreateJoint(elbow_joint_def)

    def reset(self):
        self.body.position = self.initial_position
        self.body.angle = 0
        self.body.linearVelocity = (0,0)
        self.body.angularVelocity = 0

        for i, wheel in enumerate(self.wheels):
            dx = -1 if i == 0 else 1
            wheel.position = self.body.GetWorldPoint((dx, -0.75))
            wheel.angle = 0
            wheel.linearVelocity = (0,0)
            wheel.angularVelocity = 0

        self.upper_arm.position = self.body.GetWorldPoint((0,0.75))
        self.upper_arm.angle = -np.pi / 3  # More downward angle for better ground contact
        
        self.lower_arm.position = self.upper_arm.GetWorldPoint((1.25,0))
        self.lower_arm.angle = np.pi / 3   # More upward angle to create a crawling stance

    def apply_action(self, action: Tuple[float, float]):
        # Reduced motor strength for more controlled movement
        shoulder_torque = float(np.clip(action[0], -15.0, 15.0)) * 50.0
        elbow_torque = float(np.clip(action[1], -15.0, 15.0)) * 50.0

        # Apply torque and ensure bodies are awake
        self.upper_arm.ApplyTorque(shoulder_torque, wake=True)
        self.lower_arm.ApplyTorque(elbow_torque, wake=True)
        
        # Removed debug print to eliminate overhead - was running 1% of the time

    def get_state(self) -> np.ndarray:
        return np.array([
            self.body.position.x, self.body.position.y,
            self.body.linearVelocity.x, self.body.linearVelocity.y,
            self.body.angle, self.upper_arm.angle, self.lower_arm.angle
        ], dtype=np.float32)

    def get_reward(self, prev_x: float) -> float:
        return (self.body.position.x - prev_x) * 10

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            'crate_pos': tuple(self.body.position),
            'crate_vel': tuple(self.body.linearVelocity),
            'crate_angle': self.body.angle,
            'upper_arm_pos': tuple(self.upper_arm.position),
            'upper_arm_angle': self.upper_arm.angle,
            'lower_arm_pos': tuple(self.lower_arm.position),
            'lower_arm_angle': self.lower_arm.angle,
            'wheels_pos': [tuple(w.position) for w in self.wheels],
            'wheels_angle': [w.angle for w in self.wheels]
        }

    def destroy(self):
        self.world.DestroyBody(self.body)
        self.world.DestroyBody(self.upper_arm)
        self.world.DestroyBody(self.lower_arm)
        for wheel in self.wheels:
            self.world.DestroyBody(wheel)
        # Joints are destroyed when bodies are

    def take_action(self, action: Tuple[float, float]):
        self.apply_action(action)
        
    def update(self, delta_time):
        pass

    def mutate(self, mutation_rate: float = 0.1):
        pass
            
    def crossover(self, other: 'CrawlingCrate') -> 'CrawlingCrate':
        return CrawlingCrate(
            self.world, 
            self.id, 
            self.initial_position,
            category_bits=self.filter.categoryBits,
            mask_bits=self.filter.maskBits
        )
        
    def copy(self) -> 'CrawlingCrate':
        return CrawlingCrate(
            self.world, 
            self.id, 
            self.initial_position,
            category_bits=self.filter.categoryBits,
            mask_bits=self.filter.maskBits
        ) 