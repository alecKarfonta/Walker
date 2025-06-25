"""
Physics world controller for managing the simulation.
"""

import pymunk
from typing import List, Tuple, Optional
from .contact_handler import ContactHandler
from .body_factory import BodyFactory


class WorldController:
    """Main physics world controller."""
    
    def __init__(self, gravity: Tuple[float, float] = (0, -9.8)):
        # Create the physics space
        self.space = pymunk.Space()
        self.space.gravity = gravity
        
        # Set up contact handler
        self.contact_handler = ContactHandler(self)
        self.space.collision_handler = self.contact_handler
        
        # Physics parameters
        self.timestep = 1.0 / 60.0
        self.velocity_iterations = 4
        self.position_iterations = 8
        
        # World bounds
        self.width = 800
        self.height = 600
        self.zoom = 25
        self.end = 10000
        
        # Calculate world bounds
        self.right = self.width / 2 / self.zoom
        self.left = -(self.width / 2 / self.zoom)
        self.top = self.height / 2 / self.zoom
        self.bottom = -(self.height / 2 / self.zoom)
        self.ground_height = self.bottom + 10
        
        # Ground body reference
        self.ground_body = None
        
        # Body destruction queue
        self.destroy_queue = []
        
        # Initialize the world
        self.init()
    
    def init(self):
        """Initialize the physics world."""
        # Create ground and terrain
        self.create_ground()
    
    def update(self, delta_time: float):
        """Update the physics simulation."""
        # Step the physics simulation
        self.space.step(self.timestep)
        
        # Process destruction queue
        self.destroy_queue()
    
    def create_ground(self):
        """Create the ground terrain."""
        # Create triangular terrain pieces
        for x in range(int(-self.end * 0.3) - 10, int(self.end * 0.1) + 10):
            # Create triangle shape
            triangle_vertices = [
                (0 - 5, 0),
                (0, 3),
                (0 + 5, 0)
            ]
            
            # Create static body for triangle
            body = BodyFactory.create_static_body(
                self.space, 
                position=(x * 10, self.ground_height - 3)
            )
            
            # Create triangle shape
            shape = BodyFactory.create_polygon_shape(
                body, triangle_vertices,
                density=0, friction=0.5, restitution=0.0,
                collision_type=1  # FILTER_BOUNDARY
            )
            
            self.space.add(shape)
        
        # Create main ground floor
        ground_body = BodyFactory.create_static_body(
            self.space, 
            position=(0, self.ground_height)
        )
        
        # Create ground segment
        ground_shape = BodyFactory.create_ground_segment(
            self.space,
            (self.left - self.end, 0),
            (self.right + self.end, 0),
            friction=0.5, restitution=0.0,
            collision_type=1  # FILTER_BOUNDARY
        )
        
        self.ground_body = ground_body
        
        # Create additional ground layers
        # Lower ground layer
        lower_ground = BodyFactory.create_static_body(
            self.space,
            position=(0, self.ground_height - 3)
        )
        
        BodyFactory.create_ground_segment(
            self.space,
            (self.left - self.end, 0),
            (self.right + self.end, 0),
            friction=0.5, restitution=0.0,
            collision_type=1
        )
        
        # Upper ground layer
        upper_ground = BodyFactory.create_static_body(
            self.space,
            position=(0, self.ground_height + 60)
        )
        
        BodyFactory.create_ground_segment(
            self.space,
            (self.left - 100, 0),
            (self.right + 100, 0),
            friction=0.5, restitution=0.0,
            collision_type=1
        )
        
        # Deep ground layer
        deep_ground = BodyFactory.create_static_body(
            self.space,
            position=(0, self.ground_height - 30)
        )
        
        BodyFactory.create_ground_segment(
            self.space,
            (self.left - self.end, 0),
            (self.right + self.end, 0),
            friction=0.5, restitution=0.0,
            collision_type=1
        )
    
    def destroy_body(self, body: pymunk.Body):
        """Queue a body for destruction."""
        if body not in self.destroy_queue:
            self.destroy_queue.append(body)
    
    def destroy_queue(self):
        """Process the destruction queue."""
        if self.destroy_queue:
            for body in self.destroy_queue:
                if body is not None:
                    BodyFactory.destroy_body(self.space, body)
            self.destroy_queue.clear()
    
    def get_gravity(self) -> Tuple[float, float]:
        """Get the current gravity vector."""
        return self.space.gravity
    
    def set_gravity(self, gravity: Tuple[float, float]):
        """Set the gravity vector."""
        self.space.gravity = gravity
    
    def get_timestep(self) -> float:
        """Get the current timestep."""
        return self.timestep
    
    def set_timestep(self, timestep: float):
        """Set the physics timestep."""
        self.timestep = timestep
    
    def get_velocity_iterations(self) -> int:
        """Get the number of velocity iterations."""
        return self.velocity_iterations
    
    def set_velocity_iterations(self, iterations: int):
        """Set the number of velocity iterations."""
        self.velocity_iterations = iterations
    
    def get_position_iterations(self) -> int:
        """Get the number of position iterations."""
        return self.position_iterations
    
    def set_position_iterations(self, iterations: int):
        """Set the number of position iterations."""
        self.position_iterations = iterations
    
    def add_body(self, body: pymunk.Body):
        """Add a body to the physics space."""
        self.space.add(body)
    
    def remove_body(self, body: pymunk.Body):
        """Remove a body from the physics space."""
        self.space.remove(body)
    
    def add_shape(self, shape: pymunk.Shape):
        """Add a shape to the physics space."""
        self.space.add(shape)
    
    def remove_shape(self, shape: pymunk.Shape):
        """Remove a shape from the physics space."""
        self.space.remove(shape)
    
    def add_constraint(self, constraint: pymunk.Constraint):
        """Add a constraint to the physics space."""
        self.space.add(constraint)
    
    def remove_constraint(self, constraint: pymunk.Constraint):
        """Remove a constraint from the physics space."""
        self.space.remove(constraint) 