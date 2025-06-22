"""
Body factory for creating different types of physics bodies.
"""

import pymunk
from typing import Tuple, List, Optional
import math


class BodyFactory:
    """Factory for creating physics bodies and shapes."""
    
    @staticmethod
    def create_static_body(space: pymunk.Space, position: Tuple[float, float] = (0, 0)) -> pymunk.Body:
        """Create a static body."""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        space.add(body)
        return body
    
    @staticmethod
    def create_dynamic_body(space: pymunk.Space, position: Tuple[float, float] = (0, 0), 
                          mass: float = 1.0, moment: float = None) -> pymunk.Body:
        """Create a dynamic body."""
        if moment is None:
            moment = pymunk.moment_for_circle(mass, 0, 1)
        
        body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
        body.position = position
        space.add(body)
        return body
    
    @staticmethod
    def create_kinematic_body(space: pymunk.Space, position: Tuple[float, float] = (0, 0)) -> pymunk.Body:
        """Create a kinematic body."""
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        space.add(body)
        return body
    
    @staticmethod
    def create_circle_shape(body: pymunk.Body, radius: float, offset: Tuple[float, float] = (0, 0),
                          density: float = 1.0, friction: float = 0.7, 
                          restitution: float = 0.0, collision_type: int = 0) -> pymunk.Shape:
        """Create a circle shape and add it to a body."""
        shape = pymunk.Circle(body, radius, offset)
        shape.density = density
        shape.friction = friction
        shape.restitution = restitution
        shape.collision_type = collision_type
        return shape
    
    @staticmethod
    def create_box_shape(body: pymunk.Body, size: Tuple[float, float], 
                        density: float = 1.0, friction: float = 0.7,
                        restitution: float = 0.0, collision_type: int = 0) -> pymunk.Shape:
        """Create a box shape and add it to a body."""
        shape = pymunk.Poly.create_box(body, size)
        shape.density = density
        shape.friction = friction
        shape.restitution = restitution
        shape.collision_type = collision_type
        return shape
    
    @staticmethod
    def create_polygon_shape(body: pymunk.Body, vertices: List[Tuple[float, float]],
                           density: float = 1.0, friction: float = 0.7,
                           restitution: float = 0.0, collision_type: int = 0) -> pymunk.Shape:
        """Create a polygon shape and add it to a body."""
        shape = pymunk.Poly(body, vertices)
        shape.density = density
        shape.friction = friction
        shape.restitution = restitution
        shape.collision_type = collision_type
        return shape
    
    @staticmethod
    def create_segment_shape(body: pymunk.Body, a: Tuple[float, float], b: Tuple[float, float],
                           thickness: float = 1.0, density: float = 1.0, 
                           friction: float = 0.7, restitution: float = 0.0,
                           collision_type: int = 0) -> pymunk.Shape:
        """Create a segment shape and add it to a body."""
        shape = pymunk.Segment(body, a, b, thickness)
        shape.density = density
        shape.friction = friction
        shape.restitution = restitution
        shape.collision_type = collision_type
        return shape
    
    @staticmethod
    def create_ball(space: pymunk.Space, position: Tuple[float, float], radius: float,
                   density: float = 0.25, friction: float = 0.33, restitution: float = 0.9,
                   collision_type: int = 0, collision_mask: int = 0xFFFF) -> pymunk.Body:
        """Create a ball (circle body) with the given properties."""
        body = BodyFactory.create_dynamic_body(space, position, density * math.pi * radius * radius)
        
        shape = BodyFactory.create_circle_shape(
            body, radius, density=density, friction=friction, 
            restitution=restitution, collision_type=collision_type
        )
        
        # Set collision filter
        shape.filter = pymunk.ShapeFilter(group=0, categories=collision_type, mask=collision_mask)
        
        space.add(shape)
        return body
    
    @staticmethod
    def create_crate(space: pymunk.Space, position: Tuple[float, float], 
                    size: Tuple[float, float], density: float = 0.65,
                    friction: float = 0.33, restitution: float = 0.2,
                    collision_type: int = 0, collision_mask: int = 0xFFFF) -> pymunk.Body:
        """Create a crate (box body) with the given properties."""
        mass = density * size[0] * size[1]
        moment = pymunk.moment_for_box(mass, size)
        
        body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
        body.position = position
        space.add(body)
        
        shape = BodyFactory.create_box_shape(
            body, size, density=density, friction=friction,
            restitution=restitution, collision_type=collision_type
        )
        
        # Set collision filter
        shape.filter = pymunk.ShapeFilter(group=0, categories=collision_type, mask=collision_mask)
        
        space.add(shape)
        return body
    
    @staticmethod
    def create_ground_segment(space: pymunk.Space, a: Tuple[float, float], b: Tuple[float, float],
                            friction: float = 0.5, restitution: float = 0.0,
                            collision_type: int = 0) -> pymunk.Shape:
        """Create a ground segment (static line)."""
        body = BodyFactory.create_static_body(space)
        
        shape = BodyFactory.create_segment_shape(
            body, a, b, thickness=1.0, density=0, friction=friction,
            restitution=restitution, collision_type=collision_type
        )
        
        space.add(shape)
        return shape
    
    @staticmethod
    def destroy_body(space: pymunk.Space, body: pymunk.Body):
        """Safely destroy a body and all its shapes."""
        if body is None:
            return
            
        # Remove all shapes from the body
        for shape in body.shapes:
            space.remove(shape)
        
        # Remove the body
        space.remove(body) 