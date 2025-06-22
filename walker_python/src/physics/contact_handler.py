"""
Contact handler for collision detection and response.
"""

import pymunk
from typing import Optional, Callable


class ContactHandler:
    """Handles collision detection and contact events."""
    
    def __init__(self, world_controller):
        self.world_controller = world_controller
        self.contact_events = []
    
    def begin_contact(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> bool:
        """Called when two shapes begin to touch."""
        shape_a, shape_b = arbiter.shapes
        
        # Store contact event for processing
        contact_event = {
            'type': 'begin',
            'shape_a': shape_a,
            'shape_b': shape_b,
            'arbiter': arbiter
        }
        self.contact_events.append(contact_event)
        
        # Call user data callbacks if they exist
        if hasattr(shape_a.body, 'user_data') and hasattr(shape_a.body.user_data, 'on_contact_begin'):
            shape_a.body.user_data.on_contact_begin(arbiter, shape_b)
        
        if hasattr(shape_b.body, 'user_data') and hasattr(shape_b.body.user_data, 'on_contact_begin'):
            shape_b.body.user_data.on_contact_begin(arbiter, shape_a)
        
        return True
    
    def pre_solve(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> bool:
        """Called before the collision solver runs."""
        shape_a, shape_b = arbiter.shapes
        
        # Check for high impact forces
        total_impulse = sum(arbiter.total_impulse)
        if total_impulse > 200:
            # Handle high impact collision
            pass
        
        return True
    
    def post_solve(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> None:
        """Called after the collision solver runs."""
        shape_a, shape_b = arbiter.shapes
        
        # Process post-solve events
        contact_event = {
            'type': 'post_solve',
            'shape_a': shape_a,
            'shape_b': shape_b,
            'arbiter': arbiter
        }
        self.contact_events.append(contact_event)
    
    def separate(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> None:
        """Called when two shapes separate."""
        shape_a, shape_b = arbiter.shapes
        
        # Store separation event
        contact_event = {
            'type': 'separate',
            'shape_a': shape_a,
            'shape_b': shape_b,
            'arbiter': arbiter
        }
        self.contact_events.append(contact_event)
        
        # Call user data callbacks if they exist
        if hasattr(shape_a.body, 'user_data') and hasattr(shape_a.body.user_data, 'on_contact_end'):
            shape_a.body.user_data.on_contact_end(arbiter, shape_b)
        
        if hasattr(shape_b.body, 'user_data') and hasattr(shape_b.body.user_data, 'on_contact_end'):
            shape_b.body.user_data.on_contact_end(arbiter, shape_a)
    
    def clear_events(self):
        """Clear all contact events."""
        self.contact_events.clear()
    
    def get_events(self):
        """Get all contact events and clear them."""
        events = self.contact_events.copy()
        self.clear_events()
        return events 