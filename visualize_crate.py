#!/usr/bin/env python3
"""
Simple visualization of the CrawlingCrate to debug its behavior.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from src.agents.crawling_crate import CrawlingCrate


def create_flat_world():
    """Create a flat ground in a Pymunk space."""
    space = pymunk.Space()
    space.gravity = (0, -9.8)
    # Flat ground
    static_body = space.static_body
    ground = pymunk.Segment(static_body, (-100, 0), (100, 0), 1.0)
    ground.friction = 1.0
    space.add(ground)
    return space


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    clock = pygame.time.Clock()
    
    # Initialize Pymunk
    space = create_flat_world()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # Create the crawling crate
    agent = CrawlingCrate(space, position=(600, 400))
    
    # Font for text
    font = pygame.font.Font(None, 24)
    
    step = 0
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset
                    agent.reset()
                    step = 0
                elif event.key == pygame.K_SPACE:
                    # Toggle arm movement
                    pass
        
        # Apply different crawling patterns based on step
        if step < 100:
            # Alternate arms
            action = (8.0, -8.0) if step % 20 < 10 else (-8.0, 8.0)
        elif step < 200:
            # One arm at a time
            action = (10.0, 0.0) if step % 30 < 15 else (0.0, 10.0)
        else:
            # Both arms together
            action = (12.0, 12.0) if step % 30 < 15 else (-12.0, -12.0)
        
        agent.apply_action(action)
        agent.step(1/60.0)
        
        # Clear screen
        screen.fill((255, 255, 255))
        
        # Draw the physics world
        space.debug_draw(draw_options)
        
        # Get debug info
        debug = agent.get_debug_info()
        
        # Draw debug text
        texts = [
            f"Step: {step}",
            f"Position: ({debug['crate_pos'][0]:.1f}, {debug['crate_pos'][1]:.1f})",
            f"Velocity: ({debug['crate_vel'][0]:.1f}, {debug['crate_vel'][1]:.1f})",
            f"Crate Angle: {debug['crate_angle']:.2f}",
            f"Left Arm Angle: {debug['left_arm_angle']:.2f}",
            f"Right Arm Angle: {debug['right_arm_angle']:.2f}",
            f"Left Motor: {debug['left_motor_rate']:.1f}",
            f"Right Motor: {debug['right_motor_rate']:.1f}",
            f"Left Contact: {debug['left_contact']}",
            f"Right Contact: {debug['right_contact']}",
            f"Action: {action}",
        ]
        
        for i, text in enumerate(texts):
            surface = font.render(text, True, (0, 0, 0))
            screen.blit(surface, (10, 10 + i * 25))
        
        # Instructions
        instructions = [
            "R: Reset",
            "SPACE: Toggle movement",
            "ESC: Quit"
        ]
        for i, text in enumerate(instructions):
            surface = font.render(text, True, (100, 100, 100))
            screen.blit(surface, (10, 700 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
        step += 1
    
    pygame.quit()


if __name__ == "__main__":
    main() 