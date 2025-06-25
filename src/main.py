"""
Main entry point for the Walker Python project.
"""

import pygame
import sys
import pymunk
import pymunk.pygame_util
from physics.world import WorldController
from physics.body_factory import BodyFactory


def main():
    """Main function to run the physics simulation."""
    # Initialize Pygame
    pygame.init()
    
    # Set up display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Walker Physics Test")
    clock = pygame.time.Clock()
    
    # Set up Pymunk drawing options
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # Create physics world
    world = WorldController()
    
    # Create a test ball
    ball = BodyFactory.create_ball(
        world.space,
        position=(0, 10),
        radius=5,
        density=1.0,
        friction=0.7,
        restitution=0.8
    )
    
    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update physics
        world.update(1.0 / 60.0)
        
        # Clear screen
        screen.fill((255, 255, 255))
        
        # Draw physics objects
        world.space.debug_draw(draw_options)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main() 