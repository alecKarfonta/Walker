"""
Population management module for Walker project.

This module handles evolutionary algorithms, population management,
and selection strategies for training physics-based robots.
"""

from .population_controller import PopulationController
from .evolution import EvolutionEngine
from .selection import SelectionStrategy, TournamentSelection, RouletteWheelSelection

__all__ = [
    'PopulationController',
    'EvolutionEngine', 
    'SelectionStrategy',
    'TournamentSelection',
    'RouletteWheelSelection'
] 