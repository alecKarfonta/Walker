"""
Population evaluation and analysis.
Tracks fitness distributions, diversity metrics, and evolutionary dynamics.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import math


@dataclass
class PopulationMetrics:
    """Data class to store population evaluation metrics."""
    generation: int
    timestamp: float
    population_size: int
    
    # Fitness Distributions
    fitness_distribution_analysis: Dict[str, float] = field(default_factory=dict)
    pareto_front_analysis: Dict[str, Any] = field(default_factory=dict)
    multi_objective_scores: Dict[str, float] = field(default_factory=dict)
    
    # Diversity Metrics
    genotypic_diversity: float = 0.0
    phenotypic_diversity: float = 0.0
    behavioral_diversity: float = 0.0
    
    # Evolution Effectiveness
    selection_pressure_analysis: Dict[str, float] = field(default_factory=dict)
    mutation_impact_analysis: Dict[str, float] = field(default_factory=dict)
    crossover_success_rates: Dict[str, float] = field(default_factory=dict)
    speciation_dynamics: Dict[str, Any] = field(default_factory=dict)
    
    # Population Health
    age_distribution: List[int] = field(default_factory=list)
    lineage_analysis: Dict[str, Any] = field(default_factory=dict)
    extinction_risk: float = 0.0


class PopulationEvaluator:
    """
    Evaluates population-level dynamics and evolutionary effectiveness.
    Tracks fitness distributions, diversity, and evolutionary processes.
    """
    
    def __init__(self, history_length: int = 100):
        """
        Initialize the population evaluator.
        
        Args:
            history_length: Number of generations to track in history
        """
        self.history_length = history_length
        self.population_metrics: List[PopulationMetrics] = []
        
        # Historical tracking
        self.fitness_histories: Dict[str, List[float]] = defaultdict(list)
        self.diversity_timeline: List[Tuple[int, float]] = []
        self.lineage_tracker: Dict[str, Dict[str, Any]] = {}
        self.species_evolution: Dict[int, List[Dict]] = defaultdict(list)
        
        # Population baselines
        self.initial_diversity = 0.0
        self.peak_diversity = 0.0
        self.baseline_fitness = 0.0
        self.peak_fitness = 0.0
        
    def evaluate_population(self, 
                           agents: List[Any],
                           generation: int,
                           evolution_summary: Dict[str, Any]) -> PopulationMetrics:
        """
        Evaluate population and return comprehensive metrics.
        
        Args:
            agents: List of agents in the population
            generation: Current generation number
            evolution_summary: Summary from evolution engine
            
        Returns:
            PopulationMetrics object with current evaluation
        """
        try:
            timestamp = time.time()
            population_size = len(agents)
            
            # Create new metrics object
            metrics = PopulationMetrics(
                generation=generation,
                timestamp=timestamp,
                population_size=population_size
            )
            
            # Fitness Distribution Analysis
            metrics.fitness_distribution_analysis = self._analyze_fitness_distribution(agents)
            metrics.pareto_front_analysis = self._analyze_pareto_front(agents)
            metrics.multi_objective_scores = self._calculate_multi_objective_scores(agents)
            
            # Diversity Metrics
            metrics.genotypic_diversity = self._calculate_genotypic_diversity(agents)
            metrics.phenotypic_diversity = self._calculate_phenotypic_diversity(agents)
            metrics.behavioral_diversity = self._calculate_behavioral_diversity(agents)
            
            # Evolution Effectiveness
            metrics.selection_pressure_analysis = self._analyze_selection_pressure(agents, evolution_summary)
            metrics.mutation_impact_analysis = self._analyze_mutation_impact(agents)
            metrics.crossover_success_rates = self._analyze_crossover_success(agents)
            metrics.speciation_dynamics = self._analyze_speciation_dynamics(agents, evolution_summary)
            
            # Population Health
            metrics.age_distribution = self._analyze_age_distribution(agents)
            metrics.lineage_analysis = self._analyze_lineage(agents)
            metrics.extinction_risk = self._calculate_extinction_risk(agents, metrics)
            
            # Update historical tracking
            self._update_population_tracking(metrics, agents)
            
            # Store metrics
            self.population_metrics.append(metrics)
            
            # Trim history
            if len(self.population_metrics) > self.history_length:
                self.population_metrics = self.population_metrics[-self.history_length:]
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error evaluating population: {e}")
            return PopulationMetrics(generation=generation, timestamp=time.time(), population_size=len(agents))
    
    def _analyze_fitness_distribution(self, agents: List[Any]) -> Dict[str, float]:
        """Analyze fitness distribution characteristics."""
        try:
            fitnesses = []
            for agent in agents:
                try:
                    if hasattr(agent, 'get_evolutionary_fitness'):
                        fitness = agent.get_evolutionary_fitness()
                    elif hasattr(agent, 'total_reward'):
                        fitness = agent.total_reward
                    else:
                        fitness = 0.0
                    fitnesses.append(fitness)
                except:
                    fitnesses.append(0.0)
            
            if not fitnesses:
                return {}
            
            fitnesses = np.array(fitnesses)
            
            analysis = {
                'mean': float(np.mean(fitnesses)),
                'std': float(np.std(fitnesses)),
                'min': float(np.min(fitnesses)),
                'max': float(np.max(fitnesses)),
                'median': float(np.median(fitnesses)),
                'q25': float(np.percentile(fitnesses, 25)),
                'q75': float(np.percentile(fitnesses, 75)),
                'skewness': float(self._calculate_skewness(fitnesses)),
                'kurtosis': float(self._calculate_kurtosis(fitnesses)),
                'coefficient_variation': float(np.std(fitnesses) / np.mean(fitnesses)) if np.mean(fitnesses) > 0 else 0.0
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing fitness distribution: {e}")
            return {}
    
    def _analyze_pareto_front(self, agents: List[Any]) -> Dict[str, Any]:
        """Analyze Pareto front for multi-objective optimization."""
        try:
            # Extract multiple objectives
            objectives = []
            for agent in agents:
                try:
                    obj = {
                        'fitness': getattr(agent, 'total_reward', 0.0),
                        'efficiency': self._calculate_agent_efficiency(agent),
                        'stability': self._calculate_agent_stability(agent),
                        'exploration': self._calculate_agent_exploration(agent)
                    }
                    objectives.append(obj)
                except:
                    objectives.append({'fitness': 0.0, 'efficiency': 0.0, 'stability': 0.0, 'exploration': 0.0})
            
            if not objectives:
                return {}
            
            # Find Pareto front
            pareto_front = self._find_pareto_front(objectives)
            
            analysis = {
                'pareto_front_size': len(pareto_front),
                'pareto_front_ratio': len(pareto_front) / len(objectives),
                'hypervolume': self._calculate_hypervolume(pareto_front),
                'diversity_on_front': self._calculate_front_diversity(pareto_front)
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing Pareto front: {e}")
            return {}
    
    def _calculate_multi_objective_scores(self, agents: List[Any]) -> Dict[str, float]:
        """Calculate multi-objective performance scores."""
        try:
            scores = {
                'weighted_fitness': 0.0,
                'balanced_performance': 0.0,
                'trade_off_quality': 0.0
            }
            
            if not agents:
                return scores
            
            # Calculate weighted combination of objectives
            total_weighted = 0.0
            total_balanced = 0.0
            
            for agent in agents:
                try:
                    fitness = getattr(agent, 'total_reward', 0.0)
                    efficiency = self._calculate_agent_efficiency(agent)
                    stability = self._calculate_agent_stability(agent)
                    
                    # Weighted score (fitness emphasized)
                    weighted = 0.6 * fitness + 0.2 * efficiency + 0.2 * stability
                    total_weighted += weighted
                    
                    # Balanced score (equal weights)
                    balanced = (fitness + efficiency + stability) / 3.0
                    total_balanced += balanced
                    
                except:
                    pass
            
            if len(agents) > 0:
                scores['weighted_fitness'] = total_weighted / len(agents)
                scores['balanced_performance'] = total_balanced / len(agents)
                scores['trade_off_quality'] = min(scores['weighted_fitness'], scores['balanced_performance'])
            
            return scores
            
        except Exception as e:
            print(f"⚠️  Error calculating multi-objective scores: {e}")
            return {}
    
    def _calculate_genotypic_diversity(self, agents: List[Any]) -> float:
        """Calculate genotypic diversity (parameter space diversity)."""
        try:
            if len(agents) < 2:
                return 0.0
            
            # Extract parameter vectors
            param_vectors = []
            for agent in agents:
                try:
                    if hasattr(agent, 'physical_params'):
                        params = agent.physical_params
                        vector = [
                            getattr(params, 'body_width', 1.0),
                            getattr(params, 'body_height', 0.8),
                            getattr(params, 'arm_length', 0.6),
                            getattr(params, 'motor_torque', 150.0),
                            getattr(params, 'learning_rate', 0.01),
                            getattr(params, 'epsilon', 0.3)
                        ]
                        param_vectors.append(vector)
                except:
                    param_vectors.append([1.0, 0.8, 0.6, 150.0, 0.01, 0.3])  # Default vector
            
            if len(param_vectors) < 2:
                return 0.0
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(param_vectors)):
                for j in range(i + 1, len(param_vectors)):
                    dist = np.linalg.norm(np.array(param_vectors[i]) - np.array(param_vectors[j]))
                    distances.append(dist)
            
            # Diversity as average pairwise distance
            diversity = np.mean(distances) if distances else 0.0
            return float(diversity)
            
        except Exception as e:
            print(f"⚠️  Error calculating genotypic diversity: {e}")
            return 0.0
    
    def _calculate_phenotypic_diversity(self, agents: List[Any]) -> float:
        """Calculate phenotypic diversity (observable trait diversity)."""
        try:
            if len(agents) < 2:
                return 0.0
            
            # Extract phenotypic traits
            traits = []
            for agent in agents:
                try:
                    trait_vector = [
                        getattr(agent, 'total_reward', 0.0),
                        getattr(agent, 'max_speed', 0.0) if hasattr(agent, 'max_speed') else 0.0,
                        self._calculate_agent_stability(agent),
                        self._calculate_agent_exploration(agent)
                    ]
                    traits.append(trait_vector)
                except:
                    traits.append([0.0, 0.0, 0.0, 0.0])
            
            if len(traits) < 2:
                return 0.0
            
            # Normalize traits to same scale
            traits = np.array(traits)
            normalized_traits = (traits - np.mean(traits, axis=0)) / (np.std(traits, axis=0) + 1e-6)
            
            # Calculate diversity as average pairwise distance
            distances = []
            for i in range(len(normalized_traits)):
                for j in range(i + 1, len(normalized_traits)):
                    dist = np.linalg.norm(normalized_traits[i] - normalized_traits[j])
                    distances.append(dist)
            
            diversity = np.mean(distances) if distances else 0.0
            return float(diversity)
            
        except Exception as e:
            print(f"⚠️  Error calculating phenotypic diversity: {e}")
            return 0.0
    
    def _calculate_behavioral_diversity(self, agents: List[Any]) -> float:
        """Calculate behavioral diversity (action pattern diversity)."""
        try:
            if len(agents) < 2:
                return 0.0
            
            # Extract behavioral signatures
            behaviors = []
            for agent in agents:
                try:
                    # Use action history as behavioral signature
                    if hasattr(agent, 'action_history') and agent.action_history:
                        recent_actions = agent.action_history[-10:]  # Last 10 actions
                        # Convert to frequency distribution
                        action_counts = Counter(map(str, recent_actions))
                        behavior_vector = list(action_counts.values())
                        
                        # Normalize
                        total_actions = sum(behavior_vector)
                        if total_actions > 0:
                            behavior_vector = [count / total_actions for count in behavior_vector]
                        
                        behaviors.append(behavior_vector)
                except:
                    behaviors.append([1.0])  # Default uniform behavior
            
            if len(behaviors) < 2:
                return 0.0
            
            # Pad behaviors to same length
            max_len = max(len(b) for b in behaviors)
            padded_behaviors = []
            for behavior in behaviors:
                padded = behavior + [0.0] * (max_len - len(behavior))
                padded_behaviors.append(padded)
            
            # Calculate Jensen-Shannon divergence for behavioral diversity
            diversity_scores = []
            for i in range(len(padded_behaviors)):
                for j in range(i + 1, len(padded_behaviors)):
                    js_div = self._jensen_shannon_divergence(padded_behaviors[i], padded_behaviors[j])
                    diversity_scores.append(js_div)
            
            diversity = np.mean(diversity_scores) if diversity_scores else 0.0
            return float(diversity)
            
        except Exception as e:
            print(f"⚠️  Error calculating behavioral diversity: {e}")
            return 0.0
    
    def _analyze_selection_pressure(self, agents: List[Any], evolution_summary: Dict[str, Any]) -> Dict[str, float]:
        """Analyze selection pressure in the population."""
        try:
            fitnesses = []
            for agent in agents:
                try:
                    fitness = getattr(agent, 'total_reward', 0.0)
                    fitnesses.append(fitness)
                except:
                    fitnesses.append(0.0)
            
            if not fitnesses:
                return {}
            
            fitnesses = np.array(fitnesses)
            
            analysis = {
                'selection_intensity': float(np.std(fitnesses) / np.mean(fitnesses)) if np.mean(fitnesses) > 0 else 0.0,
                'elite_advantage': float(np.max(fitnesses) - np.mean(fitnesses)),
                'selection_differential': float(np.max(fitnesses) - np.min(fitnesses)),
                'effective_population_size': self._calculate_effective_population_size(fitnesses)
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing selection pressure: {e}")
            return {}
    
    def _analyze_mutation_impact(self, agents: List[Any]) -> Dict[str, float]:
        """Analyze impact of mutations on population."""
        try:
            mutation_impacts = []
            total_mutations = 0
            
            for agent in agents:
                try:
                    if hasattr(agent, 'mutation_count'):
                        total_mutations += agent.mutation_count
                        
                        # Estimate mutation impact by comparing with parent fitness
                        if hasattr(agent, 'parent_lineage') and agent.parent_lineage:
                            # This would need parent fitness tracking
                            # For now, use current fitness as proxy
                            impact = getattr(agent, 'total_reward', 0.0) / max(agent.mutation_count, 1)
                            mutation_impacts.append(impact)
                except:
                    pass
            
            analysis = {
                'average_mutation_impact': float(np.mean(mutation_impacts)) if mutation_impacts else 0.0,
                'mutation_benefit_ratio': float(sum(1 for impact in mutation_impacts if impact > 0) / len(mutation_impacts)) if mutation_impacts else 0.0,
                'total_mutations': total_mutations,
                'mutation_diversity': float(np.std(mutation_impacts)) if mutation_impacts else 0.0
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing mutation impact: {e}")
            return {}
    
    def _analyze_crossover_success(self, agents: List[Any]) -> Dict[str, float]:
        """Analyze crossover success rates."""
        try:
            crossover_agents = []
            
            for agent in agents:
                try:
                    if hasattr(agent, 'crossover_count') and agent.crossover_count > 0:
                        fitness = getattr(agent, 'total_reward', 0.0)
                        crossover_agents.append(fitness)
                except:
                    pass
            
            if not crossover_agents:
                return {'crossover_success_rate': 0.0, 'crossover_benefit': 0.0}
            
            # Compare crossover agents with overall population
            all_fitnesses = [getattr(agent, 'total_reward', 0.0) for agent in agents]
            
            analysis = {
                'crossover_success_rate': float(len(crossover_agents) / len(agents)),
                'crossover_benefit': float(np.mean(crossover_agents) - np.mean(all_fitnesses)),
                'crossover_diversity': float(np.std(crossover_agents))
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing crossover success: {e}")
            return {}
    
    def _analyze_speciation_dynamics(self, agents: List[Any], evolution_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze speciation and niche formation."""
        try:
            species_count = evolution_summary.get('species_count', 1)
            
            # Group agents by similarity (simplified speciation)
            species_groups = self._identify_species(agents)
            
            analysis = {
                'species_count': species_count,
                'species_size_distribution': [len(group) for group in species_groups],
                'species_fitness_variance': self._calculate_species_fitness_variance(species_groups),
                'niche_formation_quality': self._assess_niche_formation(species_groups)
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing speciation dynamics: {e}")
            return {}
    
    def _analyze_age_distribution(self, agents: List[Any]) -> List[int]:
        """Analyze age distribution of agents."""
        try:
            ages = []
            for agent in agents:
                try:
                    age = getattr(agent, 'generation', 0)
                    ages.append(age)
                except:
                    ages.append(0)
            
            return ages
            
        except Exception as e:
            print(f"⚠️  Error analyzing age distribution: {e}")
            return []
    
    def _analyze_lineage(self, agents: List[Any]) -> Dict[str, Any]:
        """Analyze lineage and ancestry patterns."""
        try:
            lineage_depths = []
            lineage_diversity = set()
            
            for agent in agents:
                try:
                    if hasattr(agent, 'parent_lineage'):
                        depth = len(agent.parent_lineage)
                        lineage_depths.append(depth)
                        
                        # Track lineage diversity
                        lineage_signature = tuple(agent.parent_lineage[:3])  # First 3 ancestors
                        lineage_diversity.add(lineage_signature)
                except:
                    lineage_depths.append(0)
            
            analysis = {
                'average_lineage_depth': float(np.mean(lineage_depths)) if lineage_depths else 0.0,
                'max_lineage_depth': max(lineage_depths) if lineage_depths else 0,
                'lineage_diversity': len(lineage_diversity),
                'founding_lineages': len([d for d in lineage_depths if d == 0])
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing lineage: {e}")
            return {}
    
    def _calculate_extinction_risk(self, agents: List[Any], metrics: PopulationMetrics) -> float:
        """Calculate extinction risk for the population."""
        try:
            risk_factors = 0
            
            # Low diversity increases extinction risk
            if metrics.genotypic_diversity < 0.5:
                risk_factors += 1
            
            # Low fitness variance suggests convergence issues
            fitness_analysis = metrics.fitness_distribution_analysis
            if fitness_analysis.get('coefficient_variation', 0) < 0.1:
                risk_factors += 1
            
            # Few species suggests lack of niche exploration
            if metrics.speciation_dynamics.get('species_count', 1) < 3:
                risk_factors += 1
            
            # Poor multi-objective performance
            if metrics.multi_objective_scores.get('balanced_performance', 0) < 0.1:
                risk_factors += 1
            
            # High selection pressure can reduce diversity
            selection_analysis = metrics.selection_pressure_analysis
            if selection_analysis.get('selection_intensity', 0) > 2.0:
                risk_factors += 1
            
            # Calculate risk as proportion of factors present
            max_risk_factors = 5
            extinction_risk = risk_factors / max_risk_factors
            
            return float(extinction_risk)
            
        except Exception as e:
            print(f"⚠️  Error calculating extinction risk: {e}")
            return 0.0
    
    def _update_population_tracking(self, metrics: PopulationMetrics, agents: List[Any]):
        """Update historical tracking data."""
        try:
            # Track diversity over time
            self.diversity_timeline.append((metrics.generation, metrics.genotypic_diversity))
            
            # Update peaks
            if metrics.genotypic_diversity > self.peak_diversity:
                self.peak_diversity = metrics.genotypic_diversity
            
            fitness_mean = metrics.fitness_distribution_analysis.get('mean', 0.0)
            if fitness_mean > self.peak_fitness:
                self.peak_fitness = fitness_mean
            
            # Track individual fitness histories
            for agent in agents:
                try:
                    agent_id = str(agent.id)
                    fitness = getattr(agent, 'total_reward', 0.0)
                    self.fitness_histories[agent_id].append(fitness)
                    
                    # Trim individual histories
                    if len(self.fitness_histories[agent_id]) > 50:
                        self.fitness_histories[agent_id] = self.fitness_histories[agent_id][-50:]
                except:
                    pass
            
            # Trim diversity timeline
            if len(self.diversity_timeline) > 200:
                self.diversity_timeline = self.diversity_timeline[-200:]
                
        except Exception as e:
            print(f"⚠️  Error updating population tracking: {e}")
    
    # Helper methods
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_agent_efficiency(self, agent) -> float:
        """Calculate efficiency metric for an agent."""
        try:
            if hasattr(agent, 'physical_params') and hasattr(agent, 'total_reward'):
                motor_power = getattr(agent.physical_params, 'motor_torque', 100)
                performance = agent.total_reward
                if motor_power > 0:
                    return performance / motor_power
            return 0.0
        except:
            return 0.0
    
    def _calculate_agent_stability(self, agent) -> float:
        """Calculate stability metric for an agent."""
        try:
            if hasattr(agent, 'body') and agent.body:
                return max(0, 1.0 - abs(agent.body.angle))
            return 0.0
        except:
            return 0.0
    
    def _calculate_agent_exploration(self, agent) -> float:
        """Calculate exploration metric for an agent."""
        try:
            if hasattr(agent, 'q_table') and hasattr(agent.q_table, 'state_coverage'):
                return float(len(agent.q_table.state_coverage))
            return 0.0
        except:
            return 0.0
    
    def _find_pareto_front(self, objectives: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Find Pareto front from multi-objective data."""
        if not objectives:
            return []
        
        pareto_front = []
        for candidate in objectives:
            is_dominated = False
            for other in objectives:
                if self._dominates(other, candidate):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Check if solution a dominates solution b."""
        better_in_any = False
        for key in a.keys():
            if a[key] < b[key]:  # Assuming minimization
                return False
            elif a[key] > b[key]:
                better_in_any = True
        return better_in_any
    
    def _calculate_hypervolume(self, pareto_front: List[Dict[str, float]]) -> float:
        """Calculate hypervolume of Pareto front."""
        if not pareto_front:
            return 0.0
        
        # Simplified hypervolume calculation
        volumes = []
        for point in pareto_front:
            volume = 1.0
            for value in point.values():
                volume *= max(0, value)
            volumes.append(volume)
        
        return float(np.mean(volumes))
    
    def _calculate_front_diversity(self, pareto_front: List[Dict[str, float]]) -> float:
        """Calculate diversity on Pareto front."""
        if len(pareto_front) < 2:
            return 0.0
        
        # Convert to array for distance calculation
        points = []
        for point in pareto_front:
            points.append(list(point.values()))
        
        points = np.array(points)
        
        # Calculate average pairwise distance
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0
    
    def _jensen_shannon_divergence(self, p: List[float], q: List[float]) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        if len(p) != len(q):
            return 0.0
        
        p = np.array(p) + 1e-10  # Add small epsilon to avoid log(0)
        q = np.array(q) + 1e-10
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        m = 0.5 * (p + q)
        
        def kl_divergence(x, y):
            return np.sum(x * np.log(x / y))
        
        js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        return float(js)
    
    def _calculate_effective_population_size(self, fitnesses: np.ndarray) -> float:
        """Calculate effective population size based on fitness variance."""
        if len(fitnesses) == 0:
            return 0.0
        
        # Effective population size based on fitness variance
        mean_fitness = np.mean(fitnesses)
        variance_fitness = np.var(fitnesses)
        
        if variance_fitness > 0:
            effective_size = (mean_fitness ** 2) / variance_fitness
            return float(min(effective_size, len(fitnesses)))
        
        return float(len(fitnesses))
    
    def _identify_species(self, agents: List[Any]) -> List[List[Any]]:
        """Identify species groups based on similarity."""
        # Simplified species identification
        species_groups = []
        unassigned = agents.copy()
        
        while unassigned:
            # Start new species with first unassigned agent
            species_founder = unassigned.pop(0)
            current_species = [species_founder]
            
            # Find similar agents
            to_remove = []
            for agent in unassigned:
                if self._agents_similar(species_founder, agent):
                    current_species.append(agent)
                    to_remove.append(agent)
            
            # Remove assigned agents
            for agent in to_remove:
                unassigned.remove(agent)
            
            species_groups.append(current_species)
        
        return species_groups
    
    def _agents_similar(self, agent1: Any, agent2: Any, threshold: float = 0.3) -> bool:
        """Check if two agents are similar enough to be in same species."""
        try:
            if hasattr(agent1, 'physical_params') and hasattr(agent2, 'physical_params'):
                params1 = agent1.physical_params
                params2 = agent2.physical_params
                
                # Compare key parameters
                differences = [
                    abs(getattr(params1, 'body_width', 1.0) - getattr(params2, 'body_width', 1.0)),
                    abs(getattr(params1, 'motor_torque', 150.0) - getattr(params2, 'motor_torque', 150.0)) / 100.0,
                    abs(getattr(params1, 'learning_rate', 0.01) - getattr(params2, 'learning_rate', 0.01)) * 100
                ]
                
                avg_difference = np.mean(differences)
                return avg_difference < threshold
            
            return False
        except:
            return False
    
    def _calculate_species_fitness_variance(self, species_groups: List[List[Any]]) -> List[float]:
        """Calculate fitness variance within each species."""
        variances = []
        for species in species_groups:
            fitnesses = [getattr(agent, 'total_reward', 0.0) for agent in species]
            if len(fitnesses) > 1:
                variances.append(float(np.var(fitnesses)))
            else:
                variances.append(0.0)
        return variances
    
    def _assess_niche_formation(self, species_groups: List[List[Any]]) -> float:
        """Assess quality of niche formation."""
        if len(species_groups) <= 1:
            return 0.0
        
        # Good niche formation: balanced species sizes, distinct fitness ranges
        species_sizes = [len(group) for group in species_groups]
        size_balance = 1.0 - (np.std(species_sizes) / np.mean(species_sizes)) if np.mean(species_sizes) > 0 else 0.0
        
        # Fitness distinctiveness
        species_fitnesses = []
        for species in species_groups:
            avg_fitness = np.mean([getattr(agent, 'total_reward', 0.0) for agent in species])
            species_fitnesses.append(avg_fitness)
        
        fitness_distinctiveness = np.std(species_fitnesses) if len(species_fitnesses) > 1 else 0.0
        
        niche_quality = (size_balance + min(fitness_distinctiveness, 1.0)) / 2.0
        return float(niche_quality)
    
    def get_population_summary(self) -> Dict[str, Any]:
        """Get comprehensive population summary."""
        if not self.population_metrics:
            return {}
        
        latest_metrics = self.population_metrics[-1]
        
        return {
            'population_health': self._assess_population_health(),
            'diversity_status': {
                'genotypic_diversity': latest_metrics.genotypic_diversity,
                'phenotypic_diversity': latest_metrics.phenotypic_diversity,
                'behavioral_diversity': latest_metrics.behavioral_diversity,
                'diversity_trend': self._get_diversity_trend()
            },
            'evolutionary_dynamics': {
                'selection_pressure': latest_metrics.selection_pressure_analysis,
                'mutation_effectiveness': latest_metrics.mutation_impact_analysis,
                'speciation_quality': latest_metrics.speciation_dynamics,
                'extinction_risk': latest_metrics.extinction_risk
            },
            'fitness_landscape': {
                'distribution': latest_metrics.fitness_distribution_analysis,
                'pareto_front': latest_metrics.pareto_front_analysis,
                'multi_objective': latest_metrics.multi_objective_scores
            }
        }
    
    def _assess_population_health(self) -> str:
        """Assess overall population health."""
        if not self.population_metrics:
            return "unknown"
        
        latest = self.population_metrics[-1]
        
        health_score = 0
        max_score = 5
        
        # Diversity health
        if latest.genotypic_diversity > 0.5:
            health_score += 1
        
        # Fitness distribution health
        cv = latest.fitness_distribution_analysis.get('coefficient_variation', 0)
        if 0.1 < cv < 2.0:  # Good variance, not too little or too much
            health_score += 1
        
        # Species diversity
        if latest.speciation_dynamics.get('species_count', 1) >= 3:
            health_score += 1
        
        # Low extinction risk
        if latest.extinction_risk < 0.5:
            health_score += 1
        
        # Multi-objective performance
        if latest.multi_objective_scores.get('balanced_performance', 0) > 0.2:
            health_score += 1
        
        health_ratio = health_score / max_score
        
        if health_ratio >= 0.8:
            return "excellent"
        elif health_ratio >= 0.6:
            return "good"
        elif health_ratio >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _get_diversity_trend(self) -> str:
        """Get diversity trend over recent generations."""
        if len(self.diversity_timeline) < 5:
            return "insufficient_data"
        
        recent_diversity = [div for _, div in self.diversity_timeline[-5:]]
        
        if len(recent_diversity) > 2:
            trend = np.polyfit(range(len(recent_diversity)), recent_diversity, 1)[0]
            
            if trend > 0.01:
                return "increasing"
            elif trend < -0.01:
                return "decreasing"
            else:
                return "stable"
        
        return "stable" 