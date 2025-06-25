"""
Training progress evaluation and monitoring.
Tracks overall training effectiveness, stability, and resource utilization.
"""

import numpy as np
import time
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Data class to store training progress metrics."""
    timestamp: float
    
    # Population Level
    population_fitness_trend: List[float] = field(default_factory=list)
    diversity_maintenance: float = 0.0
    species_formation: int = 0
    elite_preservation: float = 0.0
    
    # Training Stability
    training_variance: float = 0.0
    catastrophic_forgetting_rate: float = 0.0
    knowledge_transfer_efficiency: float = 0.0
    
    # Resource Utilization
    computational_efficiency: float = 0.0
    memory_usage_optimization: float = 0.0
    training_time_per_improvement: float = 0.0
    
    # Performance Indicators
    convergence_speed: float = 0.0
    plateau_detection: bool = False
    improvement_rate: float = 0.0
    
    # System Health
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    training_fps: float = 0.0


class TrainingProgressEvaluator:
    """
    Evaluates overall training progress and system health.
    Monitors training effectiveness, stability, and resource usage.
    """
    
    def __init__(self, history_length: int = 500):
        """
        Initialize the training progress evaluator.
        
        Args:
            history_length: Number of historical data points to maintain
        """
        self.history_length = history_length
        self.training_metrics: List[TrainingMetrics] = []
        
        # Historical tracking
        self.generation_history: List[Dict[str, Any]] = []
        self.fitness_history: deque = deque(maxlen=200)
        self.diversity_history: deque = deque(maxlen=200)
        self.timing_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.last_evaluation_time = time.time()
        self.total_training_time = 0.0
        self.training_start_time = time.time()
        
        # System monitoring
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance baselines
        self.baseline_fitness = 0.0
        self.baseline_diversity = 0.0
        self.peak_fitness = 0.0
        self.peak_diversity = 0.0
        
    def evaluate_training_progress(self, 
                                 population_stats: Dict[str, Any],
                                 generation: int,
                                 training_step: int) -> TrainingMetrics:
        """
        Evaluate overall training progress and return comprehensive metrics.
        
        Args:
            population_stats: Current population statistics
            generation: Current generation number
            training_step: Current training step
            
        Returns:
            TrainingMetrics object with current evaluation
        """
        try:
            current_time = time.time()
            
            # Update historical tracking
            self._update_training_tracking(population_stats, generation, current_time)
            
            # Create new metrics object
            metrics = TrainingMetrics(timestamp=current_time)
            
            # Population Level Metrics
            metrics.population_fitness_trend = self._calculate_fitness_trend()
            metrics.diversity_maintenance = self._calculate_diversity_maintenance(population_stats)
            metrics.species_formation = self._calculate_species_formation(population_stats)
            metrics.elite_preservation = self._calculate_elite_preservation(population_stats)
            
            # Training Stability Metrics
            metrics.training_variance = self._calculate_training_variance()
            metrics.catastrophic_forgetting_rate = self._calculate_forgetting_rate()
            metrics.knowledge_transfer_efficiency = self._calculate_transfer_efficiency()
            
            # Resource Utilization Metrics
            metrics.computational_efficiency = self._calculate_computational_efficiency()
            metrics.memory_usage_optimization = self._calculate_memory_efficiency()
            metrics.training_time_per_improvement = self._calculate_time_per_improvement()
            
            # Performance Indicators
            metrics.convergence_speed = self._calculate_convergence_speed()
            metrics.plateau_detection = self._detect_training_plateau()
            metrics.improvement_rate = self._calculate_improvement_rate()
            
            # System Health
            metrics.cpu_usage = self._get_cpu_usage()
            metrics.memory_usage = self._get_memory_usage()
            metrics.training_fps = self._calculate_training_fps(current_time)
            
            # Store metrics
            self.training_metrics.append(metrics)
            
            # Trim history
            if len(self.training_metrics) > self.history_length:
                self.training_metrics = self.training_metrics[-self.history_length:]
            
            self.last_evaluation_time = current_time
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error evaluating training progress: {e}")
            return TrainingMetrics(timestamp=time.time())
    
    def _update_training_tracking(self, population_stats: Dict[str, Any], generation: int, current_time: float):
        """Update historical tracking data."""
        try:
            # Store generation data
            generation_data = {
                'generation': generation,
                'timestamp': current_time,
                'best_fitness': population_stats.get('best_fitness', 0.0),
                'average_fitness': population_stats.get('average_fitness', 0.0),
                'diversity': population_stats.get('diversity', 0.0),
                'total_agents': population_stats.get('total_agents', 0)
            }
            self.generation_history.append(generation_data)
            
            # Update fitness and diversity history
            best_fitness = population_stats.get('best_fitness', 0.0)
            avg_fitness = population_stats.get('average_fitness', 0.0)
            diversity = population_stats.get('diversity', 0.0)
            
            self.fitness_history.append(best_fitness)
            self.diversity_history.append(diversity)
            
            # Update peaks
            if best_fitness > self.peak_fitness:
                self.peak_fitness = best_fitness
            if diversity > self.peak_diversity:
                self.peak_diversity = diversity
            
            # Store timing information
            if len(self.generation_history) > 1:
                time_diff = current_time - self.generation_history[-2]['timestamp']
                self.timing_history.append(time_diff)
            
            # Trim generation history
            if len(self.generation_history) > 200:
                self.generation_history = self.generation_history[-200:]
                
        except Exception as e:
            print(f"⚠️  Error updating training tracking: {e}")
    
    def _calculate_fitness_trend(self) -> List[float]:
        """Calculate fitness trend over recent generations."""
        try:
            if len(self.fitness_history) < 5:
                return [0.0]
            
            recent_fitness = list(self.fitness_history)[-10:]
            
            # Calculate trend using linear regression
            if len(recent_fitness) > 2:
                x = np.arange(len(recent_fitness))
                trend = np.polyfit(x, recent_fitness, 1)[0]
                return [float(trend)]
            
            return [0.0]
        except:
            return [0.0]
    
    def _calculate_diversity_maintenance(self, population_stats: Dict[str, Any]) -> float:
        """Calculate how well diversity is maintained."""
        try:
            current_diversity = population_stats.get('diversity', 0.0)
            
            if len(self.diversity_history) > 10:
                recent_diversity = list(self.diversity_history)[-10:]
                avg_diversity = np.mean(recent_diversity)
                
                # Diversity maintenance score
                if self.peak_diversity > 0:
                    maintenance_score = avg_diversity / self.peak_diversity
                    return float(maintenance_score)
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_species_formation(self, population_stats: Dict[str, Any]) -> int:
        """Calculate number of species formed."""
        try:
            return int(population_stats.get('species_count', 1))
        except:
            return 1
    
    def _calculate_elite_preservation(self, population_stats: Dict[str, Any]) -> float:
        """Calculate elite preservation effectiveness."""
        try:
            if len(self.fitness_history) < 10:
                return 0.0
            
            # Check if best fitness is preserved over generations
            recent_best = list(self.fitness_history)[-10:]
            
            if len(recent_best) > 1:
                preserved_generations = sum(1 for i in range(1, len(recent_best)) 
                                          if recent_best[i] >= recent_best[i-1])
                preservation_rate = preserved_generations / (len(recent_best) - 1)
                return float(preservation_rate)
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_training_variance(self) -> float:
        """Calculate training variance/stability."""
        try:
            if len(self.fitness_history) < 10:
                return 0.0
            
            recent_fitness = list(self.fitness_history)[-20:]
            variance = np.var(recent_fitness)
            
            # Normalize variance by mean fitness
            mean_fitness = np.mean(recent_fitness)
            if mean_fitness > 0:
                normalized_variance = variance / (mean_fitness ** 2)
                return float(normalized_variance)
            
            return float(variance)
        except:
            return 0.0
    
    def _calculate_forgetting_rate(self) -> float:
        """Calculate catastrophic forgetting rate."""
        try:
            if len(self.fitness_history) < 20:
                return 0.0
            
            # Look for significant drops in performance
            recent_fitness = list(self.fitness_history)[-20:]
            
            forgetting_events = 0
            for i in range(1, len(recent_fitness)):
                if recent_fitness[i] < recent_fitness[i-1] * 0.9:  # 10% drop threshold
                    forgetting_events += 1
            
            forgetting_rate = forgetting_events / (len(recent_fitness) - 1)
            return float(forgetting_rate)
        except:
            return 0.0
    
    def _calculate_transfer_efficiency(self) -> float:
        """Calculate knowledge transfer efficiency."""
        try:
            if len(self.generation_history) < 5:
                return 0.0
            
            # Measure how quickly fitness improves in new generations
            recent_generations = self.generation_history[-10:]
            
            if len(recent_generations) > 2:
                initial_fitness = recent_generations[0]['best_fitness']
                final_fitness = recent_generations[-1]['best_fitness']
                generations_elapsed = len(recent_generations)
                
                if generations_elapsed > 0 and initial_fitness > 0:
                    improvement_per_generation = (final_fitness - initial_fitness) / generations_elapsed
                    efficiency = improvement_per_generation / initial_fitness
                    return float(max(0, efficiency))
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_computational_efficiency(self) -> float:
        """Calculate computational efficiency."""
        try:
            if len(self.timing_history) < 5:
                return 0.0
            
            recent_times = list(self.timing_history)[-10:]
            avg_time_per_generation = np.mean(recent_times)
            
            if len(self.fitness_history) >= len(recent_times):
                recent_fitness_improvements = []
                fitness_list = list(self.fitness_history)
                
                for i in range(len(recent_times)):
                    if i > 0:
                        improvement = fitness_list[-(i)] - fitness_list[-(i+1)]
                        recent_fitness_improvements.append(max(0, improvement))
                
                if recent_fitness_improvements and avg_time_per_generation > 0:
                    avg_improvement = np.mean(recent_fitness_improvements)
                    efficiency = avg_improvement / avg_time_per_generation
                    return float(efficiency)
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory usage efficiency."""
        try:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - self.baseline_memory
            
            # Memory efficiency relative to training progress
            if self.peak_fitness > self.baseline_fitness:
                fitness_improvement = self.peak_fitness - self.baseline_fitness
                if memory_growth > 0:
                    efficiency = fitness_improvement / memory_growth
                    return float(efficiency)
            
            return 1.0  # Default efficiency if no growth
        except:
            return 0.0
    
    def _calculate_time_per_improvement(self) -> float:
        """Calculate training time per fitness improvement."""
        try:
            if len(self.generation_history) < 5:
                return 0.0
            
            recent_generations = self.generation_history[-10:]
            
            if len(recent_generations) > 2:
                initial_fitness = recent_generations[0]['best_fitness']
                final_fitness = recent_generations[-1]['best_fitness']
                time_elapsed = recent_generations[-1]['timestamp'] - recent_generations[0]['timestamp']
                
                fitness_improvement = final_fitness - initial_fitness
                if fitness_improvement > 0:
                    time_per_improvement = time_elapsed / fitness_improvement
                    return float(time_per_improvement)
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_convergence_speed(self) -> float:
        """Calculate convergence speed."""
        try:
            if len(self.fitness_history) < 10:
                return 0.0
            
            recent_fitness = list(self.fitness_history)[-10:]
            
            # Calculate rate of change in fitness
            if len(recent_fitness) > 2:
                x = np.arange(len(recent_fitness))
                slope = np.polyfit(x, recent_fitness, 1)[0]
                
                # Normalize by current fitness level
                current_fitness = recent_fitness[-1]
                if current_fitness > 0:
                    normalized_slope = slope / current_fitness
                    return float(max(0, normalized_slope))
            
            return 0.0
        except:
            return 0.0
    
    def _detect_training_plateau(self) -> bool:
        """Detect if training has plateaued."""
        try:
            if len(self.fitness_history) < 20:
                return False
            
            recent_fitness = list(self.fitness_history)[-20:]
            
            # Check if recent improvements are very small
            if len(recent_fitness) > 10:
                recent_improvement = recent_fitness[-1] - recent_fitness[-10]
                improvement_threshold = 0.001  # Very small improvement threshold
                
                return recent_improvement < improvement_threshold
            
            return False
        except:
            return False
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate overall improvement rate."""
        try:
            if len(self.fitness_history) < 5:
                return 0.0
            
            current_fitness = list(self.fitness_history)[-1]
            initial_fitness = list(self.fitness_history)[0]
            
            total_time = time.time() - self.training_start_time
            
            if total_time > 0 and initial_fitness >= 0:
                total_improvement = current_fitness - initial_fitness
                improvement_rate = total_improvement / total_time
                return float(improvement_rate)
            
            return 0.0
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        try:
            return float(self.process.cpu_percent())
        except:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return float(self.process.memory_info().rss / 1024 / 1024)
        except:
            return 0.0
    
    def _calculate_training_fps(self, current_time: float) -> float:
        """Calculate training FPS (generations per second)."""
        try:
            if len(self.timing_history) > 0:
                avg_time_per_generation = np.mean(list(self.timing_history))
                if avg_time_per_generation > 0:
                    fps = 1.0 / avg_time_per_generation
                    return float(fps)
            return 0.0
        except:
            return 0.0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_metrics:
            return {}
        
        latest_metrics = self.training_metrics[-1]
        
        return {
            'training_health': self._assess_training_health(),
            'performance_trends': {
                'fitness_trend': latest_metrics.population_fitness_trend[-1] if latest_metrics.population_fitness_trend else 0.0,
                'convergence_speed': latest_metrics.convergence_speed,
                'improvement_rate': latest_metrics.improvement_rate,
                'plateau_detected': latest_metrics.plateau_detection
            },
            'stability_metrics': {
                'training_variance': latest_metrics.training_variance,
                'elite_preservation': latest_metrics.elite_preservation,
                'diversity_maintenance': latest_metrics.diversity_maintenance,
                'forgetting_rate': latest_metrics.catastrophic_forgetting_rate
            },
            'efficiency_metrics': {
                'computational_efficiency': latest_metrics.computational_efficiency,
                'memory_efficiency': latest_metrics.memory_usage_optimization,
                'time_per_improvement': latest_metrics.training_time_per_improvement
            },
            'system_health': {
                'cpu_usage': latest_metrics.cpu_usage,
                'memory_usage': latest_metrics.memory_usage,
                'training_fps': latest_metrics.training_fps
            }
        }
    
    def _assess_training_health(self) -> str:
        """Assess overall training health."""
        if not self.training_metrics:
            return "unknown"
        
        latest = self.training_metrics[-1]
        
        issues = 0
        
        # Check for issues
        if latest.plateau_detection:
            issues += 1
        if latest.catastrophic_forgetting_rate > 0.3:
            issues += 1
        if latest.training_variance > 1.0:
            issues += 1
        if latest.memory_usage > 2000:  # 2GB threshold
            issues += 1
        if latest.cpu_usage > 90:
            issues += 1
        
        if issues == 0:
            return "excellent"
        elif issues <= 1:
            return "good"
        elif issues <= 2:
            return "fair"
        else:
            return "poor"
    
    def get_recommendations(self) -> List[str]:
        """Get training optimization recommendations."""
        if not self.training_metrics:
            return ["Insufficient data for recommendations"]
        
        latest = self.training_metrics[-1]
        recommendations = []
        
        if latest.plateau_detection:
            recommendations.append("Training has plateaued. Consider adjusting learning parameters or adding curriculum learning.")
        
        if latest.catastrophic_forgetting_rate > 0.2:
            recommendations.append("High forgetting rate detected. Consider more conservative learning rates or experience replay.")
        
        if latest.diversity_maintenance < 0.5:
            recommendations.append("Diversity is declining. Consider increasing mutation rates or population size.")
        
        if latest.memory_usage > 1500:  # 1.5GB threshold
            recommendations.append("High memory usage. Consider Q-table pruning or reducing population size.")
        
        if latest.computational_efficiency < 0.001:
            recommendations.append("Low computational efficiency. Consider optimizing hyperparameters or training schedule.")
        
        if latest.training_variance > 0.5:
            recommendations.append("High training variance. Consider reducing learning rates or increasing training stability.")
        
        if not recommendations:
            recommendations.append("Training is progressing well. Continue current configuration.")
        
        return recommendations 