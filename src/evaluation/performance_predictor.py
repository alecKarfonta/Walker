"""
Performance prediction framework.
Predicts robot training outcomes based on early training metrics.
"""

import numpy as np
import pandas as pd
import time
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PredictionFeatures:
    """Features used for performance prediction."""
    # Early learning metrics (first N steps)
    early_reward_mean: float = 0.0
    early_reward_std: float = 0.0
    early_reward_trend: float = 0.0
    early_convergence_rate: float = 0.0
    early_exploration_efficiency: float = 0.0
    early_q_value_distribution: float = 0.0
    early_action_diversity: float = 0.0
    early_state_coverage: float = 0.0
    
    # Physical parameters
    body_width: float = 1.5
    body_height: float = 0.8
    motor_torque: float = 150.0
    motor_speed: float = 3.0
    wheel_radius: float = 0.4
    
    # Learning parameters
    learning_rate: float = 0.01
    epsilon: float = 0.3
    discount_factor: float = 0.9
    exploration_bonus: float = 0.1
    
    # Environmental factors
    episode_length: int = 1000
    world_complexity: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML models."""
        return {
            'early_reward_mean': self.early_reward_mean,
            'early_reward_std': self.early_reward_std,
            'early_reward_trend': self.early_reward_trend,
            'early_convergence_rate': self.early_convergence_rate,
            'early_exploration_efficiency': self.early_exploration_efficiency,
            'early_q_value_distribution': self.early_q_value_distribution,
            'early_action_diversity': self.early_action_diversity,
            'early_state_coverage': self.early_state_coverage,
            'body_width': self.body_width,
            'body_height': self.body_height,
            'motor_torque': self.motor_torque,
            'motor_speed': self.motor_speed,
            'wheel_radius': self.wheel_radius,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'discount_factor': self.discount_factor,
            'exploration_bonus': self.exploration_bonus,
            'episode_length': float(self.episode_length),
            'world_complexity': self.world_complexity
        }


@dataclass
class PredictionTarget:
    """Target outcomes to predict."""
    final_fitness: float = 0.0
    convergence_time: float = 1000.0
    max_fitness_achieved: float = 0.0
    learning_stability: float = 0.0
    final_success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML models."""
        return {
            'final_fitness': self.final_fitness,
            'convergence_time': self.convergence_time,
            'max_fitness_achieved': self.max_fitness_achieved,
            'learning_stability': self.learning_stability,
            'final_success_rate': self.final_success_rate
        }


@dataclass
class PredictionResult:
    """Results from performance prediction."""
    predicted_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    prediction_accuracy: Dict[str, float]
    feature_importance: Dict[str, float]
    model_performance: Dict[str, float]


class PerformancePredictor:
    """
    Predicts long-term training performance based on early metrics.
    """
    
    def __init__(self, early_steps_ratio: float = 0.1):
        """
        Initialize the performance predictor.
        
        Args:
            early_steps_ratio: Fraction of total training to use for early prediction
        """
        self.early_steps_ratio = early_steps_ratio
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        
        # Training data storage
        self.training_features: List[PredictionFeatures] = []
        self.training_targets: List[PredictionTarget] = []
        
        # Model performance tracking
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importance_history: List[Dict[str, float]] = []
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize prediction models."""
        
        # Different models for different prediction tasks
        self.models = {
            'final_fitness': {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=6
                ),
                'linear': Ridge(alpha=1.0)
            },
            'convergence_time': {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=8
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=5
                ),
                'linear': Ridge(alpha=0.5)
            },
            'max_fitness_achieved': {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=6
                ),
                'linear': Ridge(alpha=1.0)
            },
            'learning_stability': {
                'random_forest': RandomForestRegressor(
                    n_estimators=80, 
                    random_state=42, 
                    max_depth=8
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=80, 
                    random_state=42,
                    max_depth=5
                ),
                'linear': Ridge(alpha=2.0)
            }
        }
        
        # Initialize scalers for each target
        for target_name in self.models.keys():
            self.scalers[target_name] = StandardScaler()
    
    def extract_features_from_agent(self, 
                                   agent, 
                                   step_count: int,
                                   total_steps: int) -> PredictionFeatures:
        """
        Extract prediction features from an agent's current state.
        
        Args:
            agent: The robot agent to extract features from
            step_count: Current training step
            total_steps: Total expected training steps
            
        Returns:
            PredictionFeatures object with extracted features
        """
        features = PredictionFeatures()
        
        try:
            # Extract physical parameters
            if hasattr(agent, 'physical_params'):
                params = agent.physical_params
                features.body_width = getattr(params, 'body_width', 1.5)
                features.body_height = getattr(params, 'body_height', 0.8)
                features.motor_torque = getattr(params, 'motor_torque', 150.0)
                features.motor_speed = getattr(params, 'motor_speed', 3.0)
                features.wheel_radius = getattr(params, 'wheel_radius', 0.4)
                features.learning_rate = getattr(params, 'learning_rate', 0.01)
                features.epsilon = getattr(params, 'epsilon', 0.3)
                features.discount_factor = getattr(params, 'discount_factor', 0.9)
                features.exploration_bonus = getattr(params, 'exploration_bonus', 0.1)
            
            # Extract early learning metrics
            if hasattr(agent, 'recent_rewards') and agent.recent_rewards:
                rewards = list(agent.recent_rewards)
                if rewards:
                    features.early_reward_mean = np.mean(rewards)
                    features.early_reward_std = np.std(rewards)
                    
                    # Calculate trend (slope of recent rewards)
                    if len(rewards) > 5:
                        x = np.arange(len(rewards))
                        z = np.polyfit(x, rewards, 1)
                        features.early_reward_trend = z[0]  # Slope
            
            # Q-learning specific metrics
            if hasattr(agent, 'q_table'):
                if hasattr(agent.q_table, 'get_convergence_estimate'):
                    features.early_convergence_rate = agent.q_table.get_convergence_estimate()
                
                if hasattr(agent.q_table, 'state_coverage'):
                    features.early_state_coverage = len(agent.q_table.state_coverage)
                
                # Q-value distribution
                if hasattr(agent.q_table, 'q_values'):
                    q_values = []
                    for state_actions in agent.q_table.q_values.values():
                        q_values.extend(state_actions.values())
                    if q_values:
                        features.early_q_value_distribution = np.std(q_values)
            
            # Action diversity
            if hasattr(agent, 'action_history') and agent.action_history:
                unique_actions = len(set(map(tuple, agent.action_history)))
                total_actions = len(agent.action_history)
                features.early_action_diversity = unique_actions / max(total_actions, 1)
            
            # Exploration efficiency
            if hasattr(agent, 'total_reward') and hasattr(agent, 'epsilon'):
                exploration_actions = step_count * agent.epsilon
                if exploration_actions > 0:
                    features.early_exploration_efficiency = agent.total_reward / exploration_actions
            
            # Environmental factors
            features.episode_length = getattr(agent, 'episode_length', 1000)
            
        except Exception as e:
            print(f"⚠️  Error extracting features from agent: {e}")
        
        return features
    
    def extract_targets_from_agent(self, 
                                  agent,
                                  final_fitness: float,
                                  convergence_time: float,
                                  fitness_history: List[float]) -> PredictionTarget:
        """
        Extract prediction targets from completed training.
        
        Args:
            agent: The completed robot agent
            final_fitness: Final fitness achieved
            convergence_time: Time to convergence
            fitness_history: History of fitness values during training
            
        Returns:
            PredictionTarget object with target values
        """
        targets = PredictionTarget()
        
        try:
            targets.final_fitness = final_fitness
            targets.convergence_time = convergence_time
            
            if fitness_history:
                targets.max_fitness_achieved = max(fitness_history)
                
                # Calculate learning stability (1 - variance in final 20% of training)
                final_portion = fitness_history[-len(fitness_history)//5:]
                if len(final_portion) > 1:
                    stability = 1.0 - (np.std(final_portion) / (np.mean(final_portion) + 1e-8))
                    targets.learning_stability = max(0.0, stability)
                
                # Calculate success rate (portion of training with positive rewards)
                success_count = sum(1 for f in fitness_history if f > 0.01)  # Success threshold
                targets.final_success_rate = success_count / len(fitness_history)
        
        except Exception as e:
            print(f"⚠️  Error extracting targets: {e}")
        
        return targets
    
    def add_training_data(self, 
                         features: PredictionFeatures, 
                         targets: PredictionTarget) -> None:
        """Add training data to the predictor."""
        self.training_features.append(features)
        self.training_targets.append(targets)
    
    def train_models(self, validation_split: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Train prediction models on collected data.
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of model performance metrics
        """
        if len(self.training_features) < 10:
            print("⚠️  Insufficient training data for model training (need at least 10 samples)")
            return {}
        
        print(f"🤖 Training prediction models on {len(self.training_features)} samples")
        
        # Convert to arrays
        X = np.array([features.to_dict() for features in self.training_features])
        y_dict = {}
        for target in self.training_targets:
            target_dict = target.to_dict()
            for key, value in target_dict.items():
                if key not in y_dict:
                    y_dict[key] = []
                y_dict[key].append(value)
        
        # Convert to numpy arrays
        for key in y_dict:
            y_dict[key] = np.array(y_dict[key])
        
        # Store feature names
        self.feature_names = list(self.training_features[0].to_dict().keys())
        self.target_names = list(y_dict.keys())
        
        # Train models for each target
        results = {}
        
        for target_name, y in y_dict.items():
            if target_name not in self.models:
                continue
            
            print(f"  📈 Training models for {target_name}")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            
            # Scale features
            self.scalers[target_name].fit(X_train)
            X_train_scaled = self.scalers[target_name].transform(X_train)
            X_val_scaled = self.scalers[target_name].transform(X_val)
            
            target_results = {}
            
            # Train each model type
            for model_name, model in self.models[target_name].items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    mae = mean_absolute_error(y_val, y_pred)
                    
                    target_results[model_name] = {
                        'mse': mse,
                        'r2': r2,
                        'mae': mae,
                        'rmse': np.sqrt(mse)
                    }
                    
                except Exception as e:
                    print(f"    ❌ Error training {model_name} for {target_name}: {e}")
                    continue
            
            results[target_name] = target_results
            self.model_metrics[target_name] = target_results
        
        # Extract feature importance from best models
        self._extract_feature_importance()
        
        print(f"✅ Model training complete for {len(results)} targets")
        return results
    
    def _extract_feature_importance(self) -> None:
        """Extract feature importance from trained models."""
        
        feature_importance = defaultdict(float)
        
        for target_name, models in self.models.items():
            if target_name not in self.model_metrics:
                continue
            
            # Use the best performing model for feature importance
            best_model_name = max(
                self.model_metrics[target_name].keys(),
                key=lambda k: self.model_metrics[target_name][k].get('r2', 0)
            )
            
            best_model = models[best_model_name]
            
            # Extract importance based on model type
            if hasattr(best_model, 'feature_importances_'):
                # Tree-based models
                importances = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                # Linear models
                importances = np.abs(best_model.coef_)
            else:
                continue
            
            # Add to overall importance
            for i, importance in enumerate(importances):
                if i < len(self.feature_names):
                    feature_importance[self.feature_names[i]] += importance
        
        # Normalize
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] /= total_importance
        
        self.feature_importance_history.append(dict(feature_importance))
    
    def predict_performance(self, features: PredictionFeatures) -> PredictionResult:
        """
        Predict performance based on early training features.
        
        Args:
            features: Early training features
            
        Returns:
            PredictionResult with predictions and confidence intervals
        """
        if not self.models or not any(self.model_metrics.values()):
            return PredictionResult(
                predicted_values={},
                confidence_intervals={},
                prediction_accuracy={},
                feature_importance={},
                model_performance={}
            )
        
        # Convert features to array
        X = np.array([features.to_dict()])
        
        predictions = {}
        confidence_intervals = {}
        prediction_accuracy = {}
        
        for target_name, models in self.models.items():
            if target_name not in self.model_metrics or target_name not in self.scalers:
                continue
            
            # Find best model for this target
            best_model_name = max(
                self.model_metrics[target_name].keys(),
                key=lambda k: self.model_metrics[target_name][k].get('r2', 0)
            )
            
            best_model = models[best_model_name]
            
            try:
                # Scale features
                X_scaled = self.scalers[target_name].transform(X)
                
                # Make prediction
                pred = best_model.predict(X_scaled)[0]
                predictions[target_name] = float(pred)
                
                # Estimate confidence interval (simplified)
                model_rmse = self.model_metrics[target_name][best_model_name]['rmse']
                confidence_intervals[target_name] = (
                    float(pred - 1.96 * model_rmse),
                    float(pred + 1.96 * model_rmse)
                )
                
                # Prediction accuracy is R²
                prediction_accuracy[target_name] = self.model_metrics[target_name][best_model_name]['r2']
                
            except Exception as e:
                print(f"⚠️  Error predicting {target_name}: {e}")
                continue
        
        # Get latest feature importance
        feature_importance = {}
        if self.feature_importance_history:
            feature_importance = self.feature_importance_history[-1]
        
        # Overall model performance
        model_performance = {}
        for target_name, metrics in self.model_metrics.items():
            if metrics:
                best_r2 = max(m.get('r2', 0) for m in metrics.values())
                model_performance[target_name] = best_r2
        
        return PredictionResult(
            predicted_values=predictions,
            confidence_intervals=confidence_intervals,
            prediction_accuracy=prediction_accuracy,
            feature_importance=feature_importance,
            model_performance=model_performance
        )
    
    def predict_early_stopping(self, 
                              features: PredictionFeatures,
                              performance_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Predict if training should be stopped early based on projected performance.
        
        Args:
            features: Current training features
            performance_threshold: Minimum acceptable final performance
            
        Returns:
            Dictionary with early stopping recommendation
        """
        prediction_result = self.predict_performance(features)
        
        if 'final_fitness' not in prediction_result.predicted_values:
            return {
                'should_stop': False,
                'reason': 'Unable to predict final fitness',
                'confidence': 0.0
            }
        
        predicted_fitness = prediction_result.predicted_values['final_fitness']
        prediction_confidence = prediction_result.prediction_accuracy.get('final_fitness', 0.0)
        
        # Conservative early stopping - only stop if we're confident it won't reach threshold
        confidence_interval = prediction_result.confidence_intervals.get('final_fitness', (0, 0))
        upper_bound = confidence_interval[1]
        
        should_stop = (predicted_fitness < performance_threshold and 
                      upper_bound < performance_threshold and 
                      prediction_confidence > 0.7)
        
        reason = ""
        if should_stop:
            reason = f"Predicted final fitness ({predicted_fitness:.3f}) unlikely to reach threshold ({performance_threshold:.3f})"
        elif predicted_fitness < performance_threshold:
            reason = f"Predicted fitness low but confidence interval includes possibility of success"
        else:
            reason = f"Predicted fitness ({predicted_fitness:.3f}) meets threshold"
        
        return {
            'should_stop': should_stop,
            'reason': reason,
            'confidence': prediction_confidence,
            'predicted_fitness': predicted_fitness,
            'confidence_interval': confidence_interval,
            'threshold': performance_threshold
        }
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """Generate report on feature importance across all models."""
        
        if not self.feature_importance_history:
            return {"error": "No feature importance data available"}
        
        # Average importance across all training iterations
        avg_importance = defaultdict(float)
        for importance_dict in self.feature_importance_history:
            for feature, importance in importance_dict.items():
                avg_importance[feature] += importance
        
        # Normalize
        n_iterations = len(self.feature_importance_history)
        for feature in avg_importance:
            avg_importance[feature] /= n_iterations
        
        # Sort by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        report = {
            'most_important_features': sorted_features[:10],
            'least_important_features': sorted_features[-5:],
            'feature_categories': {
                'early_learning_metrics': [f for f, _ in sorted_features if f.startswith('early_')],
                'physical_parameters': [f for f, _ in sorted_features if f in ['body_width', 'body_height', 'motor_torque', 'motor_speed', 'wheel_radius']],
                'learning_parameters': [f for f, _ in sorted_features if f in ['learning_rate', 'epsilon', 'discount_factor', 'exploration_bonus']]
            },
            'insights': self._generate_feature_insights(sorted_features)
        }
        
        return report
    
    def _generate_feature_insights(self, sorted_features: List[Tuple[str, float]]) -> List[str]:
        """Generate insights from feature importance analysis."""
        
        insights = []
        
        if not sorted_features:
            return ["No feature data available for insights"]
        
        # Most important feature
        most_important = sorted_features[0]
        insights.append(f"Most predictive feature: {most_important[0]} (importance: {most_important[1]:.3f})")
        
        # Category analysis
        early_features = [f for f, i in sorted_features if f.startswith('early_')]
        physical_features = [f for f, i in sorted_features if f in ['body_width', 'body_height', 'motor_torque', 'motor_speed']]
        learning_features = [f for f, i in sorted_features if f in ['learning_rate', 'epsilon', 'discount_factor']]
        
        if early_features:
            insights.append(f"Early learning metrics are highly predictive (top: {early_features[0]})")
        
        if physical_features and physical_features[0] in [f for f, _ in sorted_features[:5]]:
            insights.append("Physical parameters significantly impact final performance")
        
        if learning_features and learning_features[0] in [f for f, _ in sorted_features[:5]]:
            insights.append("Learning algorithm parameters are critical for success")
        
        return insights
    
    def save_models(self, filepath: str) -> None:
        """Save trained models to file."""
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_metrics': self.model_metrics,
            'feature_importance_history': self.feature_importance_history,
            'early_steps_ratio': self.early_steps_ratio
        }
        
        try:
            joblib.dump(model_data, filepath)
            print(f"✅ Models saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from file."""
        
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.target_names = model_data['target_names']
            self.model_metrics = model_data['model_metrics']
            self.feature_importance_history = model_data['feature_importance_history']
            self.early_steps_ratio = model_data['early_steps_ratio']
            
            print(f"✅ Models loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")


# Mock classes for testing
class MockPredictiveAgent:
    """Mock agent for testing performance prediction."""
    
    def __init__(self, **params):
        self.params = params
        self.recent_rewards = []
        self.action_history = []
        self.total_reward = 0.0
        self.q_table = MockQTable()
        
        # Simulate physical parameters
        class MockPhysicalParams:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                # Set defaults
                self.body_width = kwargs.get('body_width', 1.5)
                self.motor_torque = kwargs.get('motor_torque', 150.0)
                self.learning_rate = kwargs.get('learning_rate', 0.01)
                self.epsilon = kwargs.get('epsilon', 0.3)
                
        self.physical_params = MockPhysicalParams(**params)
        
        # Simulate learning progress
        self._simulate_learning()
    
    def _simulate_learning(self):
        """Simulate realistic learning progression."""
        # Generate realistic reward progression
        base_performance = np.random.uniform(0.5, 1.5)
        for i in range(50):
            # Simulate improvement over time with noise
            progress = min(1.0, i / 30.0)  # Learning curve
            reward = base_performance * progress + np.random.normal(0, 0.1)
            self.recent_rewards.append(reward)
            self.total_reward += reward
            
            # Add random actions
            action = np.random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
            self.action_history.append(action)


class MockQTable:
    """Mock Q-table for testing."""
    
    def __init__(self):
        self.state_coverage = set()
        self.q_values = {}
        
        # Add some mock states
        for i in range(np.random.randint(10, 50)):
            state = tuple(np.random.randint(0, 10, 3))
            self.state_coverage.add(state)
            self.q_values[state] = {a: np.random.normal(0, 1) for a in range(6)}
    
    def get_convergence_estimate(self):
        """Mock convergence estimate."""
        return np.random.uniform(0.3, 0.9) 