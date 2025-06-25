"""Dynamic Gene Expression System for Enhanced Evolution"""

import random
import math
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

class ExpressionLevel(Enum):
    """Gene expression levels"""
    SUPPRESSED = 0.0
    LOW = 0.25
    MODERATE = 0.5
    HIGH = 0.75
    OVEREXPRESSED = 1.0

class TriggerType(Enum):
    """Environmental triggers for gene expression"""
    TEMPERATURE = "temperature"
    STRESS = "stress"
    COMPETITION = "competition"
    COOPERATION = "cooperation"
    RESOURCE_SCARCITY = "resource_scarcity"
    POPULATION_DENSITY = "population_density"
    SEASON = "season"

@dataclass
class Gene:
    """Represents a gene with expression capabilities"""
    name: str
    base_value: float
    expression_level: float = 1.0
    triggers: Dict[TriggerType, Tuple[float, float]] = field(default_factory=dict)  # (threshold, sensitivity)
    dominant: bool = True
    mutation_rate: float = 0.1
    
    def get_expressed_value(self) -> float:
        """Get the gene's value modified by expression level"""
        return self.base_value * self.expression_level
    
    def update_expression(self, environmental_signals: Dict[TriggerType, float]):
        """Update gene expression based on environmental signals"""
        total_modifier = 1.0
        
        for trigger_type, (threshold, sensitivity) in self.triggers.items():
            if trigger_type in environmental_signals:
                signal_strength = environmental_signals[trigger_type]
                
                # Calculate expression modifier based on signal
                if signal_strength > threshold:
                    # Upregulate
                    modifier = 1.0 + (signal_strength - threshold) * sensitivity
                    total_modifier *= modifier
                elif signal_strength < threshold:
                    # Downregulate
                    modifier = 1.0 - (threshold - signal_strength) * sensitivity * 0.5
                    total_modifier *= max(0.1, modifier)
        
        # Apply modifier to expression level
        self.expression_level = max(0.1, min(2.0, total_modifier))

@dataclass
class GeneNetwork:
    """Collection of genes with interaction capabilities"""
    genes: Dict[str, Gene] = field(default_factory=dict)
    gene_interactions: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)  # gene -> [(target, strength)]
    network_state: Dict[str, float] = field(default_factory=dict)
    
    def add_gene(self, gene: Gene):
        """Add a gene to the network"""
        self.genes[gene.name] = gene
        self.network_state[gene.name] = gene.expression_level
    
    def add_interaction(self, regulator_gene: str, target_gene: str, strength: float):
        """Add regulatory interaction between genes"""
        if regulator_gene not in self.gene_interactions:
            self.gene_interactions[regulator_gene] = []
        self.gene_interactions[regulator_gene].append((target_gene, strength))
    
    def update_network(self, environmental_signals: Dict[TriggerType, float]):
        """Update entire gene network based on environmental signals"""
        
        # First, update individual gene expressions
        for gene in self.genes.values():
            gene.update_expression(environmental_signals)
        
        # Then, apply gene-gene interactions
        new_expressions = {}
        for gene_name, gene in self.genes.items():
            expression = gene.expression_level
            
            # Apply regulatory effects from other genes
            if gene_name in self.gene_interactions:
                for target_gene, strength in self.gene_interactions[gene_name]:
                    if target_gene in self.genes:
                        # Regulatory effect proportional to regulator's expression
                        regulatory_effect = gene.expression_level * strength * 0.1
                        self.genes[target_gene].expression_level *= (1 + regulatory_effect)
            
            new_expressions[gene_name] = self.genes[gene_name].expression_level
        
        # Update network state
        self.network_state = new_expressions
        
        # Ensure expression levels stay within bounds
        for gene in self.genes.values():
            gene.expression_level = max(0.1, min(2.0, gene.expression_level))
    
    def get_phenotype(self) -> Dict[str, float]:
        """Get current phenotype based on gene expression"""
        phenotype = {}
        for gene_name, gene in self.genes.items():
            phenotype[gene_name] = gene.get_expressed_value()
        return phenotype

class GeneExpressionSystem:
    """Manages dynamic gene expression for evolution"""
    
    def __init__(self):
        self.agent_networks: Dict[str, GeneNetwork] = {}
        self.environmental_history: List[Dict[TriggerType, float]] = []
        self.expression_patterns: Dict[str, List[float]] = {}  # Track expression over time
        
        # Define standard gene templates
        self.gene_templates = {
            'speed': Gene(
                name='speed',
                base_value=1.0,
                triggers={
                    TriggerType.STRESS: (0.6, 0.8),  # Upregulate under stress
                    TriggerType.COMPETITION: (0.5, 0.6),
                    TriggerType.TEMPERATURE: (0.7, -0.4)  # Downregulate in heat
                }
            ),
            'strength': Gene(
                name='strength',
                base_value=1.0,
                triggers={
                    TriggerType.COMPETITION: (0.4, 1.0),
                    TriggerType.RESOURCE_SCARCITY: (0.6, 0.7),
                    TriggerType.POPULATION_DENSITY: (0.8, 0.5)
                }
            ),
            'efficiency': Gene(
                name='efficiency',
                base_value=1.0,
                triggers={
                    TriggerType.RESOURCE_SCARCITY: (0.3, 1.2),
                    TriggerType.SEASON: (0.4, 0.3),
                    TriggerType.TEMPERATURE: (0.5, -0.2)
                }
            ),
            'cooperation': Gene(
                name='cooperation',
                base_value=1.0,
                triggers={
                    TriggerType.POPULATION_DENSITY: (0.6, 0.9),
                    TriggerType.STRESS: (0.7, 0.4),
                    TriggerType.RESOURCE_SCARCITY: (0.5, 0.6)
                }
            ),
            'adaptability': Gene(
                name='adaptability',
                base_value=1.0,
                triggers={
                    TriggerType.STRESS: (0.3, 1.5),
                    TriggerType.SEASON: (0.4, 0.8),
                    TriggerType.TEMPERATURE: (0.6, 0.7)
                }
            ),
            'aggression': Gene(
                name='aggression',
                base_value=0.5,
                triggers={
                    TriggerType.COMPETITION: (0.7, 1.2),
                    TriggerType.RESOURCE_SCARCITY: (0.8, 1.0),
                    TriggerType.POPULATION_DENSITY: (0.9, 0.8)
                }
            ),
            'resilience': Gene(
                name='resilience',
                base_value=1.0,
                triggers={
                    TriggerType.STRESS: (0.5, 0.9),
                    TriggerType.TEMPERATURE: (0.8, 0.6),
                    TriggerType.SEASON: (0.6, 0.4)
                }
            )
        }
        
        print("🧬 Gene expression system initialized!")
    
    def create_agent_network(self, agent_id: str, base_traits: Dict[str, float]) -> GeneNetwork:
        """Create a gene network for a new agent"""
        network = GeneNetwork()
        
        # Create genes based on agent's traits
        for trait_name, base_value in base_traits.items():
            if trait_name in self.gene_templates:
                # Create gene from template
                gene_template = self.gene_templates[trait_name]
                gene = Gene(
                    name=trait_name,
                    base_value=base_value,
                    triggers=gene_template.triggers.copy(),
                    mutation_rate=gene_template.mutation_rate
                )
                
                # Add some random variation to triggers
                for trigger_type in gene.triggers:
                    threshold, sensitivity = gene.triggers[trigger_type]
                    # Mutate trigger parameters slightly
                    new_threshold = max(0.1, min(0.9, threshold + random.uniform(-0.1, 0.1)))
                    new_sensitivity = max(0.1, min(2.0, sensitivity + random.uniform(-0.2, 0.2)))
                    gene.triggers[trigger_type] = (new_threshold, new_sensitivity)
                
                network.add_gene(gene)
        
        # Add some gene interactions
        self._add_gene_interactions(network)
        
        self.agent_networks[agent_id] = network
        self.expression_patterns[agent_id] = []
        
        print(f"🧬 Created gene network for agent {agent_id[:8]} with {len(network.genes)} genes")
        return network
    
    def _add_gene_interactions(self, network: GeneNetwork):
        """Add regulatory interactions between genes"""
        gene_names = list(network.genes.keys())
        
        # Add some common interactions
        interactions = [
            ('stress', 'speed', 0.3),
            ('competition', 'aggression', 0.5),
            ('cooperation', 'aggression', -0.4),  # Negative regulation
            ('adaptability', 'resilience', 0.2),
            ('efficiency', 'speed', -0.1),  # Trade-off
            ('strength', 'efficiency', -0.15)  # Another trade-off
        ]
        
        for regulator, target, strength in interactions:
            if regulator in gene_names and target in gene_names:
                network.add_interaction(regulator, target, strength)
        
        # Add some random interactions
        for _ in range(random.randint(2, 5)):
            if len(gene_names) >= 2:
                regulator = random.choice(gene_names)
                target = random.choice([g for g in gene_names if g != regulator])
                strength = random.uniform(-0.3, 0.3)
                network.add_interaction(regulator, target, strength)
    
    def update_agent_expression(self, agent_id: str, environmental_conditions: Dict[str, float]):
        """Update gene expression for a specific agent"""
        if agent_id not in self.agent_networks:
            return
        
        # Convert environmental conditions to trigger signals
        environmental_signals = self._convert_to_signals(environmental_conditions)
        
        # Update the agent's gene network
        network = self.agent_networks[agent_id]
        network.update_network(environmental_signals)
        
        # Record expression pattern
        avg_expression = sum(gene.expression_level for gene in network.genes.values()) / len(network.genes)
        self.expression_patterns[agent_id].append(avg_expression)
        
        # Keep only recent history
        if len(self.expression_patterns[agent_id]) > 50:
            self.expression_patterns[agent_id] = self.expression_patterns[agent_id][-25:]
    
    def _convert_to_signals(self, environmental_conditions: Dict[str, float]) -> Dict[TriggerType, float]:
        """Convert environmental conditions to gene expression signals"""
        signals = {}
        
        # Map environmental conditions to trigger types
        condition_mapping = {
            'temperature': TriggerType.TEMPERATURE,
            'stress_level': TriggerType.STRESS,
            'competition_intensity': TriggerType.COMPETITION,
            'cooperation_level': TriggerType.COOPERATION,
            'resource_scarcity': TriggerType.RESOURCE_SCARCITY,
            'population_density': TriggerType.POPULATION_DENSITY,
            'seasonal_factor': TriggerType.SEASON
        }
        
        for condition, trigger_type in condition_mapping.items():
            if condition in environmental_conditions:
                signals[trigger_type] = environmental_conditions[condition]
        
        return signals
    
    def get_agent_phenotype(self, agent_id: str) -> Dict[str, float]:
        """Get current phenotype for an agent based on gene expression"""
        if agent_id not in self.agent_networks:
            return {}
        
        return self.agent_networks[agent_id].get_phenotype()
    
    def mutate_gene_network(self, parent_network: GeneNetwork, mutation_rate: float = 0.1) -> GeneNetwork:
        """Create a mutated copy of a gene network for offspring"""
        new_network = GeneNetwork()
        
        # Copy and mutate genes
        for gene_name, parent_gene in parent_network.genes.items():
            new_gene = Gene(
                name=gene_name,
                base_value=parent_gene.base_value,
                triggers=parent_gene.triggers.copy(),
                dominant=parent_gene.dominant,
                mutation_rate=parent_gene.mutation_rate
            )
            
            # Mutate base value
            if random.random() < mutation_rate:
                new_gene.base_value *= random.uniform(0.8, 1.2)
                new_gene.base_value = max(0.1, min(3.0, new_gene.base_value))
            
            # Mutate trigger parameters
            if random.random() < mutation_rate * 0.5:
                for trigger_type in new_gene.triggers:
                    threshold, sensitivity = new_gene.triggers[trigger_type]
                    if random.random() < 0.3:  # 30% chance to mutate each trigger
                        new_threshold = threshold + random.uniform(-0.1, 0.1)
                        new_sensitivity = sensitivity + random.uniform(-0.2, 0.2)
                        new_gene.triggers[trigger_type] = (
                            max(0.1, min(0.9, new_threshold)),
                            max(0.1, min(2.0, new_sensitivity))
                        )
            
            new_network.add_gene(new_gene)
        
        # Copy gene interactions (with possible mutations)
        for regulator, targets in parent_network.gene_interactions.items():
            for target, strength in targets:
                if random.random() < mutation_rate * 0.3:  # 30% chance to mutate interaction
                    new_strength = strength + random.uniform(-0.1, 0.1)
                    new_strength = max(-1.0, min(1.0, new_strength))
                    new_network.add_interaction(regulator, target, new_strength)
                else:
                    new_network.add_interaction(regulator, target, strength)
        
        # Occasionally add new random interaction
        if random.random() < mutation_rate * 0.2:
            gene_names = list(new_network.genes.keys())
            if len(gene_names) >= 2:
                regulator = random.choice(gene_names)
                target = random.choice([g for g in gene_names if g != regulator])
                strength = random.uniform(-0.3, 0.3)
                new_network.add_interaction(regulator, target, strength)
        
        return new_network
    
    def crossover_networks(self, parent1_network: GeneNetwork, parent2_network: GeneNetwork) -> GeneNetwork:
        """Create offspring network through genetic crossover"""
        offspring_network = GeneNetwork()
        
        # Get all gene names from both parents
        all_genes = set(parent1_network.genes.keys()) | set(parent2_network.genes.keys())
        
        for gene_name in all_genes:
            # Determine which parent to inherit from (or combine)
            inherit_from = random.choice([1, 2])
            
            if inherit_from == 1 and gene_name in parent1_network.genes:
                parent_gene = parent1_network.genes[gene_name]
            elif gene_name in parent2_network.genes:
                parent_gene = parent2_network.genes[gene_name]
            else:
                continue  # Skip if gene not in selected parent
            
            # Create offspring gene
            offspring_gene = Gene(
                name=gene_name,
                base_value=parent_gene.base_value,
                triggers=parent_gene.triggers.copy(),
                dominant=parent_gene.dominant,
                mutation_rate=parent_gene.mutation_rate
            )
            
            # Possibly blend values if gene exists in both parents
            if (gene_name in parent1_network.genes and 
                gene_name in parent2_network.genes and 
                random.random() < 0.3):  # 30% chance to blend
                
                gene1 = parent1_network.genes[gene_name]
                gene2 = parent2_network.genes[gene_name]
                
                # Blend base values
                blend_factor = random.uniform(0.3, 0.7)
                offspring_gene.base_value = (gene1.base_value * blend_factor + 
                                           gene2.base_value * (1 - blend_factor))
                
                # Blend trigger parameters
                for trigger_type in offspring_gene.triggers:
                    if trigger_type in gene1.triggers and trigger_type in gene2.triggers:
                        t1, s1 = gene1.triggers[trigger_type]
                        t2, s2 = gene2.triggers[trigger_type]
                        
                        new_threshold = t1 * blend_factor + t2 * (1 - blend_factor)
                        new_sensitivity = s1 * blend_factor + s2 * (1 - blend_factor)
                        offspring_gene.triggers[trigger_type] = (new_threshold, new_sensitivity)
            
            offspring_network.add_gene(offspring_gene)
        
        # Inherit gene interactions from both parents
        for parent_network in [parent1_network, parent2_network]:
            for regulator, targets in parent_network.gene_interactions.items():
                if regulator in offspring_network.genes:
                    for target, strength in targets:
                        if target in offspring_network.genes and random.random() < 0.6:
                            offspring_network.add_interaction(regulator, target, strength)
        
        return offspring_network
    
    def get_expression_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary of agent's gene expression"""
        if agent_id not in self.agent_networks:
            return {}
        
        network = self.agent_networks[agent_id]
        
        # Calculate expression statistics
        expressions = [gene.expression_level for gene in network.genes.values()]
        avg_expression = sum(expressions) / len(expressions) if expressions else 0
        
        # Find most/least expressed genes
        sorted_genes = sorted(network.genes.items(), 
                            key=lambda x: x[1].expression_level, reverse=True)
        
        most_expressed = sorted_genes[0] if sorted_genes else None
        least_expressed = sorted_genes[-1] if sorted_genes else None
        
        return {
            'gene_count': len(network.genes),
            'average_expression': avg_expression,
            'expression_variance': sum((e - avg_expression)**2 for e in expressions) / len(expressions) if expressions else 0,
            'most_expressed_gene': most_expressed[0] if most_expressed else None,
            'most_expressed_level': most_expressed[1].expression_level if most_expressed else None,
            'least_expressed_gene': least_expressed[0] if least_expressed else None,
            'least_expressed_level': least_expressed[1].expression_level if least_expressed else None,
            'interaction_count': sum(len(targets) for targets in network.gene_interactions.values()),
            'recent_expression_trend': (
                sum(self.expression_patterns[agent_id][-5:]) / 5 if 
                len(self.expression_patterns[agent_id]) >= 5 else avg_expression
            )
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall gene expression system status"""
        if not self.agent_networks:
            return {'active_networks': 0}
        
        # Calculate population-wide statistics
        all_expressions = []
        all_gene_counts = []
        all_interaction_counts = []
        
        for network in self.agent_networks.values():
            expressions = [gene.expression_level for gene in network.genes.values()]
            all_expressions.extend(expressions)
            all_gene_counts.append(len(network.genes))
            all_interaction_counts.append(sum(len(targets) for targets in network.gene_interactions.values()))
        
        return {
            'active_networks': len(self.agent_networks),
            'total_genes': sum(all_gene_counts),
            'average_genes_per_agent': sum(all_gene_counts) / len(all_gene_counts) if all_gene_counts else 0,
            'population_avg_expression': sum(all_expressions) / len(all_expressions) if all_expressions else 0,
            'total_interactions': sum(all_interaction_counts),
            'highly_expressed_genes': len([e for e in all_expressions if e > 1.3]),
            'suppressed_genes': len([e for e in all_expressions if e < 0.7])
        } 