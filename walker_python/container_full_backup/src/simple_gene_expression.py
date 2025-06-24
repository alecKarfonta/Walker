"""Simple Gene Expression System for Enhanced Evolution"""

import random
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SimpleGene:
    """A gene with dynamic expression"""
    name: str
    base_value: float
    expression_level: float = 1.0
    stress_response: float = 0.0  # How much this gene responds to stress
    competition_response: float = 0.0  # How much this gene responds to competition
    
    def get_expressed_value(self) -> float:
        """Get the actual trait value after expression"""
        return self.base_value * self.expression_level
    
    def update_expression(self, stress_level: float, competition_level: float):
        """Update gene expression based on environmental factors"""
        # Base expression level
        new_expression = 1.0
        
        # Stress effects
        if stress_level > 0.5:  # High stress
            new_expression += self.stress_response * (stress_level - 0.5)
        
        # Competition effects
        if competition_level > 0.6:  # High competition
            new_expression += self.competition_response * (competition_level - 0.6)
        
        # Keep expression within reasonable bounds
        self.expression_level = max(0.2, min(2.0, new_expression))

class SimpleGeneExpressionSystem:
    """Manages dynamic gene expression for agents"""
    
    def __init__(self):
        self.agent_genes: Dict[str, Dict[str, SimpleGene]] = {}
        print("🧬 Simple gene expression system initialized!")
    
    def create_agent_genes(self, agent_id: str, traits: Dict[str, float]):
        """Create genes for a new agent"""
        genes = {}
        
        for trait_name, base_value in traits.items():
            gene = SimpleGene(
                name=trait_name,
                base_value=base_value,
                stress_response=random.uniform(-0.3, 0.8),  # Can be positive or negative
                competition_response=random.uniform(-0.2, 0.6)
            )
            genes[trait_name] = gene
        
        self.agent_genes[agent_id] = genes
        print(f"🧬 Created {len(genes)} genes for agent {agent_id[:8]}")
    
    def update_expression(self, agent_id: str, environmental_conditions: Dict[str, float]):
        """Update gene expression based on environment"""
        if agent_id not in self.agent_genes:
            return
        
        stress_level = environmental_conditions.get('stress', 0.5)
        competition_level = environmental_conditions.get('competition', 0.5)
        
        # Update all genes for this agent
        for gene in self.agent_genes[agent_id].values():
            gene.update_expression(stress_level, competition_level)
    
    def get_expressed_traits(self, agent_id: str) -> Dict[str, float]:
        """Get current trait values after gene expression"""
        if agent_id not in self.agent_genes:
            return {}
        
        return {name: gene.get_expressed_value() 
                for name, gene in self.agent_genes[agent_id].items()}
    
    def mutate_genes(self, parent_id: str, child_id: str):
        """Create mutated copy of parent's genes for child"""
        if parent_id not in self.agent_genes:
            return
        
        child_genes = {}
        for name, parent_gene in self.agent_genes[parent_id].items():
            # Create child gene with some mutation
            child_gene = SimpleGene(
                name=name,
                base_value=parent_gene.base_value * random.uniform(0.9, 1.1),
                stress_response=parent_gene.stress_response + random.uniform(-0.1, 0.1),
                competition_response=parent_gene.competition_response + random.uniform(-0.1, 0.1)
            )
            
            # Keep response values in reasonable bounds
            child_gene.stress_response = max(-0.5, min(1.0, child_gene.stress_response))
            child_gene.competition_response = max(-0.5, min(1.0, child_gene.competition_response))
            
            child_genes[name] = child_gene
        
        self.agent_genes[child_id] = child_genes
    
    def get_expression_summary(self, agent_id: str) -> Dict[str, float]:
        """Get summary of agent's current gene expression"""
        if agent_id not in self.agent_genes:
            return {}
        
        genes = self.agent_genes[agent_id]
        avg_expression = sum(gene.expression_level for gene in genes.values()) / len(genes)
        
        highly_expressed = len([g for g in genes.values() if g.expression_level > 1.3])
        suppressed = len([g for g in genes.values() if g.expression_level < 0.7])
        
        return {
            'average_expression': avg_expression,
            'highly_expressed_count': highly_expressed,
            'suppressed_count': suppressed,
            'total_genes': len(genes)
        }
    
    def get_population_expression_stats(self) -> Dict[str, float]:
        """Get population-wide gene expression statistics"""
        if not self.agent_genes:
            return {}
        
        all_expressions = []
        for agent_genes in self.agent_genes.values():
            for gene in agent_genes.values():
                all_expressions.append(gene.expression_level)
        
        if not all_expressions:
            return {}
        
        return {
            'population_avg_expression': sum(all_expressions) / len(all_expressions),
            'highly_expressed_genes': len([e for e in all_expressions if e > 1.3]),
            'suppressed_genes': len([e for e in all_expressions if e < 0.7]),
            'total_active_genes': len(all_expressions)
        } 