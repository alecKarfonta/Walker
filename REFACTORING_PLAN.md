# Agent Hierarchy Simplification Plan

## Problem
The current agent hierarchy is overly complex with 5 classes doing overlapping work:
- `BaseAgent` (abstract interface)
- `BasicAgent` (unused simple Q-learning)  
- `CrawlingCrate` (deprecated physics-only)
- `CrawlingCrateAgent` (physics + basic learning)
- `EvolutionaryCrawlingAgent` (physics + learning + evolution) ← **Actually used**

## Solution
Consolidate into 2 classes:
- `BaseAgent` (abstract interface) - Keep as is
- `CrawlingAgent` (consolidated) - Combine all functionality

## Implementation Steps

### Phase 1: Create New Consolidated Class
1. Create `src/agents/crawling_agent.py`
2. Combine all functionality from:
   - Box2D physics (`CrawlingCrate`)
   - Neural network learning (`CrawlingCrateAgent`) 
   - Multi-limb evolution (`EvolutionaryCrawlingAgent`)
3. Use composition over inheritance where possible

### Phase 2: Update Imports
Update all files that import `EvolutionaryCrawlingAgent`:
- `train_robots_web_visual.py`
- `src/persistence/*.py` 
- `src/population/*.py`
- `src/rendering/*.py`
- `src/agents/robot_memory_pool.py`

### Phase 3: Remove Deprecated Classes
1. Delete `src/agents/basic_agent.py` (unused)
2. Delete `src/agents/crawling_crate.py` (deprecated)
3. Delete `src/agents/crawling_crate_agent.py` (consolidated)
4. Delete `src/agents/evolutionary_crawling_agent.py` (consolidated)

### Phase 4: Clean Up References
- Update type hints
- Update documentation
- Update tests

## Benefits
- **Simpler codebase**: 5 classes → 2 classes  
- **Easier maintenance**: Single class to modify
- **Better performance**: No inheritance overhead
- **Clearer purpose**: One class = one robot type
- **Reduced confusion**: No more "which agent class do I use?"

## File Changes Required
- **Create**: `src/agents/crawling_agent.py` (~500 lines)
- **Update**: 15+ files with import changes
- **Delete**: 4 agent class files
- **Net reduction**: ~800 lines of duplicate code 