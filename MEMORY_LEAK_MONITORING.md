# Walker System Memory Leak Monitoring & Debug System

## Overview

This document describes the comprehensive memory leak monitoring and debugging system implemented for the Walker training environment. The system provides real-time monitoring, automated alerts, and emergency cleanup mechanisms to identify and resolve memory leaks.

## Components Implemented

### 1. Enhanced Debug Logging (in `train_robots_web_visual.py`)

**Features:**
- **Detailed Body Type Analysis**: Breaks down Box2D bodies by type (static, dynamic, terrain, agent, unknown)
- **Growth Rate Tracking**: Monitors entity growth per minute to detect leaks
- **Automated Alerts**: Warns when metrics exceed safe thresholds
- **Emergency Cleanup**: Automatically triggers aggressive cleanup when critical thresholds are exceeded

**Key Metrics Tracked:**
```
- Memory usage (MB)
- Total world bodies 
- Body type breakdown:
  - Static bodies (ground, fixed obstacles)
  - Dynamic bodies (moving objects)
  - Terrain bodies (generated terrain)
  - Agent bodies (robot bodies)
  - Unknown bodies (potential leaks)
- Food sources count
- Event accumulation (consumption, death, predation)
- Replay buffer memory usage
```

**Thresholds & Alerts:**
- **Memory**: Warning at 1000MB, Critical at 1500MB
- **World Bodies**: Warning at 500, Critical at 1000
- **Static Bodies**: Critical at 300 (terrain leak detection)
- **Food Sources**: Warning at 50 (ecosystem overflow)

### 2. Enhanced Cleanup Mechanisms

**Regular Performance Cleanup (`_cleanup_performance_data`):**
- Aggressive replay buffer limiting (reduced from 25k to 10k experiences max)
- Emergency food source limiting (max 30 sources)
- Orphaned body detection and cleanup
- Event list size limiting
- Memory pool optimization

**Emergency Cleanup (`_emergency_cleanup`):**
- Triggered automatically when memory > 2GB or bodies > 1500
- Reduces food sources to 10
- Clears all event lists
- Destroys orphaned bodies
- Forces Python garbage collection
- Can be manually triggered via API: `POST /force_cleanup`

### 3. Enhanced Prometheus Metrics

**New metrics exported:**
```
# Body type breakdown for memory leak detection
walker_body_types_total{type="static"} 1
walker_body_types_total{type="dynamic"} 155  
walker_body_types_total{type="terrain"} 0
walker_body_types_total{type="agent"} 30
walker_body_types_total{type="unknown"} 0

# System health metrics
walker_system_memory_mb 738
walker_system_cpu_percent 56.75
walker_entities_total{type="world_bodies"} 186
walker_entities_total{type="food_sources"} 6
```

### 4. System Health Monitor (`monitor_system_health.py`)

**Features:**
- **Continuous Monitoring**: Tracks metrics every 20-30 seconds
- **Trend Analysis**: Detects concerning growth patterns
- **Automated Alerts**: Prevents alert spam while highlighting issues
- **Auto-Cleanup**: Can automatically trigger emergency cleanup
- **Growth Rate Analysis**: Calculates entities/minute growth rates

**Usage:**
```bash
# Basic monitoring
python3 monitor_system_health.py

# With auto-cleanup and aggressive thresholds
python3 monitor_system_health.py --auto-cleanup --memory-warning 600 --memory-critical 900

# Using the convenience script
./start_monitoring.sh --auto-cleanup --aggressive-thresholds
```

## Current System Health (Baseline)

As of implementation:
```
Memory: ~738MB (stable)
World Bodies: 186 (30 agents × ~6 bodies each + ground)
Body Types:
  - Static: 1 (ground)
  - Dynamic: 155 (agent parts)  
  - Terrain: 0 (static terrain)
  - Agent: 30 (main agent bodies)
  - Unknown: 0 (no leaks detected)
Food Sources: 6 (normal ecosystem activity)
Replay Experiences: 0 (agents learning)
```

## Memory Leak Detection Strategy

### 1. World Body Growth Monitoring
- **Expected**: ~6 bodies per agent + fixed terrain = ~200 bodies for 30 agents
- **Warning**: Growth rate > 10 bodies/minute
- **Likely Sources**: 
  - Terrain regeneration without cleanup
  - Agent body destruction failures
  - Food source physics bodies not cleaned up

### 2. Memory Growth Monitoring  
- **Expected**: 600-800MB baseline, gradual growth with learning
- **Warning**: Growth rate > 20MB/minute
- **Likely Sources**:
  - Replay buffer accumulation
  - Event list accumulation
  - Orphaned object references

### 3. Food Source Monitoring
- **Expected**: 5-15 food sources during normal ecosystem activity
- **Warning**: > 30 sources, Growth rate > 5/minute
- **Likely Sources**:
  - Resource generation without consumption
  - Cleanup not removing depleted sources

## How to Use the Monitoring System

### 1. Start Monitoring
```bash
# Start with auto-cleanup enabled
./start_monitoring.sh --auto-cleanup --aggressive-thresholds

# Or manually with custom settings
python3 monitor_system_health.py \
  --interval 20 \
  --auto-cleanup \
  --memory-warning 600 \
  --memory-critical 900 \
  --bodies-warning 300 \
  --bodies-critical 500
```

### 2. Check Current Status
```bash
# Get detailed performance status
curl -s http://localhost:7777/performance_status | python3 -m json.tool

# Get Prometheus metrics
curl -s http://localhost:7777/metrics | grep walker_body_types

# Force emergency cleanup if needed
curl -s -X POST http://localhost:7777/force_cleanup
```

### 3. Grafana Dashboard Monitoring
Access the Grafana dashboard at `http://localhost:3009` to view:
- Memory usage trends
- Body type breakdown charts  
- Growth rate visualizations
- Alert thresholds

## Troubleshooting Memory Leaks

### If World Bodies Keep Growing:
1. Check body type breakdown: `curl -s http://localhost:7777/metrics | grep walker_body_types`
2. If `static` bodies growing: Terrain generation leak
3. If `dynamic` bodies growing: Agent cleanup failure
4. If `unknown` bodies growing: Unidentified leak source

### If Memory Keeps Growing:
1. Check replay buffer sizes in performance status
2. Monitor event accumulation (consumption, death, predation)
3. Use manual cleanup: `curl -X POST http://localhost:7777/force_cleanup`
4. Check for orphaned objects in logs

### Emergency Actions:
1. **Auto-cleanup**: Enabled with `--auto-cleanup` flag
2. **Manual cleanup**: `POST /force_cleanup`
3. **Container restart**: `docker compose restart walker-training-app`
4. **Threshold adjustment**: Lower warning/critical values

## Monitoring Output Examples

### Normal Operation:
```
11:37:19: Memory=738.3MB, Bodies=186, Food=6, Step=5558
   🔍 Body Types: Static=1, Dynamic=155, Terrain=0, Agent=30, Unknown=0
```

### Memory Leak Detected:
```
11:45:22: Memory=1205.8MB, Bodies=1247, Food=67, Step=15440
🚨 CRITICAL: world_bodies = 1247 (>= 1000)
🚨 BODY GROWTH ALERT: static_bodies +234 (15.6/min)
⚠️ WARNING: food_sources = 67 (>= 50)
🆘 EMERGENCY: Triggering emergency cleanup!
```

### Post-Cleanup:
```
11:46:01: Memory=856.2MB, Bodies=198, Food=10, Step=15670  
🧹 Emergency cleanup completed: 349.6MB freed
   Bodies: 1247 → 198 (-1049)
   Food: 67 → 10 (-57)
```

## Files Modified/Added

- `train_robots_web_visual.py`: Enhanced logging, cleanup, emergency systems
- `monitor_system_health.py`: Standalone monitoring script
- `start_monitoring.sh`: Convenient startup script
- `MEMORY_LEAK_MONITORING.md`: This documentation

## Integration with Existing Systems

The monitoring system integrates seamlessly with:
- **Prometheus**: Enhanced metrics exported automatically
- **Grafana**: New panels for body type breakdown and health monitoring
- **MLflow**: Performance metrics logged for analysis
- **Docker**: Container health monitoring
- **REST API**: All functionality accessible via HTTP endpoints

This comprehensive system provides early detection, automated mitigation, and detailed analysis capabilities to maintain system stability during long-running training sessions. 