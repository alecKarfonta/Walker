# Reward Signal Quality Monitoring in Grafana

## Overview

Your Walker robot training system now includes comprehensive reward signal quality monitoring integrated with Grafana. This enables real-time visualization and alerting for one of the most critical aspects of reinforcement learning - the quality of reward signals.

## What's Integrated

### ✅ **Complete Infrastructure**

1. **Reward Signal Evaluator** - 20+ quality metrics per agent
2. **Dashboard Exporter** - Prometheus metrics export 
3. **Grafana Dashboards** - Pre-configured visualization panels
4. **API Endpoints** - 6 new endpoints for reward signal data
5. **Prometheus Scraping** - Automated metrics collection
6. **Alerting System** - Automated quality issue detection

### 📊 **Key Metrics Available in Grafana**

#### Quality Overview
- **Average Quality Score** (0-1) - Overall reward signal effectiveness
- **Signal-to-Noise Ratio** - Clarity of learning signals  
- **Reward Consistency** - Reliability across similar states
- **Reward Sparsity** - Frequency of non-zero rewards
- **Exploration Incentive** - How well rewards encourage exploration

#### Quality Distribution
- **Agent Quality Tiers** - Excellent/Good/Fair/Poor/Very Poor
- **Quality Percentage** - % of agents with good reward signals
- **Issue Tracking** - Count of agents with specific problems

#### Problem Detection
- **Sparse Rewards** - Agents with >90% zero rewards
- **Noisy Rewards** - Agents with low signal-to-noise ratio
- **Inconsistent Rewards** - Agents with unreliable reward functions
- **Poor Exploration** - Agents with weak exploration incentives

## Accessing Grafana

### 🌐 **Connection Details**
- **URL**: http://localhost:3009
- **Username**: `admin`
- **Password**: `walker-admin-2024`

### 📈 **Available Dashboards**

1. **Walker Training Overview** - Main training dashboard (existing)
2. **Walker Reward Signal Quality** - Dedicated reward signal monitoring (new)

## Dashboard Configuration

### Import Reward Signal Dashboard

1. **Navigate to Grafana**: http://localhost:3009
2. **Login** with credentials above
3. **Import Dashboard**:
   - Go to "+" → "Import"
   - Upload `config/grafana/dashboard-configs/reward-signal-dashboard.json`
   - Or manually configure panels using the metrics below

### 📊 **Available Prometheus Metrics**

All metrics are prefixed with `walker_reward_signals_`:

```prometheus
# Overall Quality Metrics
walker_reward_signals_avg_quality_score
walker_reward_signals_avg_signal_to_noise_ratio  
walker_reward_signals_avg_consistency
walker_reward_signals_avg_sparsity

# Quality Distribution
walker_reward_signals_agents_excellent
walker_reward_signals_agents_good
walker_reward_signals_agents_fair
walker_reward_signals_agents_poor
walker_reward_signals_agents_very_poor

# Issue Tracking
walker_reward_signals_sparse_reward_agents
walker_reward_signals_noisy_reward_agents
walker_reward_signals_inconsistent_reward_agents
walker_reward_signals_poor_exploration_agents

# System Status
walker_reward_signals_total_agents
walker_reward_signals_total_rewards_recorded
walker_reward_signals_agents_with_good_rewards
walker_reward_signals_percentage_good_quality
```

### 🎨 **Recommended Panels**

#### 1. Quality Score Gauge
```
Query: walker_reward_signals_avg_quality_score
Type: Gauge
Thresholds: Red (0-0.4), Yellow (0.4-0.6), Green (0.6-1.0)
```

#### 2. Signal Metrics Timeline
```
Queries:
- walker_reward_signals_avg_signal_to_noise_ratio
- walker_reward_signals_avg_consistency  
- 1 - walker_reward_signals_avg_sparsity (Reward Density)
Type: Time Series
```

#### 3. Quality Distribution Pie Chart
```
Queries:
- walker_reward_signals_agents_excellent
- walker_reward_signals_agents_good
- walker_reward_signals_agents_fair
- walker_reward_signals_agents_poor + walker_reward_signals_agents_very_poor
Type: Pie Chart
```

#### 4. Issue Detection Stats
```
Queries:
- walker_reward_signals_sparse_reward_agents
- walker_reward_signals_noisy_reward_agents
- walker_reward_signals_inconsistent_reward_agents
- walker_reward_signals_poor_exploration_agents
Type: Stat Panel with thresholds
```

## Alerting Setup

### 🚨 **Pre-configured Alerts**

The system automatically generates alerts for:

- **Poor Average Quality** - Score < 0.3 (Critical)
- **Sparse Rewards** - >30% of agents have sparse rewards (Warning)
- **Noisy Rewards** - >20% of agents have noisy rewards (Warning)

### Custom Alert Rules

Add these to Prometheus `alert.rules.yml`:

```yaml
groups:
- name: reward_signal_quality
  rules:
  - alert: RewardQualityTooLow
    expr: walker_reward_signals_avg_quality_score < 0.3
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Reward signal quality is critically low"
      description: "Average reward quality score is {{ $value }}"

  - alert: TooManySparseRewards
    expr: walker_reward_signals_sparse_reward_agents > 10
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Many agents have sparse rewards"
      description: "{{ $value }} agents have sparse reward signals"
```

## Data Flow

```
Training Agents → Reward Signals → Evaluator → Dashboard Exporter → Prometheus → Grafana
```

1. **Agents generate rewards** during training
2. **Reward Signal Adapter** captures signals non-intrusively
3. **Evaluator analyzes quality** every 100 samples per agent
4. **Dashboard Exporter** aggregates and exports to Prometheus format
5. **Prometheus scrapes metrics** every 15-30 seconds
6. **Grafana visualizes** real-time data with alerts

## Troubleshooting

### No Data in Grafana

**Possible Causes:**
1. Agents haven't started generating rewards yet
2. Prometheus not scraping correctly
3. Dashboard Exporter not running

**Solutions:**
```bash
# Check if agents are training
curl http://localhost:7777/reward_signal_status

# Check Prometheus targets
curl "http://localhost:9889/api/v1/targets"

# Check dashboard exporter
curl http://localhost:2322/api/health
```

### Missing Metrics

**Check Prometheus Configuration:**
```bash
# Verify scraping config
docker exec walker-prometheus cat /etc/prometheus/prometheus.yml
```

**Restart services if needed:**
```bash
docker compose restart walker-prometheus walker-grafana
```

### Dashboard Import Issues

1. **Manual Panel Creation**: Use the Prometheus queries above
2. **Check Data Source**: Ensure Prometheus is configured in Grafana
3. **Verify Metrics**: Test queries in Prometheus web UI (http://localhost:9889)

## Best Practices

### 📈 **Monitoring Workflow**

1. **Setup Dashboards** - Import or create reward signal quality panels
2. **Configure Alerts** - Set thresholds for quality score and issue counts
3. **Regular Review** - Monitor trends in quality metrics
4. **Issue Investigation** - Use diagnostics endpoints for detailed analysis
5. **Reward Tuning** - Adjust reward functions based on quality feedback

### 🎯 **Quality Targets**

- **Excellent Quality**: Score > 0.8
- **Acceptable Quality**: Score > 0.6  
- **Needs Improvement**: Score < 0.4
- **Critical Issues**: Score < 0.2

### 🔧 **Optimization Tips**

1. **Monitor Early** - Check quality soon after training starts
2. **Watch Trends** - Quality should improve over time
3. **Balance Metrics** - Don't optimize for just one metric
4. **Use Alerts** - Set up notifications for quality degradation
5. **Regular Analysis** - Review quality reports weekly

## API Endpoints for Custom Dashboards

If you prefer to create custom dashboards or integrations:

```bash
# System Status
curl http://localhost:7777/reward_signal_status

# Quality Summary  
curl http://localhost:7777/reward_signal_summary

# Agent Diagnostics
curl http://localhost:7777/reward_signal_agent/<agent_id>/diagnostics

# Quality Comparison
curl http://localhost:7777/reward_signal_comparison

# Enhanced Performance (includes reward signals)
curl http://localhost:7777/performance_status
```

## Conclusion

Your Walker training system now provides comprehensive reward signal quality monitoring through Grafana. This enables you to:

- **Identify reward function problems** before they impact learning
- **Compare reward quality** across different agent types
- **Monitor training health** in real-time
- **Optimize reward functions** based on objective metrics
- **Set up alerts** for immediate notification of issues

The reward signal is the foundation of successful reinforcement learning - now you have the tools to ensure it's working optimally! 🎯📊 