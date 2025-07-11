#!/bin/bash

# 📈 Walker Performance Data Analyzer
# Analyzes CSV performance data from monitoring reports

REPORT_DIR="monitoring_reports"

echo "📈 WALKER PERFORMANCE DATA ANALYZER"
echo "==================================="

# Find latest CSV file
LATEST_CSV=$(ls -t "$REPORT_DIR"/performance_data_*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_CSV" ]; then
    echo "❌ No CSV performance data found in $REPORT_DIR"
    echo "Run monitoring first: ./monitor_system.sh"
    exit 1
fi

echo "📊 Analyzing: $LATEST_CSV"
DATA_POINTS=$(tail -n +2 "$LATEST_CSV" | wc -l)
echo "📊 Data Points: $DATA_POINTS"
echo ""

if [ "$DATA_POINTS" -eq 0 ]; then
    echo "⚠️ No data points available yet. Wait for monitoring to collect data."
    exit 0
fi

# Performance trends
echo "🎯 PERFORMANCE TRENDS:"
echo "====================="

echo "📊 Agent Count Over Time:"
echo "Check# | Agents"
tail -n +2 "$LATEST_CSV" | awk -F',' '{printf "%6s | %s\n", $2, $6}'
echo ""

echo "⚡ Physics Performance:"
echo "Check# | FPS"
tail -n +2 "$LATEST_CSV" | awk -F',' '{printf "%6s | %s\n", $2, $7}'
echo ""

echo "🏆 Reward Progression:"
echo "Check# | Top Reward | Avg Reward"
tail -n +2 "$LATEST_CSV" | awk -F',' '{printf "%6s | %10.1f | %10.1f\n", $2, $9, $10}'
echo ""

echo "🌐 API Performance:"
echo "Check# | Response (ms)"
tail -n +2 "$LATEST_CSV" | awk -F',' '{printf "%6s | %s\n", $2, $17}'
echo ""

echo "🧠 System Resources:"
echo "Check# | CPU% | Memory%"
tail -n +2 "$LATEST_CSV" | awk -F',' '{printf "%6s | %4s | %7s\n", $2, $14, $16}'
echo ""

echo "🌍 World Status:"
echo "Check# | Obstacles | Food | Dynamic"
tail -n +2 "$LATEST_CSV" | awk -F',' '{printf "%6s | %9s | %4s | %7s\n", $2, $11, $12, $13}'
echo ""

# Summary statistics
echo "📈 SUMMARY STATISTICS:"
echo "====================="

if [ "$DATA_POINTS" -gt 1 ]; then
    # Agent count stats
    AGENT_MIN=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f6 | sort -n | head -1)
    AGENT_MAX=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f6 | sort -n | tail -1)
    AGENT_AVG=$(tail -n +2 "$LATEST_CSV" | awk -F',' '{sum+=$6} END {printf "%.1f", sum/NR}')
    
    # Physics FPS stats
    FPS_MIN=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f7 | sort -n | head -1)
    FPS_MAX=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f7 | sort -n | tail -1)
    FPS_AVG=$(tail -n +2 "$LATEST_CSV" | awk -F',' '{sum+=$7} END {printf "%.1f", sum/NR}')
    
    # Response time stats
    RT_MIN=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f17 | sort -n | head -1)
    RT_MAX=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f17 | sort -n | tail -1)
    RT_AVG=$(tail -n +2 "$LATEST_CSV" | awk -F',' '{sum+=$17} END {printf "%.1f", sum/NR}')
    
    # Reward progression
    REWARD_FIRST=$(tail -n +2 "$LATEST_CSV" | head -1 | cut -d',' -f10)
    REWARD_LAST=$(tail -n +2 "$LATEST_CSV" | tail -1 | cut -d',' -f10)
    
    echo "🤖 Agents:       $AGENT_MIN - $AGENT_MAX (avg: $AGENT_AVG)"
    echo "⚡ Physics FPS:  $FPS_MIN - $FPS_MAX (avg: $FPS_AVG)"
    echo "🌐 API Response: ${RT_MIN}ms - ${RT_MAX}ms (avg: ${RT_AVG}ms)"
    echo "🏆 Avg Reward:   $REWARD_FIRST → $REWARD_LAST"
    
    # Health status
    HEALTHY_COUNT=$(tail -n +2 "$LATEST_CSV" | awk -F',' '$5=="HEALTHY" {count++} END {print count+0}')
    HEALTH_RATE=$(echo "scale=1; $HEALTHY_COUNT * 100 / $DATA_POINTS" | bc 2>/dev/null || echo "100")
    echo "✅ Health Rate:  ${HEALTH_RATE}% ($HEALTHY_COUNT/$DATA_POINTS healthy)"
else
    echo "⚠️ Need more data points for statistical analysis"
fi

echo ""
echo "🔧 ANALYSIS COMMANDS:"
echo "===================="
echo "Raw CSV view:        cat '$LATEST_CSV'"
echo "Spreadsheet import:  Open '$LATEST_CSV' in Excel/LibreOffice"
echo "Column headers:      head -1 '$LATEST_CSV' | tr ',' '\n' | nl"
echo "Latest data:         tail -5 '$LATEST_CSV'"
echo "Formatted view:      tail -10 '$LATEST_CSV' | column -t -s,"
echo ""

# Python/R analysis suggestions
echo "📊 ADVANCED ANALYSIS IDEAS:"
echo "==========================="
echo "• Import CSV into Python pandas for plotting trends"
echo "• Use R for statistical analysis and visualization"
echo "• Create Excel charts for performance dashboards"
echo "• Monitor correlation between FPS and agent performance"
echo "• Track response time patterns during high load"
echo "• Analyze reward progression over training time"
echo ""

echo "📍 Re-run analysis: ./analyze_performance.sh" 