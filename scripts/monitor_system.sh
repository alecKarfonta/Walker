#!/bin/bash

# 🔍 Walker Training System Monitor
# Monitors system health every 5 minutes for 1 hour and saves detailed reports

REPORT_DIR="monitoring_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="${REPORT_DIR}/system_monitor_${TIMESTAMP}.log"
SUMMARY_FILE="${REPORT_DIR}/latest_summary.json"
CSV_FILE="${REPORT_DIR}/performance_data_${TIMESTAMP}.csv"
CHECK_INTERVAL=300  # 5 minutes in seconds
TOTAL_CHECKS=12     # 12 checks = 1 hour

# Create reports directory
mkdir -p "$REPORT_DIR"

# Initialize CSV file with headers
cat > "$CSV_FILE" << EOF
timestamp,check_number,progress_percent,api_status,health_status,agent_count,physics_fps,simulation_speed,top_reward,avg_reward,obstacle_count,food_count,dynamic_obstacles,cpu_percent,memory_usage_mb,memory_percent,api_response_time_ms,achievements_last_5min,system_alerts_last_5min
EOF

# Initialize report file
cat > "$REPORT_FILE" << EOF
🔍 WALKER TRAINING SYSTEM MONITORING REPORT
==========================================
Start Time: $(date)
Check Interval: 5 minutes
Total Duration: 1 hour (12 checks)
Report File: $REPORT_FILE
CSV Data File: $CSV_FILE

EOF

echo "🔍 Starting Walker system monitoring..."
echo "📊 Report will be saved to: $REPORT_FILE"
echo "📈 Summary will be updated in: $SUMMARY_FILE"
echo "📋 CSV data will be saved to: $CSV_FILE"

for i in $(seq 1 $TOTAL_CHECKS); do
    CHECK_TIME=$(date)
    ISO_TIMESTAMP=$(date -Iseconds)
    echo "🕐 Check $i/$TOTAL_CHECKS at $CHECK_TIME"
    
    # Initialize CSV data variables
    API_RESPONSE_TIME="unknown"
    AGENT_COUNT="0"
    PHYSICS_FPS="0"
    SIMULATION_SPEED="0"
    TOP_REWARD="0"
    AVG_REWARD="0"
    OBSTACLE_COUNT="0"
    FOOD_COUNT="0"
    DYNAMIC_OBSTACLES="0"
    CPU_PERCENT="0"
    MEMORY_USAGE_MB="0"
    MEMORY_PERCENT="0"
    ACHIEVEMENTS_COUNT="0"
    ALERTS_COUNT="0"
    
    # Write check header to report
    cat >> "$REPORT_FILE" << EOF

=====================================
CHECK $i/$TOTAL_CHECKS - $CHECK_TIME
=====================================

EOF

    # 1. Container Status
    echo "🐳 CONTAINER STATUS:" >> "$REPORT_FILE"
    docker ps --filter name=walker-training-app --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" >> "$REPORT_FILE" 2>&1
    echo "" >> "$REPORT_FILE"

    # 2. API Health Check with timing
    echo "🌐 API HEALTH CHECK:" >> "$REPORT_FILE"
    API_START_TIME=$(date +%s%3N)
    API_STATUS=$(curl -s -w "%{http_code}" http://localhost:7777/status -o /dev/null 2>/dev/null || echo "FAILED")
    API_END_TIME=$(date +%s%3N)
    API_RESPONSE_TIME=$((API_END_TIME - API_START_TIME))
    echo "HTTP Status: $API_STATUS" >> "$REPORT_FILE"
    echo "Response Time: ${API_RESPONSE_TIME}ms" >> "$REPORT_FILE"
    
    if [ "$API_STATUS" = "200" ]; then
        echo "✅ API Responding" >> "$REPORT_FILE"
        
        # 3. System Metrics
        echo "" >> "$REPORT_FILE"
        echo "📊 SYSTEM METRICS:" >> "$REPORT_FILE"
        METRICS=$(curl -s http://localhost:7777/status 2>/dev/null | jq '{
            agent_count: (.agents | length),
            physics_fps: .physics_fps,
            has_ecosystem: (has("ecosystem")),
            has_environment: (has("environment")),
            simulation_speed: .simulation_speed,
            camera_position: .camera
        }' 2>/dev/null || echo '{"error": "Failed to parse JSON"}')
        echo "$METRICS" >> "$REPORT_FILE"
        
        # Extract metrics for CSV
        AGENT_COUNT=$(echo "$METRICS" | jq -r '.agent_count // 0')
        PHYSICS_FPS=$(echo "$METRICS" | jq -r '.physics_fps // 0')
        SIMULATION_SPEED=$(echo "$METRICS" | jq -r '.simulation_speed // 0')
        
        # 4. Training Performance
        echo "" >> "$REPORT_FILE"
        echo "🎯 TRAINING PERFORMANCE:" >> "$REPORT_FILE"
        PERFORMANCE=$(curl -s http://localhost:7777/status 2>/dev/null | jq '{
            top_performers: (.leaderboard[0:3] | map({
                id: .id,
                total_reward: .total_reward,
                position: .position
            })),
            agent_sample: (.agents[0:10] | map({
                id: .id,
                total_reward: .total_reward
            }))
        }' 2>/dev/null || echo '{"error": "Failed to parse performance data"}')
        echo "$PERFORMANCE" >> "$REPORT_FILE"
        
        # Extract performance data for CSV
        TOP_REWARD=$(echo "$PERFORMANCE" | jq -r '.top_performers[0].total_reward // 0')
        AVG_REWARD=$(echo "$PERFORMANCE" | jq -r '.agent_sample | map(.total_reward // 0) | add / length' 2>/dev/null || echo "0")
        
        # 5. Dynamic World Status
        echo "" >> "$REPORT_FILE"
        echo "🌍 DYNAMIC WORLD STATUS:" >> "$REPORT_FILE"
        WORLD_STATUS=$(curl -s http://localhost:7777/status 2>/dev/null | jq '{
            obstacles: (.environment.obstacles | length),
            food_sources: (.environment.food_sources | length),
            dynamic_obstacles: (.environment.obstacles | map(select(.is_dynamic_world == true)) | length),
            has_ecosystem_data: (has("ecosystem"))
        }' 2>/dev/null || echo '{"error": "Failed to parse world data"}')
        echo "$WORLD_STATUS" >> "$REPORT_FILE"
        
        # Extract world data for CSV
        OBSTACLE_COUNT=$(echo "$WORLD_STATUS" | jq -r '.obstacles // 0')
        FOOD_COUNT=$(echo "$WORLD_STATUS" | jq -r '.food_sources // 0')
        DYNAMIC_OBSTACLES=$(echo "$WORLD_STATUS" | jq -r '.dynamic_obstacles // 0')
        
        # 6. Memory and Performance
        echo "" >> "$REPORT_FILE"
        echo "🧠 SYSTEM RESOURCES:" >> "$REPORT_FILE"
        CONTAINER_STATS=$(docker stats walker-training-app --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null || echo "Failed to get container stats")
        echo "$CONTAINER_STATS" >> "$REPORT_FILE"
        
        # Extract resource data for CSV (more robust parsing)
        RESOURCE_DATA=$(docker stats walker-training-app --no-stream --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" 2>/dev/null)
        if [ -n "$RESOURCE_DATA" ]; then
            CPU_PERCENT=$(echo "$RESOURCE_DATA" | cut -d',' -f1 | sed 's/%//')
            MEMORY_USAGE_RAW=$(echo "$RESOURCE_DATA" | cut -d',' -f2)
            MEMORY_PERCENT=$(echo "$RESOURCE_DATA" | cut -d',' -f3 | sed 's/%//')
            # Convert memory to MB (handle GiB, MiB, etc.)
            MEMORY_USAGE_MB=$(echo "$MEMORY_USAGE_RAW" | sed 's/GiB/*1024/g; s/MiB//g; s/GiB//g; s/[^0-9.]//g' | bc 2>/dev/null || echo "0")
        fi
        
        # 7. Recent Log Activity with counting
        echo "" >> "$REPORT_FILE"
        echo "📋 RECENT LOG ACTIVITY:" >> "$REPORT_FILE"
        echo "Last 5 achievement messages:" >> "$REPORT_FILE"
        ACHIEVEMENTS=$(docker logs walker-training-app --since=5m 2>/dev/null | grep -E "(🎖️|🏆|🌟)" | tail -5)
        echo "$ACHIEVEMENTS" >> "$REPORT_FILE"
        ACHIEVEMENTS_COUNT=$(echo "$ACHIEVEMENTS" | wc -l)
        
        echo "" >> "$REPORT_FILE"
        echo "Last 5 system messages:" >> "$REPORT_FILE"
        ALERTS=$(docker logs walker-training-app --since=5m 2>/dev/null | grep -E "(🚨|⚠️|🔧|🌍)" | tail -5)
        echo "$ALERTS" >> "$REPORT_FILE"
        ALERTS_COUNT=$(echo "$ALERTS" | wc -l)
        
        # 8. Health Summary
        echo "" >> "$REPORT_FILE"
        echo "✅ HEALTH SUMMARY:" >> "$REPORT_FILE"
        HAS_ECOSYSTEM=$(echo "$METRICS" | jq -r '.has_ecosystem // false')
        HAS_ENVIRONMENT=$(echo "$METRICS" | jq -r '.has_environment // false')
        
        if [ "$AGENT_COUNT" != "unknown" ] && [ "$AGENT_COUNT" -gt 0 ]; then
            echo "✅ Agents Active: $AGENT_COUNT" >> "$REPORT_FILE"
        else
            echo "❌ No Agents Active" >> "$REPORT_FILE"
        fi
        
        if [ "$PHYSICS_FPS" != "unknown" ] && [ "$PHYSICS_FPS" -gt 0 ]; then
            echo "✅ Physics Running: ${PHYSICS_FPS} FPS" >> "$REPORT_FILE"
        else
            echo "❌ Physics Not Running" >> "$REPORT_FILE"
        fi
        
        if [ "$HAS_ECOSYSTEM" = "true" ]; then
            echo "✅ Ecosystem Active" >> "$REPORT_FILE"
        else
            echo "❌ Ecosystem Missing" >> "$REPORT_FILE"
        fi
        
        if [ "$HAS_ENVIRONMENT" = "true" ]; then
            echo "✅ Environment Data Available" >> "$REPORT_FILE"
        else
            echo "❌ Environment Data Missing" >> "$REPORT_FILE"
        fi
        
        HEALTH_STATUS="HEALTHY"
    else
        echo "❌ API Not Responding (Status: $API_STATUS)" >> "$REPORT_FILE"
        HEALTH_STATUS="UNHEALTHY"
    fi
    
    # 9. Append data to CSV file
    PROGRESS_PERCENT=$(echo "scale=1; $i * 100 / $TOTAL_CHECKS" | bc 2>/dev/null || echo "0")
    cat >> "$CSV_FILE" << EOF
$ISO_TIMESTAMP,$i,$PROGRESS_PERCENT,$API_STATUS,$HEALTH_STATUS,$AGENT_COUNT,$PHYSICS_FPS,$SIMULATION_SPEED,$TOP_REWARD,$AVG_REWARD,$OBSTACLE_COUNT,$FOOD_COUNT,$DYNAMIC_OBSTACLES,$CPU_PERCENT,$MEMORY_USAGE_MB,$MEMORY_PERCENT,$API_RESPONSE_TIME,$ACHIEVEMENTS_COUNT,$ALERTS_COUNT
EOF
    
    # 10. Update Summary File
    cat > "$SUMMARY_FILE" << EOF
{
    "last_check": "$CHECK_TIME",
    "check_number": $i,
    "total_checks": $TOTAL_CHECKS,
    "api_status": "$API_STATUS",
    "health_status": "$HEALTH_STATUS",
    "report_file": "$REPORT_FILE",
    "csv_file": "$CSV_FILE",
    "progress_percent": $PROGRESS_PERCENT,
    "current_metrics": {
        "agent_count": $AGENT_COUNT,
        "physics_fps": $PHYSICS_FPS,
        "top_reward": $TOP_REWARD,
        "avg_reward": $AVG_REWARD,
        "api_response_time_ms": $API_RESPONSE_TIME
    }
}
EOF
    
    echo "📊 Check $i complete - Status: $HEALTH_STATUS"
    echo "📋 CSV updated with performance data"
    
    # Sleep until next check (except for last check)
    if [ $i -lt $TOTAL_CHECKS ]; then
        echo "⏰ Waiting 5 minutes until next check..."
        sleep $CHECK_INTERVAL
    fi
done

# Final report summary
cat >> "$REPORT_FILE" << EOF

==========================================
MONITORING COMPLETED
==========================================
End Time: $(date)
Total Checks: $TOTAL_CHECKS
Report Duration: 1 hour

🎯 FINAL STATUS: Monitoring completed successfully.
📊 Full report saved to: $REPORT_FILE
📋 CSV performance data: $CSV_FILE
📈 Latest summary available at: $SUMMARY_FILE

To analyze performance trends:
  Open $CSV_FILE in Excel/LibreOffice
  Import into Python/R for analysis
  Use: head -1 $CSV_FILE && tail -5 $CSV_FILE

To view this report:
  cat $REPORT_FILE
  
To view summary:
  cat $SUMMARY_FILE | jq .

EOF

echo ""
echo "🎉 Monitoring completed!"
echo "📊 Full report: $REPORT_FILE" 
echo "📋 CSV data: $CSV_FILE"
echo "📈 Summary: $SUMMARY_FILE"
echo ""
echo "📈 PERFORMANCE DATA ANALYSIS:"
echo "  View CSV: cat $CSV_FILE"
echo "  Excel/Calc: Open $CSV_FILE"
echo "  Quick stats: head -1 $CSV_FILE && tail -5 $CSV_FILE" 