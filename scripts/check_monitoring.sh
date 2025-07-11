#!/bin/bash

# 📊 Walker Monitoring Status Checker
# Quick script to check monitoring progress and view reports

REPORT_DIR="monitoring_reports"

echo "🔍 WALKER SYSTEM MONITORING STATUS"
echo "=================================="

# Check if monitoring is running
if pgrep -f "monitor_system.sh" > /dev/null; then
    echo "✅ Monitoring script is RUNNING"
    echo "📊 Process ID: $(pgrep -f monitor_system.sh)"
    echo ""
    
    # Show current status
    if [ -f "$REPORT_DIR/latest_summary.json" ]; then
        echo "📈 CURRENT STATUS:"
        cat "$REPORT_DIR/latest_summary.json" | jq .
        echo ""
        
        # Show recent performance trends from CSV
        CSV_FILE=$(cat "$REPORT_DIR/latest_summary.json" | jq -r '.csv_file // empty')
        if [ -n "$CSV_FILE" ] && [ -f "$CSV_FILE" ]; then
            echo "📋 RECENT PERFORMANCE TRENDS:"
            echo "Last 3 data points from CSV:"
            echo "Check#, Agent Count, Physics FPS, Top Reward, Avg Reward, Response Time"
            tail -3 "$CSV_FILE" | awk -F',' '{printf "  %s: %s agents, %s FPS, %.1f top reward, %.1f avg reward, %sms\n", $2, $6, $7, $9, $10, $16}'
            echo ""
        fi
    fi
    
    # Show recent output
    echo "📋 RECENT MONITORING OUTPUT:"
    tail -n 10 monitoring_output.log 2>/dev/null || echo "No output log found"
    echo ""
    
else
    echo "❌ Monitoring script is NOT running"
    echo ""
fi

# List available reports and CSV files
echo "📁 AVAILABLE REPORTS:"
if [ -d "$REPORT_DIR" ]; then
    echo "  Log Reports:"
    ls -lt "$REPORT_DIR"/*.log 2>/dev/null | head -3 | while read line; do
        echo "    $line"
    done
    
    echo "  CSV Performance Data:"
    ls -lt "$REPORT_DIR"/*.csv 2>/dev/null | head -3 | while read line; do
        echo "    $line"
    done
    echo ""
    
    # Show report commands
    LATEST_REPORT=$(ls -t "$REPORT_DIR"/*.log 2>/dev/null | head -1)
    LATEST_CSV=$(ls -t "$REPORT_DIR"/*.csv 2>/dev/null | head -1)
    
    if [ -n "$LATEST_REPORT" ]; then
        echo "🔧 USEFUL COMMANDS:"
        echo "  View latest report:     cat '$LATEST_REPORT'"
        echo "  View current summary:   cat '$REPORT_DIR/latest_summary.json' | jq ."
        echo "  Monitor in real-time:   tail -f monitoring_output.log"
        echo ""
    fi
    
    if [ -n "$LATEST_CSV" ]; then
        echo "📊 CSV DATA ANALYSIS:"
        echo "  View CSV headers:       head -1 '$LATEST_CSV'"
        echo "  View recent data:       tail -5 '$LATEST_CSV'"
        echo "  Quick trend view:       tail -10 '$LATEST_CSV' | column -t -s,"
        echo "  Open in spreadsheet:    Open '$LATEST_CSV' with Excel/LibreOffice"
        echo ""
        
        # Show CSV column info
        if [ -f "$LATEST_CSV" ]; then
            echo "📋 CSV COLUMNS AVAILABLE:"
            head -1 "$LATEST_CSV" | tr ',' '\n' | nl | sed 's/^/    /'
            echo ""
            
            # Show basic stats if file has data
            DATA_LINES=$(tail -n +2 "$LATEST_CSV" | wc -l)
            if [ "$DATA_LINES" -gt 0 ]; then
                echo "📈 QUICK STATS (from $DATA_LINES data points):"
                
                # Get agent count range
                AGENT_MIN=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f6 | sort -n | head -1)
                AGENT_MAX=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f6 | sort -n | tail -1)
                echo "  Agent Count: $AGENT_MIN - $AGENT_MAX"
                
                # Get physics FPS range
                FPS_MIN=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f7 | sort -n | head -1)
                FPS_MAX=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f7 | sort -n | tail -1)
                echo "  Physics FPS: $FPS_MIN - $FPS_MAX"
                
                # Get response time range
                RT_MIN=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f16 | sort -n | head -1)
                RT_MAX=$(tail -n +2 "$LATEST_CSV" | cut -d',' -f16 | sort -n | tail -1)
                echo "  API Response: ${RT_MIN}ms - ${RT_MAX}ms"
                
                # Get top reward trend
                TOP_REWARD_FIRST=$(tail -n +2 "$LATEST_CSV" | head -1 | cut -d',' -f9)
                TOP_REWARD_LAST=$(tail -n +2 "$LATEST_CSV" | tail -1 | cut -d',' -f9)
                echo "  Top Reward: $TOP_REWARD_FIRST → $TOP_REWARD_LAST"
                echo ""
            fi
        fi
    fi
    
    echo "🔧 MONITORING CONTROL:"
    echo "  Check if running:       pgrep -f monitor_system.sh"
    echo "  Stop monitoring:        pkill -f monitor_system.sh"
    echo "  Start new monitoring:   nohup ./monitor_system.sh > monitoring_output.log 2>&1 &"
    echo ""
else
    echo "  No reports directory found"
    echo ""
fi

# Show system quick check
echo "🚀 QUICK SYSTEM CHECK:"
CONTAINER_STATUS=$(docker ps --filter name=walker-training-app --format "{{.Status}}" 2>/dev/null)
API_STATUS=$(curl -s -w "%{http_code}" http://localhost:7777/status -o /dev/null 2>/dev/null || echo "FAILED")

if [ -n "$CONTAINER_STATUS" ]; then
    echo "  Container: $CONTAINER_STATUS"
else
    echo "  Container: NOT RUNNING"
fi

if [ "$API_STATUS" = "200" ]; then
    echo "  API: RESPONDING"
    AGENT_COUNT=$(curl -s http://localhost:7777/status 2>/dev/null | jq -r '.agents | length' 2>/dev/null || echo "unknown")
    PHYSICS_FPS=$(curl -s http://localhost:7777/status 2>/dev/null | jq -r '.physics_fps' 2>/dev/null || echo "unknown")
    echo "  Agents: $AGENT_COUNT"
    echo "  Physics: $PHYSICS_FPS FPS"
else
    echo "  API: NOT RESPONDING ($API_STATUS)"
fi

echo ""
echo "📍 For detailed analysis, use the CSV commands above"
echo "📍 To re-run this check: ./check_monitoring.sh" 