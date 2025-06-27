#!/bin/bash

# Walker System Health Monitoring Startup Script

echo "🔍 Starting Walker System Health Monitor..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if required Python packages are available
python3 -c "import requests" 2>/dev/null || {
    echo "❌ 'requests' package not found. Install with: pip3 install requests"
    exit 1
}

# Default settings
API_URL="http://localhost:7777"
PROMETHEUS_URL="http://localhost:9889" 
INTERVAL=30
AUTO_CLEANUP=""
MEMORY_WARNING=800
MEMORY_CRITICAL=1200
BODIES_WARNING=400
BODIES_CRITICAL=800

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto-cleanup)
            AUTO_CLEANUP="--auto-cleanup"
            echo "✅ Auto-cleanup enabled"
            shift
            ;;
        --aggressive-thresholds)
            MEMORY_WARNING=600
            MEMORY_CRITICAL=900
            BODIES_WARNING=300
            BODIES_CRITICAL=500
            echo "⚡ Using aggressive thresholds"
            shift
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --auto-cleanup           Enable automatic emergency cleanup"
            echo "  --aggressive-thresholds  Use lower thresholds for faster alerts"
            echo "  --interval SECONDS       Monitoring interval (default: 30)"
            echo "  --help                   Show this help"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Walker API is available
echo "🔗 Checking Walker API at $API_URL..."
if ! curl -s --connect-timeout 5 "$API_URL/performance_status" > /dev/null; then
    echo "⚠️ WARNING: Walker API not responding at $API_URL"
    echo "   Make sure the Walker training system is running"
    echo "   Starting monitor anyway (will retry connections)..."
fi

# Check if Prometheus is available
echo "🔗 Checking Prometheus at $PROMETHEUS_URL..."
if ! curl -s --connect-timeout 5 "$PROMETHEUS_URL/api/v1/query?query=up" > /dev/null; then
    echo "⚠️ WARNING: Prometheus not responding at $PROMETHEUS_URL"
    echo "   Monitor will work with limited functionality"
fi

echo ""
echo "📊 Starting system health monitor with settings:"
echo "   API URL: $API_URL"
echo "   Prometheus URL: $PROMETHEUS_URL"
echo "   Interval: ${INTERVAL}s"
echo "   Memory thresholds: ${MEMORY_WARNING}MB (warning), ${MEMORY_CRITICAL}MB (critical)"
echo "   Bodies thresholds: ${BODIES_WARNING} (warning), ${BODIES_CRITICAL} (critical)"
echo "   Auto-cleanup: $([ -n "$AUTO_CLEANUP" ] && echo "enabled" || echo "disabled")"
echo ""

# Start the monitor
python3 monitor_system_health.py \
    --api-url "$API_URL" \
    --prometheus-url "$PROMETHEUS_URL" \
    --interval "$INTERVAL" \
    --memory-warning "$MEMORY_WARNING" \
    --memory-critical "$MEMORY_CRITICAL" \
    --bodies-warning "$BODIES_WARNING" \
    --bodies-critical "$BODIES_CRITICAL" \
    $AUTO_CLEANUP

echo "🛑 System Health Monitor stopped" 