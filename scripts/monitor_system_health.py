#!/usr/bin/env python3
"""
Walker System Health Monitor
Continuously monitors system metrics to identify memory leaks and performance issues.
"""

import time
import json
import requests
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, deque
import signal
import sys

class SystemHealthMonitor:
    def __init__(self, api_url="http://localhost:7777", 
                 prometheus_url="http://localhost:9889",
                 alert_thresholds=None):
        self.api_url = api_url
        self.prometheus_url = prometheus_url
        self.running = True
        
        # Default alert thresholds
        self.thresholds = alert_thresholds or {
            'memory_mb': {'warning': 1000, 'critical': 1500},
            'world_bodies': {'warning': 500, 'critical': 1000},
            'food_sources': {'warning': 30, 'critical': 50},
            'static_bodies': {'warning': 200, 'critical': 300},
            'dynamic_bodies': {'warning': 150, 'critical': 250},
        }
        
        # Track metrics history for trend analysis
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))  # Keep last 100 readings
        self.alerts_sent = set()  # Prevent spam
        self.growth_rates = {}
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def get_metrics(self):
        """Get current metrics from the Walker API."""
        try:
            response = requests.get(f"{self.api_url}/performance_status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Failed to get metrics: {e}")
            return None
    
    def get_prometheus_metric(self, metric_name):
        """Get a specific metric from Prometheus."""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {'query': metric_name}
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data['data']['result']:
                    return float(data['data']['result'][0]['value'][1])
            return None
        except Exception as e:
            print(f"⚠️ Prometheus error for {metric_name}: {e}")
            return None
    
    def analyze_trends(self, metric_name, values):
        """Analyze metric trends to detect growth patterns."""
        if len(values) < 5:  # Need at least 5 data points
            return None
        
        # Calculate growth rate (per minute)
        recent_values = list(values)[-5:]  # Last 5 readings
        time_span = len(recent_values) * 30 / 60  # Assuming 30s intervals, convert to minutes
        
        if time_span > 0:
            growth = recent_values[-1] - recent_values[0]
            growth_rate = growth / time_span
            
            # Detect concerning trends
            if metric_name == 'world_bodies' and growth_rate > 10:  # >10 bodies per minute
                return f"🚨 CRITICAL: {metric_name} growing at {growth_rate:.1f}/min"
            elif metric_name == 'memory_mb' and growth_rate > 20:  # >20MB per minute
                return f"⚠️ WARNING: {metric_name} growing at {growth_rate:.1f}MB/min"
            elif metric_name == 'food_sources' and growth_rate > 5:  # >5 food sources per minute
                return f"⚠️ WARNING: {metric_name} growing at {growth_rate:.1f}/min"
            
            self.growth_rates[metric_name] = growth_rate
        
        return None
    
    def check_alerts(self, metrics):
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if not metrics:
            return alerts
        
        # Extract values
        values = {
            'memory_mb': metrics.get('performance', {}).get('memory_mb', 0),
            'world_bodies': metrics.get('entities', {}).get('world_bodies', 0),
            'food_sources': metrics.get('entities', {}).get('food_sources', 0),
        }
        
        # Get detailed body type breakdown from Prometheus
        for body_type in ['static', 'dynamic']:
            count = self.get_prometheus_metric(f'walker_body_types_total{{type="{body_type}"}}')
            if count is not None:
                values[f'{body_type}_bodies'] = count
        
        # Check thresholds
        for metric, value in values.items():
            if metric in self.thresholds:
                thresholds = self.thresholds[metric]
                alert_key = f"{metric}_{int(value)}"
                
                if value >= thresholds['critical'] and f"critical_{alert_key}" not in self.alerts_sent:
                    alerts.append(f"{timestamp} 🚨 CRITICAL: {metric} = {value} (>= {thresholds['critical']})")
                    self.alerts_sent.add(f"critical_{alert_key}")
                elif value >= thresholds['warning'] and f"warning_{alert_key}" not in self.alerts_sent:
                    alerts.append(f"{timestamp} ⚠️ WARNING: {metric} = {value} (>= {thresholds['warning']})")
                    self.alerts_sent.add(f"warning_{alert_key}")
        
        return alerts
    
    def analyze_memory_leak_source(self):
        """Analyze which body types are growing to identify memory leak source."""
        analysis = []
        
        # Check body type growth rates
        for body_type in ['static', 'dynamic', 'terrain']:
            metric_name = f'walker_body_types_total{{type="{body_type}"}}'
            current_count = self.get_prometheus_metric(metric_name)
            
            if current_count is not None:
                # Store in history
                self.metrics_history[f'{body_type}_bodies'].append(current_count)
                
                # Analyze trends
                trend_alert = self.analyze_trends(f'{body_type}_bodies', self.metrics_history[f'{body_type}_bodies'])
                if trend_alert:
                    analysis.append(trend_alert)
        
        return analysis
    
    def force_cleanup(self):
        """Trigger emergency cleanup via API."""
        try:
            response = requests.post(f"{self.api_url}/force_cleanup", timeout=10)
            if response.status_code == 200:
                data = response.json()
                memory_saved = data.get('memory_saved_mb', 0)
                print(f"🧹 Emergency cleanup completed: {memory_saved}MB freed")
                return True
            else:
                print(f"❌ Cleanup failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
            return False
    
    def monitor_loop(self, interval=30, auto_cleanup=False):
        """Main monitoring loop."""
        print(f"🔍 Starting Walker System Health Monitor")
        print(f"   API: {self.api_url}")
        print(f"   Prometheus: {self.prometheus_url}")
        print(f"   Interval: {interval}s")
        print(f"   Auto-cleanup: {auto_cleanup}")
        print(f"   Press Ctrl+C to stop\n")
        
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Get current metrics
                metrics = self.get_metrics()
                if not metrics:
                    print(f"{timestamp} ❌ Failed to get metrics")
                    time.sleep(interval)
                    continue
                
                # Extract key values
                perf = metrics.get('performance', {})
                entities = metrics.get('entities', {})
                
                memory_mb = perf.get('memory_mb', 0)
                world_bodies = entities.get('world_bodies', 0)
                food_sources = entities.get('food_sources', 0)
                step_count = perf.get('step_count', 0)
                
                # Store in history for trend analysis
                self.metrics_history['memory_mb'].append(memory_mb)
                self.metrics_history['world_bodies'].append(world_bodies)
                self.metrics_history['food_sources'].append(food_sources)
                
                # Print current status
                print(f"{timestamp} Status: Memory={memory_mb:.1f}MB, Bodies={world_bodies}, Food={food_sources}, Step={step_count}")
                
                # Check for alerts
                alerts = self.check_alerts(metrics)
                for alert in alerts:
                    print(alert)
                
                # Analyze memory leak sources
                leak_analysis = self.analyze_memory_leak_source()
                for analysis in leak_analysis:
                    print(f"{timestamp} {analysis}")
                
                # Auto-cleanup if enabled and thresholds exceeded
                if auto_cleanup and (memory_mb > 1200 or world_bodies > 800):
                    print(f"{timestamp} 🆘 Triggering auto-cleanup (Memory: {memory_mb}MB, Bodies: {world_bodies})")
                    if self.force_cleanup():
                        # Clear some alerts after successful cleanup
                        self.alerts_sent = {alert for alert in self.alerts_sent if not alert.startswith('warning_')}
                
                # Print growth rates every 10 iterations
                if iteration % 10 == 0 and self.growth_rates:
                    print(f"{timestamp} Growth Rates: " + 
                          ", ".join([f"{k}={v:.1f}/min" for k, v in self.growth_rates.items()]))
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{timestamp} ❌ Monitor error: {e}")
                time.sleep(interval)
        
        print("🛑 System Health Monitor stopped")

def main():
    parser = argparse.ArgumentParser(description='Walker System Health Monitor')
    parser.add_argument('--api-url', default='http://localhost:7777',
                       help='Walker API URL (default: http://localhost:7777)')
    parser.add_argument('--prometheus-url', default='http://localhost:9889',
                       help='Prometheus URL (default: http://localhost:9889)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Monitoring interval in seconds (default: 30)')
    parser.add_argument('--auto-cleanup', action='store_true',
                       help='Enable automatic emergency cleanup')
    parser.add_argument('--memory-warning', type=int, default=1000,
                       help='Memory warning threshold in MB (default: 1000)')
    parser.add_argument('--memory-critical', type=int, default=1500,
                       help='Memory critical threshold in MB (default: 1500)')
    parser.add_argument('--bodies-warning', type=int, default=500,
                       help='World bodies warning threshold (default: 500)')
    parser.add_argument('--bodies-critical', type=int, default=1000,
                       help='World bodies critical threshold (default: 1000)')
    
    args = parser.parse_args()
    
    # Setup custom thresholds
    thresholds = {
        'memory_mb': {'warning': args.memory_warning, 'critical': args.memory_critical},
        'world_bodies': {'warning': args.bodies_warning, 'critical': args.bodies_critical},
        'food_sources': {'warning': 30, 'critical': 50},
        'static_bodies': {'warning': 200, 'critical': 300},
        'dynamic_bodies': {'warning': 150, 'critical': 250},
    }
    
    # Create and start monitor
    monitor = SystemHealthMonitor(
        api_url=args.api_url,
        prometheus_url=args.prometheus_url,
        alert_thresholds=thresholds
    )
    
    monitor.monitor_loop(interval=args.interval, auto_cleanup=args.auto_cleanup)

if __name__ == "__main__":
    main() 