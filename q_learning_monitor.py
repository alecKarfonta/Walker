#!/usr/bin/env python3
"""
Q-Learning Performance Monitor
Monitor and display Q-learning performance metrics for different agent types.
"""

import time
import json
import requests
from datetime import datetime
from typing import Dict, Any


def get_q_learning_status(api_url: str = "http://localhost:7777") -> Dict[str, Any]:
    """Get Q-learning evaluation status from the API."""
    try:
        response = requests.get(f"{api_url}/q_learning_status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {'status': 'error', 'message': f'API returned {response.status_code}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to connect: {e}'}


def get_q_learning_summary(api_url: str = "http://localhost:7777") -> Dict[str, Any]:
    """Get comprehensive Q-learning summary from the API."""
    try:
        response = requests.get(f"{api_url}/q_learning_summary", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {'status': 'error', 'message': f'API returned {response.status_code}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to connect: {e}'}


def get_agent_diagnostics(agent_id: str, api_url: str = "http://localhost:7777") -> Dict[str, Any]:
    """Get Q-learning diagnostics for a specific agent."""
    try:
        response = requests.get(f"{api_url}/q_learning_agent/{agent_id}/diagnostics", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {'status': 'error', 'message': f'API returned {response.status_code}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to connect: {e}'}


def get_type_comparison(api_url: str = "http://localhost:7777") -> Dict[str, Any]:
    """Get comparative analysis across agent types."""
    try:
        response = requests.get(f"{api_url}/q_learning_comparison", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {'status': 'error', 'message': f'API returned {response.status_code}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to connect: {e}'}


def format_learning_health(health: str) -> str:
    """Format learning health with emoji indicators."""
    health_icons = {
        'excellent': '🟢',
        'good': '🟡',
        'fair': '��',
        'needs_attention': '🔴',
        'unknown': '⚪'
    }
    return f"{health_icons.get(health, '⚪')} {health.title()}"


def display_summary_report(summary: Dict[str, Any]):
    """Display a comprehensive summary report."""
    print("\n" + "="*80)
    print("🧠 Q-LEARNING PERFORMANCE SUMMARY")
    print("="*80)
    
    if summary.get('status') == 'error':
        print(f"❌ Error: {summary.get('message', 'Unknown error')}")
        return
    
    if summary.get('status') == 'no_data':
        print("📊 No Q-learning data available yet")
        return
    
    # Overall statistics
    stats = summary.get('overall_statistics', {})
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Agents Evaluated: {summary.get('total_agents_evaluated', 0)}")
    print(f"   Avg Prediction MAE: {stats.get('avg_prediction_mae', 0):.4f}")
    print(f"   Avg Efficiency Score: {stats.get('avg_efficiency_score', 0):.3f}")
    print(f"   Avg Convergence Score: {stats.get('avg_convergence_score', 0):.3f}")
    
    accuracy_range = stats.get('prediction_accuracy_range', [0, 0])
    print(f"   Prediction Accuracy Range: {accuracy_range[0]:.4f} - {accuracy_range[1]:.4f}")
    
    # Best performers
    best = summary.get('best_performers', {})
    if best:
        print(f"\n🏆 BEST PERFORMERS:")
        
        most_accurate = best.get('most_accurate', {})
        if most_accurate:
            print(f"   Most Accurate: Agent {most_accurate.get('agent_id', 'N/A')[:8]} ({most_accurate.get('agent_type', 'unknown')})")
            print(f"                  MAE: {most_accurate.get('mae', 0):.4f}")
        
        most_efficient = best.get('most_efficient', {})
        if most_efficient:
            print(f"   Most Efficient: Agent {most_efficient.get('agent_id', 'N/A')[:8]} ({most_efficient.get('agent_type', 'unknown')})")
            print(f"                   Efficiency: {most_efficient.get('efficiency_score', 0):.3f}")
        
        best_convergence = best.get('best_convergence', {})
        if best_convergence:
            print(f"   Best Convergence: Agent {best_convergence.get('agent_id', 'N/A')[:8]} ({best_convergence.get('agent_type', 'unknown')})")
            print(f"                     Convergence: {best_convergence.get('convergence_score', 0):.3f}")
    
    # Agent type comparison
    type_comparison = summary.get('agent_type_comparison', {})
    if type_comparison:
        print(f"\n🔄 AGENT TYPE COMPARISON:")
        for agent_type, metrics in type_comparison.items():
            print(f"   {agent_type.replace('_', ' ').title()}:")
            print(f"     Agents: {metrics.get('agent_count', 0)}")
            print(f"     Avg Prediction MAE: {metrics.get('avg_prediction_mae', 0):.4f}")
            print(f"     Avg Convergence: {metrics.get('avg_convergence_score', 0):.3f}")
            print(f"     Avg Efficiency: {metrics.get('avg_efficiency_score', 0):.3f}")
            print(f"     Avg Steps to First Reward: {metrics.get('avg_steps_to_first_reward', -1):.0f}")
            
            common_issues = metrics.get('common_issues', [])
            if common_issues:
                print(f"     Common Issues: {', '.join(common_issues)}")
    
    # System health
    health = summary.get('system_health', {})
    if health:
        print(f"\n💚 SYSTEM HEALTH:")
        print(f"   Agents with Issues: {health.get('agents_with_issues', 0)}")
        print(f"   Agents Learning Well: {health.get('agents_learning_well', 0)}")
        print(f"   Agents Converged: {health.get('agents_converged', 0)}")


def display_type_comparison(comparison: Dict[str, Any]):
    """Display agent type comparison in a focused format."""
    if comparison.get('status') == 'error':
        print(f"❌ Error: {comparison.get('message', 'Unknown error')}")
        return
    
    comp_data = comparison.get('comparison', {})
    if not comp_data:
        print("📊 No agent type comparison data available")
        return
    
    print("\n" + "="*60)
    print("🔄 AGENT TYPE PERFORMANCE COMPARISON")
    print("="*60)
    
    # Create a sorted list of agent types by efficiency
    type_list = []
    for agent_type, metrics in comp_data.items():
        type_list.append((agent_type, metrics))
    
    # Sort by efficiency score (descending)
    type_list.sort(key=lambda x: x[1].get('avg_efficiency_score', 0), reverse=True)
    
    for rank, (agent_type, metrics) in enumerate(type_list, 1):
        print(f"\n{rank}. {agent_type.replace('_', ' ').title()}")
        print(f"   📊 Agents: {metrics.get('agent_count', 0)}")
        print(f"   🎯 Prediction Accuracy: {1.0 - metrics.get('avg_prediction_mae', 1.0):.3f}")
        print(f"   ⚡ Learning Efficiency: {metrics.get('avg_efficiency_score', 0):.3f}")
        print(f"   🎲 Convergence Score: {metrics.get('avg_convergence_score', 0):.3f}")
        print(f"   🚀 Learning Velocity: {metrics.get('avg_learning_velocity', 0):.4f}")
        
        avg_steps = metrics.get('avg_steps_to_first_reward', -1)
        if avg_steps > 0:
            print(f"   ⏱️  Steps to First Reward: {avg_steps:.0f}")
        
        issues = metrics.get('common_issues', [])
        if issues:
            print(f"   ⚠️  Common Issues: {', '.join(issues)}")


def monitor_q_learning_performance(api_url: str = "http://localhost:7777", 
                                 interval: int = 60, show_agents: bool = False):
    """
    Continuously monitor Q-learning performance.
    
    Args:
        api_url: API endpoint URL
        interval: Update interval in seconds
        show_agents: Whether to show individual agent diagnostics
    """
    print(f"🧠 Starting Q-Learning Performance Monitor")
    print(f"   API: {api_url}")
    print(f"   Update interval: {interval}s")
    print(f"   Show agent details: {show_agents}")
    print(f"   Press Ctrl+C to stop\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n🔄 Update #{iteration} at {timestamp}")
            
            # Get comprehensive summary
            summary = get_q_learning_summary(api_url)
            if summary.get('status') == 'success':
                display_summary_report(summary.get('summary', {}))
            else:
                print(f"❌ Failed to get Q-learning summary: {summary.get('message', 'Unknown error')}")
            
            # Get type comparison
            comparison = get_type_comparison(api_url)
            if comparison.get('status') == 'success':
                display_type_comparison(comparison)
            else:
                print(f"❌ Failed to get type comparison: {comparison.get('message', 'Unknown error')}")
            
            # Show individual agent diagnostics if requested
            if show_agents:
                # Get overall status to find agent IDs
                status = get_q_learning_status(api_url)
                if status.get('status') == 'success' and 'integration_status' in status:
                    agents_monitored = status['integration_status'].get('agents_monitored', 0)
                    print(f"\n👥 Individual Agent Diagnostics ({agents_monitored} agents monitored):")
                    
                    # Note: In a real implementation, you'd need a way to get agent IDs
                    # This is just a placeholder showing the structure
                    print("   (Individual agent diagnostics would be shown here)")
                    print("   (Use specific agent IDs with get_agent_diagnostics() function)")
            
            print(f"\n⏰ Next update in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")


def main():
    """Main function with CLI options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Q-Learning Performance Monitor')
    parser.add_argument('--api-url', default='http://localhost:7777',
                       help='Walker API URL (default: http://localhost:7777)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in seconds (default: 60)')
    parser.add_argument('--show-agents', action='store_true',
                       help='Show individual agent diagnostics')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    parser.add_argument('--summary', action='store_true',
                       help='Show only summary report')
    parser.add_argument('--comparison', action='store_true',
                       help='Show only type comparison')
    
    args = parser.parse_args()
    
    if args.once or args.summary or args.comparison:
        # Single run mode
        if args.summary or (not args.comparison):
            summary = get_q_learning_summary(args.api_url)
            if summary.get('status') == 'success':
                display_summary_report(summary.get('summary', {}))
            else:
                print(f"❌ Failed to get summary: {summary.get('message', 'Unknown error')}")
        
        if args.comparison or (not args.summary):
            comparison = get_type_comparison(args.api_url)
            if comparison.get('status') == 'success':
                display_type_comparison(comparison)
            else:
                print(f"❌ Failed to get comparison: {comparison.get('message', 'Unknown error')}")
    else:
        # Continuous monitoring mode
        monitor_q_learning_performance(
            api_url=args.api_url,
            interval=args.interval,
            show_agents=args.show_agents
        )


if __name__ == "__main__":
    main()
