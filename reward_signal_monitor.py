#!/usr/bin/env python3
"""
Reward Signal Quality Monitor
Real-time monitoring tool for reward signal evaluation.
"""

import time
import json
import requests
from typing import Dict, Any, Optional
import argparse


class RewardSignalMonitor:
    """Monitor reward signal quality in real-time."""
    
    def __init__(self, api_url: str = "http://localhost:7777"):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        
    def get_reward_signal_status(self) -> Optional[Dict[str, Any]]:
        """Get current reward signal evaluation status."""
        try:
            response = self.session.get(f"{self.api_url}/reward_signal_status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting reward signal status: {e}")
            return None
    
    def get_reward_signal_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive reward signal quality summary."""
        try:
            response = self.session.get(f"{self.api_url}/reward_signal_summary", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting reward signal summary: {e}")
            return None
    
    def get_agent_diagnostics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed diagnostics for a specific agent."""
        try:
            response = self.session.get(f"{self.api_url}/reward_signal_agent/{agent_id}/diagnostics", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting agent diagnostics: {e}")
            return None
    
    def get_reward_comparison(self) -> Optional[Dict[str, Any]]:
        """Get comparative analysis of reward quality."""
        try:
            response = self.session.get(f"{self.api_url}/reward_signal_comparison", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting reward comparison: {e}")
            return None
    
    def print_status_summary(self, status: Dict[str, Any]):
        """Print a formatted status summary."""
        print("\n" + "="*60)
        print("📊 REWARD SIGNAL QUALITY STATUS")
        print("="*60)
        
        print(f"Status: {'🟢 Active' if status.get('active', False) else '🔴 Inactive'}")
        print(f"Total Agents Tracked: {status.get('total_agents', 0)}")
        print(f"Agents with Metrics: {status.get('agents_with_metrics', 0)}")
        print(f"Total Rewards Recorded: {status.get('total_rewards_recorded', 0)}")
        
        if status.get('registered_agents'):
            print(f"\n📝 Registered Agents:")
            for agent_id, info in status['registered_agents'].items():
                last_reward_time = time.time() - info.get('last_reward_time', 0)
                print(f"  • {agent_id} ({info['type']}): {info['reward_count']} rewards, "
                      f"last: {last_reward_time:.1f}s ago")
    
    def print_quality_summary(self, summary: Dict[str, Any]):
        """Print a formatted quality summary."""
        print("\n" + "="*60)
        print("🎯 REWARD SIGNAL QUALITY SUMMARY")
        print("="*60)
        
        if summary.get('status') == 'no_data':
            print("⚠️  No reward signal data available")
            return
        
        stats = summary.get('overall_statistics', {})
        print(f"Average Quality Score: {stats.get('avg_quality_score', 0):.3f}")
        print(f"Average Signal-to-Noise Ratio: {stats.get('avg_snr', 0):.3f}")
        print(f"Average Consistency: {stats.get('avg_consistency', 0):.3f}")
        print(f"Quality Range: {stats.get('quality_range', [0, 0])}")
        
        print(f"\n🏆 Best Performer:")
        best = summary.get('best_performer', {})
        print(f"  Agent: {best.get('agent_id', 'N/A')}")
        print(f"  Quality Score: {best.get('quality_score', 0):.3f}")
        print(f"  SNR: {best.get('snr', 0):.3f}")
        
        print(f"\n⚠️  Worst Performer:")
        worst = summary.get('worst_performer', {})
        print(f"  Agent: {worst.get('agent_id', 'N/A')}")
        print(f"  Quality Score: {worst.get('quality_score', 0):.3f}")
        print(f"  Issues: {', '.join(worst.get('issues', []))}")
        
        print(f"\n🔍 Common Issues:")
        for issue, count in summary.get('common_issues', {}).items():
            print(f"  • {issue.replace('_', ' ').title()}: {count} agents")
        
        recommendations = summary.get('system_recommendations', [])
        if recommendations:
            print(f"\n💡 System Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
    
    def print_quality_comparison(self, comparison: Dict[str, Any]):
        """Print quality tier comparison."""
        print("\n" + "="*60)
        print("📈 REWARD QUALITY TIER ANALYSIS")
        print("="*60)
        
        if comparison.get('status') == 'no_data':
            print("⚠️  No reward signal data available")
            return
        
        tier_counts = comparison.get('tier_counts', {})
        total_agents = comparison.get('total_agents', 0)
        
        print(f"Total Agents Analyzed: {total_agents}")
        print(f"\n🏆 Quality Distribution:")
        
        tier_emojis = {
            'excellent': '🟢',
            'good': '🔵', 
            'fair': '🟡',
            'poor': '🟠',
            'very_poor': '🔴'
        }
        
        for tier, count in tier_counts.items():
            if count > 0:
                percentage = (count / total_agents) * 100 if total_agents > 0 else 0
                emoji = tier_emojis.get(tier, '⚫')
                print(f"  {emoji} {tier.replace('_', ' ').title()}: {count} agents ({percentage:.1f}%)")
        
        # Show top agents in each tier
        quality_tiers = comparison.get('quality_tiers', {})
        for tier_name, agents in quality_tiers.items():
            if agents and tier_name in ['excellent', 'good']:
                print(f"\n{tier_emojis.get(tier_name, '⚫')} {tier_name.replace('_', ' ').title()} Agents:")
                for agent in agents[:3]:  # Show top 3
                    print(f"  • {agent['agent_id']}: Quality={agent['quality_score']:.3f}, "
                          f"SNR={agent['signal_to_noise_ratio']:.3f}")
                if len(agents) > 3:
                    print(f"  ...and {len(agents) - 3} more")
    
    def print_agent_diagnostics(self, agent_id: str, diagnostics: Dict[str, Any]):
        """Print detailed agent diagnostics."""
        print("\n" + "="*60)
        print(f"🤖 AGENT {agent_id} REWARD SIGNAL DIAGNOSTICS")
        print("="*60)
        
        if diagnostics.get('status') == 'insufficient_data':
            print("⚠️  Insufficient reward data for analysis")
            return
        
        if 'error' in diagnostics:
            print(f"❌ Error: {diagnostics['error']}")
            return
        
        analysis = diagnostics.get('reward_signal_analysis', {})
        overall = analysis.get('overall_quality', {})
        
        print(f"Agent Type: {diagnostics.get('agent_type', 'Unknown')}")
        print(f"Overall Quality: {overall.get('score', 0):.3f} ({overall.get('rating', 'Unknown')})")
        
        # Signal characteristics
        chars = analysis.get('signal_characteristics', {})
        print(f"\n📊 Signal Characteristics:")
        print(f"  Sparsity: {chars.get('sparsity', {}).get('value', 0):.3f} - {chars.get('sparsity', {}).get('interpretation', 'N/A')}")
        print(f"  Noise Level: {chars.get('noise_level', {}).get('snr', 0):.3f} - {chars.get('noise_level', {}).get('interpretation', 'N/A')}")
        print(f"  Consistency: {chars.get('consistency', {}).get('value', 0):.3f} - {chars.get('consistency', {}).get('interpretation', 'N/A')}")
        print(f"  Exploration: {chars.get('exploration_support', {}).get('value', 0):.3f} - {chars.get('exploration_support', {}).get('interpretation', 'N/A')}")
        
        # Learning implications
        learning = analysis.get('learning_implications', {})
        print(f"\n🧠 Learning Implications:")
        print(f"  Convergence Support: {learning.get('convergence_support', 0):.3f}")
        print(f"  Behavioral Alignment: {learning.get('behavioral_alignment', 0):.3f}")
        print(f"  Temporal Consistency: {learning.get('temporal_consistency', 0):.3f}")
        
        # Issues and recommendations
        issues = overall.get('issues', [])
        if issues:
            print(f"\n⚠️  Quality Issues:")
            for issue in issues:
                print(f"  • {issue.replace('_', ' ').title()}")
        
        recommendations = overall.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Data summary
        data_summary = diagnostics.get('data_summary', {})
        print(f"\n📈 Data Summary:")
        print(f"  Total Rewards: {data_summary.get('total_rewards_recorded', 0)}")
        print(f"  Non-zero Rewards: {data_summary.get('non_zero_rewards', 0)}")
        print(f"  Steps Analyzed: {data_summary.get('steps_analyzed', 0)}")
    
    def monitor_continuous(self, interval: int = 30):
        """Monitor reward signal quality continuously."""
        print("🔄 Starting continuous reward signal monitoring...")
        print(f"📊 Monitoring interval: {interval} seconds")
        print("⏹️  Press Ctrl+C to stop")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n{'='*20} ITERATION {iteration} {'='*20}")
                print(f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get status
                status = self.get_reward_signal_status()
                if status:
                    self.print_status_summary(status)
                
                # Get summary every 5 iterations (or when explicitly requested)
                if iteration % 5 == 1:
                    summary = self.get_reward_signal_summary()
                    if summary:
                        self.print_quality_summary(summary)
                
                print(f"\n⏳ Waiting {interval} seconds until next check...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
    
    def analyze_agent(self, agent_id: str):
        """Analyze a specific agent's reward signals."""
        print(f"🔍 Analyzing reward signals for agent: {agent_id}")
        
        diagnostics = self.get_agent_diagnostics(agent_id)
        if diagnostics:
            self.print_agent_diagnostics(agent_id, diagnostics)
        else:
            print(f"❌ Could not get diagnostics for agent {agent_id}")
    
    def show_comparison(self):
        """Show reward quality comparison across agents."""
        print("📊 Getting reward quality comparison...")
        
        comparison = self.get_reward_comparison()
        if comparison:
            self.print_quality_comparison(comparison)
        else:
            print("❌ Could not get reward quality comparison")
    
    def run_single_report(self):
        """Run a single comprehensive report."""
        print("📋 Generating comprehensive reward signal quality report...")
        
        # Status
        status = self.get_reward_signal_status()
        if status:
            self.print_status_summary(status)
        
        # Summary
        summary = self.get_reward_signal_summary()
        if summary:
            self.print_quality_summary(summary)
        
        # Comparison
        comparison = self.get_reward_comparison()
        if comparison:
            self.print_quality_comparison(comparison)
        
        print("\n✅ Report complete!")


def main():
    parser = argparse.ArgumentParser(description="Reward Signal Quality Monitor")
    parser.add_argument("--url", default="http://localhost:7777", help="API base URL")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Continuous monitoring')
    monitor_parser.add_argument("--interval", type=int, default=30, help="Monitoring interval")
    
    # Agent analysis command
    agent_parser = subparsers.add_parser('agent', help='Analyze specific agent')
    agent_parser.add_argument('agent_id', help='Agent ID to analyze')
    
    # Comparison command
    subparsers.add_parser('compare', help='Show reward quality comparison')
    
    # Report command
    subparsers.add_parser('report', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    monitor = RewardSignalMonitor(args.url)
    
    if args.command == 'monitor':
        monitor.monitor_continuous(args.interval)
    elif args.command == 'agent':
        monitor.analyze_agent(args.agent_id)
    elif args.command == 'compare':
        monitor.show_comparison()
    elif args.command == 'report':
        monitor.run_single_report()
    else:
        # Default: single report
        monitor.run_single_report()


if __name__ == "__main__":
    main() 