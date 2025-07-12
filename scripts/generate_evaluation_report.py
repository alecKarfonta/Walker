#!/usr/bin/env python3
"""
Evaluation Report Generator for Walker Robot Training System

Generates comprehensive learning performance reports by interfacing with:
- Running training environment (if active)
- MLflow experiment database
- Monitoring data
- Robot storage system

Usage:
    python scripts/generate_evaluation_report.py [--live] [--historical] [--export-dir DIR]
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import requests
from src.evaluation.metrics_collector import MetricsCollector
from src.evaluation.training_evaluator import TrainingProgressEvaluator
from src.evaluation.individual_evaluator import IndividualRobotEvaluator
from src.evaluation.q_learning_evaluator import QLearningEvaluator
from src.evaluation.population_evaluator import PopulationEvaluator

class EvaluationReportGenerator:
    """Generates comprehensive evaluation reports from multiple data sources."""
    
    def __init__(self, export_dir: str = "evaluation_exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize evaluators
        self.training_evaluator = TrainingProgressEvaluator()
        self.individual_evaluator = IndividualRobotEvaluator()
        self.q_learning_evaluator = QLearningEvaluator()
        self.population_evaluator = PopulationEvaluator()
        
        print(f"📊 Evaluation Report Generator initialized")
        print(f"📁 Export directory: {self.export_dir}")
    
    def check_system_status(self) -> Dict[str, Any]:
        """Check if the training system is currently running."""
        try:
            response = requests.get("http://localhost:2322/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'active': True,
                    'agent_count': data.get('statistics', {}).get('total_agents', 0),
                    'generation': data.get('statistics', {}).get('generation', 0),
                    'physics_fps': data.get('physics_fps', 0),
                    'data': data
                }
        except Exception as e:
            print(f"⚠️ Training system not accessible: {e}")
        
        return {'active': False}
    
    def generate_live_report(self) -> Optional[Dict[str, Any]]:
        """Generate report from live training system."""
        print("🔄 Generating live system report...")
        
        system_status = self.check_system_status()
        if not system_status['active']:
            print("❌ Cannot generate live report - training system not running")
            return None
        
        # Get comprehensive system data
        try:
            status_data = system_status['data']
            
            # Extract learning performance metrics
            agents_data = status_data.get('all_agents', [])
            population_stats = status_data.get('statistics', {})
            
            live_report = {
                'report_type': 'live_system',
                'timestamp': time.time(),
                'generation': system_status['generation'],
                'system_health': {
                    'active_agents': system_status['agent_count'],
                    'physics_fps': system_status['physics_fps'],
                    'simulation_running': system_status['physics_fps'] > 0
                },
                'learning_performance': self._analyze_agent_learning(agents_data),
                'population_metrics': self._analyze_population(population_stats),
                'ecosystem_status': status_data.get('ecosystem', {}),
                'recommendations': self._generate_recommendations(agents_data, population_stats)
            }
            
            # Save live report
            report_file = self.export_dir / f"live_report_{self.timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(live_report, f, indent=2, default=str)
            
            print(f"✅ Live report saved: {report_file}")
            return live_report
            
        except Exception as e:
            print(f"❌ Error generating live report: {e}")
            return None
    
    def generate_historical_report(self) -> Optional[Dict[str, Any]]:
        """Generate report from historical data sources."""
        print("📚 Generating historical data report...")
        
        historical_report = {
            'report_type': 'historical_analysis',
            'timestamp': time.time(),
            'data_sources': {},
            'analysis': {}
        }
        
        # Analyze monitoring data
        monitoring_data = self._analyze_monitoring_data()
        if monitoring_data:
            historical_report['data_sources']['monitoring'] = monitoring_data
            historical_report['analysis']['performance_trends'] = self._analyze_performance_trends(monitoring_data)
        
        # Analyze experiment database
        experiment_data = self._analyze_experiments()
        if experiment_data:
            historical_report['data_sources']['experiments'] = experiment_data
        
        # Analyze robot storage
        storage_data = self._analyze_robot_storage()
        if storage_data:
            historical_report['data_sources']['robot_storage'] = storage_data
        
        # Generate insights
        historical_report['insights'] = self._generate_historical_insights(historical_report['data_sources'])
        
        # Save historical report
        report_file = self.export_dir / f"historical_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(historical_report, f, indent=2, default=str)
        
        print(f"✅ Historical report saved: {report_file}")
        return historical_report
    
    def _analyze_agent_learning(self, agents_data: List[Dict]) -> Dict[str, Any]:
        """Analyze learning performance of individual agents."""
        if not agents_data:
            return {'status': 'no_agents', 'metrics': {}}
        
        learning_metrics = {
            'total_agents': len(agents_data),
            'learning_active': 0,
            'reward_distribution': [],
            'learning_approaches': {},
            'top_performers': [],
            'learning_issues': []
        }
        
        for agent in agents_data:
            try:
                # Check if agent is actively learning
                total_reward = agent.get('total_reward', 0)
                episode_reward = agent.get('episode_reward', 0)
                
                if total_reward > 0 or episode_reward > 0:
                    learning_metrics['learning_active'] += 1
                
                learning_metrics['reward_distribution'].append({
                    'agent_id': agent.get('id'),
                    'total_reward': total_reward,
                    'episode_reward': episode_reward
                })
                
                # Track learning approaches
                approach = agent.get('learning_approach', 'unknown')
                learning_metrics['learning_approaches'][approach] = learning_metrics['learning_approaches'].get(approach, 0) + 1
                
                # Identify top performers
                if total_reward > 100:  # Threshold for good performance
                    learning_metrics['top_performers'].append({
                        'agent_id': agent.get('id'),
                        'total_reward': total_reward,
                        'approach': approach
                    })
                
            except Exception as e:
                learning_metrics['learning_issues'].append(f"Agent analysis error: {e}")
        
        # Sort top performers
        learning_metrics['top_performers'].sort(key=lambda x: x['total_reward'], reverse=True)
        learning_metrics['top_performers'] = learning_metrics['top_performers'][:10]  # Top 10
        
        return learning_metrics
    
    def _analyze_population(self, population_stats: Dict) -> Dict[str, Any]:
        """Analyze population-level metrics."""
        return {
            'generation': population_stats.get('generation', 0),
            'diversity': population_stats.get('diversity', 0),
            'fitness_metrics': {
                'best_fitness': population_stats.get('best_fitness', 0),
                'average_fitness': population_stats.get('average_fitness', 0),
                'worst_fitness': population_stats.get('worst_fitness', 0)
            },
            'species_info': {
                'species_count': population_stats.get('species_count', 1),
                'hall_of_fame_size': population_stats.get('hall_of_fame_size', 0)
            },
            'learning_stats': population_stats.get('q_learning_stats', {})
        }
    
    def _analyze_monitoring_data(self) -> Optional[Dict[str, Any]]:
        """Analyze data from monitoring reports."""
        try:
            monitoring_dir = Path("monitoring_reports")
            if not monitoring_dir.exists():
                return None
            
            # Find latest CSV file
            csv_files = list(monitoring_dir.glob("performance_data_*.csv"))
            if not csv_files:
                return None
            
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            
            # Read and analyze CSV data
            with open(latest_csv, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None
            
            headers = lines[0].strip().split(',')
            data_points = []
            
            for line in lines[1:]:
                values = line.strip().split(',')
                if len(values) == len(headers):
                    data_points.append(dict(zip(headers, values)))
            
            return {
                'file': str(latest_csv),
                'data_points': len(data_points),
                'latest_data': data_points[-1] if data_points else None,
                'performance_summary': self._summarize_performance_data(data_points)
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing monitoring data: {e}")
            return None
    
    def _analyze_experiments(self) -> Optional[Dict[str, Any]]:
        """Analyze MLflow experiments database."""
        try:
            exp_db = Path("experiments/walker_experiments.db")
            if exp_db.exists():
                return {
                    'database_exists': True,
                    'database_size_mb': exp_db.stat().st_size / (1024 * 1024),
                    'last_modified': exp_db.stat().st_mtime,
                    'note': 'Requires SQLite tools for detailed analysis'
                }
            return None
        except Exception as e:
            print(f"⚠️ Error analyzing experiments: {e}")
            return None
    
    def _analyze_robot_storage(self) -> Optional[Dict[str, Any]]:
        """Analyze robot storage system."""
        try:
            storage_dir = Path("robot_storage")
            if not storage_dir.exists():
                return None
            
            robots_dir = storage_dir / "robots"
            backups_dir = storage_dir / "backups" 
            history_dir = storage_dir / "history"
            
            return {
                'storage_exists': True,
                'robots_stored': len(list(robots_dir.glob("*"))) if robots_dir.exists() else 0,
                'backups_count': len(list(backups_dir.glob("*"))) if backups_dir.exists() else 0,
                'history_entries': len(list(history_dir.glob("*"))) if history_dir.exists() else 0
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing robot storage: {e}")
            return None
    
    def _summarize_performance_data(self, data_points: List[Dict]) -> Dict[str, Any]:
        """Summarize performance trends from monitoring data."""
        if not data_points:
            return {}
        
        try:
            agent_counts = [int(d.get('agent_count', 0)) for d in data_points if d.get('agent_count', '').isdigit()]
            physics_fps = [int(d.get('physics_fps', 0)) for d in data_points if d.get('physics_fps', '').replace('.', '').isdigit()]
            
            return {
                'total_data_points': len(data_points),
                'agent_performance': {
                    'max_agents': max(agent_counts) if agent_counts else 0,
                    'agent_activity_periods': len([c for c in agent_counts if c > 0])
                },
                'physics_performance': {
                    'max_fps': max(physics_fps) if physics_fps else 0,
                    'avg_fps': sum(physics_fps) / len(physics_fps) if physics_fps else 0
                },
                'system_stability': {
                    'healthy_checks': len([d for d in data_points if d.get('health_status') == 'HEALTHY']),
                    'total_checks': len(data_points)
                }
            }
        except Exception as e:
            return {'error': f"Analysis failed: {e}"}
    
    def _analyze_performance_trends(self, monitoring_data: Dict) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        summary = monitoring_data.get('performance_summary', {})
        
        return {
            'learning_activity': {
                'peak_agents': summary.get('agent_performance', {}).get('max_agents', 0),
                'learning_periods': summary.get('agent_performance', {}).get('agent_activity_periods', 0)
            },
            'system_performance': {
                'peak_fps': summary.get('physics_performance', {}).get('max_fps', 0),
                'avg_fps': summary.get('physics_performance', {}).get('avg_fps', 0)
            },
            'stability_score': {
                'uptime_ratio': summary.get('system_stability', {}).get('healthy_checks', 0) / max(1, summary.get('system_stability', {}).get('total_checks', 1))
            }
        }
    
    def _generate_recommendations(self, agents_data: List[Dict], population_stats: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if not agents_data:
            recommendations.append("🚀 Start training - no active agents detected")
            return recommendations
        
        # Learning performance recommendations
        active_learners = len([a for a in agents_data if a.get('total_reward', 0) > 0])
        total_agents = len(agents_data)
        
        if active_learners / total_agents < 0.5:
            recommendations.append("🧠 Low learning activity - check learning parameters and reward systems")
        
        # Performance recommendations
        avg_reward = sum(a.get('total_reward', 0) for a in agents_data) / len(agents_data)
        if avg_reward < 100:
            recommendations.append("📈 Low average rewards - consider adjusting reward scaling or learning rates")
        
        # Population diversity
        diversity = population_stats.get('diversity', 0)
        if diversity < 0.3:
            recommendations.append("🌈 Low population diversity - increase mutation rates or population size")
        
        return recommendations
    
    def _generate_historical_insights(self, data_sources: Dict) -> List[str]:
        """Generate insights from historical data analysis."""
        insights = []
        
        # Monitoring insights
        if 'monitoring' in data_sources:
            monitoring = data_sources['monitoring']
            if monitoring.get('performance_summary', {}).get('agent_performance', {}).get('max_agents', 0) > 0:
                insights.append(f"📊 Peak learning activity: {monitoring['performance_summary']['agent_performance']['max_agents']} agents")
            
            stability = monitoring.get('performance_summary', {}).get('system_stability', {})
            if stability.get('healthy_checks', 0) > 0:
                uptime_ratio = stability['healthy_checks'] / max(1, stability.get('total_checks', 1))
                insights.append(f"⏱️ System uptime: {uptime_ratio*100:.1f}% healthy during monitoring")
        
        # Storage insights
        if 'robot_storage' in data_sources:
            storage = data_sources['robot_storage']
            if storage.get('robots_stored', 0) > 0:
                insights.append(f"🤖 {storage['robots_stored']} robots preserved in storage")
        
        # Database insights
        if 'experiments' in data_sources:
            exp = data_sources['experiments']
            if exp.get('database_exists'):
                insights.append(f"📈 Experiment database active ({exp.get('database_size_mb', 0):.1f} MB)")
        
        return insights
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive report combining live and historical data."""
        print("📋 Generating comprehensive evaluation report...")
        
        comprehensive_report = {
            'report_type': 'comprehensive',
            'generated_at': time.time(),
            'generator_version': '1.0',
            'sections': {}
        }
        
        # Live system analysis
        live_report = self.generate_live_report()
        if live_report:
            comprehensive_report['sections']['live_system'] = live_report
        
        # Historical analysis
        historical_report = self.generate_historical_report()
        if historical_report:
            comprehensive_report['sections']['historical_analysis'] = historical_report
        
        # Summary and recommendations
        comprehensive_report['summary'] = self._generate_executive_summary(comprehensive_report['sections'])
        
        # Save comprehensive report
        report_file = self.export_dir / f"comprehensive_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"✅ Comprehensive report saved: {report_file}")
        
        # Generate human-readable summary
        self._generate_readable_summary(comprehensive_report)
        
        return comprehensive_report
    
    def _generate_executive_summary(self, sections: Dict) -> Dict[str, Any]:
        """Generate executive summary from all report sections."""
        summary = {
            'system_status': 'unknown',
            'learning_effectiveness': 'unknown',
            'key_metrics': {},
            'priority_recommendations': []
        }
        
        # Determine system status
        if 'live_system' in sections:
            live = sections['live_system']
            if live.get('system_health', {}).get('simulation_running', False):
                summary['system_status'] = 'active'
                summary['key_metrics']['active_agents'] = live['system_health']['active_agents']
                summary['key_metrics']['physics_fps'] = live['system_health']['physics_fps']
            else:
                summary['system_status'] = 'inactive'
        else:
            summary['system_status'] = 'offline'
        
        # Analyze learning effectiveness
        if 'live_system' in sections:
            learning_perf = sections['live_system'].get('learning_performance', {})
            active_ratio = learning_perf.get('learning_active', 0) / max(1, learning_perf.get('total_agents', 1))
            if active_ratio > 0.7:
                summary['learning_effectiveness'] = 'high'
            elif active_ratio > 0.3:
                summary['learning_effectiveness'] = 'moderate'
            else:
                summary['learning_effectiveness'] = 'low'
        
        # Collect priority recommendations
        if 'live_system' in sections:
            summary['priority_recommendations'].extend(sections['live_system'].get('recommendations', []))
        
        return summary
    
    def _generate_readable_summary(self, comprehensive_report: Dict):
        """Generate human-readable summary file."""
        summary_content = []
        summary_content.append("# Walker Robot Training - Learning Performance Report")
        summary_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_content.append("")
        
        # Executive summary
        exec_summary = comprehensive_report.get('summary', {})
        summary_content.append("## Executive Summary")
        summary_content.append(f"- **System Status:** {exec_summary.get('system_status', 'unknown').title()}")
        summary_content.append(f"- **Learning Effectiveness:** {exec_summary.get('learning_effectiveness', 'unknown').title()}")
        
        key_metrics = exec_summary.get('key_metrics', {})
        if key_metrics:
            summary_content.append("- **Key Metrics:**")
            for metric, value in key_metrics.items():
                summary_content.append(f"  - {metric.replace('_', ' ').title()}: {value}")
        summary_content.append("")
        
        # Live system status
        if 'live_system' in comprehensive_report['sections']:
            live = comprehensive_report['sections']['live_system']
            summary_content.append("## Live System Analysis")
            
            learning_perf = live.get('learning_performance', {})
            summary_content.append(f"- **Active Learning Agents:** {learning_perf.get('learning_active', 0)}/{learning_perf.get('total_agents', 0)}")
            
            top_performers = learning_perf.get('top_performers', [])
            if top_performers:
                summary_content.append(f"- **Top Performer:** Agent {top_performers[0].get('agent_id')} (Reward: {top_performers[0].get('total_reward', 0):.1f})")
            
            approaches = learning_perf.get('learning_approaches', {})
            if approaches:
                summary_content.append("- **Learning Approaches:**")
                for approach, count in approaches.items():
                    summary_content.append(f"  - {approach}: {count} agents")
            summary_content.append("")
        
        # Historical analysis
        if 'historical_analysis' in comprehensive_report['sections']:
            historical = comprehensive_report['sections']['historical_analysis']
            summary_content.append("## Historical Data Analysis")
            
            insights = historical.get('insights', [])
            for insight in insights:
                summary_content.append(f"- {insight}")
            summary_content.append("")
        
        # Recommendations
        recommendations = exec_summary.get('priority_recommendations', [])
        if recommendations:
            summary_content.append("## Recommendations")
            for rec in recommendations:
                summary_content.append(f"- {rec}")
            summary_content.append("")
        
        # Save readable summary
        summary_file = self.export_dir / f"report_summary_{self.timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_content))
        
        print(f"📄 Human-readable summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Walker Robot Learning Performance Reports")
    parser.add_argument('--live', action='store_true', help='Generate live system report')
    parser.add_argument('--historical', action='store_true', help='Generate historical data report')
    parser.add_argument('--comprehensive', action='store_true', help='Generate comprehensive report (default)')
    parser.add_argument('--export-dir', default='evaluation_exports', help='Export directory for reports')
    
    args = parser.parse_args()
    
    # Default to comprehensive if no specific type selected
    if not any([args.live, args.historical]):
        args.comprehensive = True
    
    print("🤖 Walker Robot Training - Evaluation Report Generator")
    print("=" * 60)
    
    generator = EvaluationReportGenerator(args.export_dir)
    
    try:
        if args.live:
            result = generator.generate_live_report()
            if result:
                print("✅ Live report generated successfully")
            else:
                print("❌ Failed to generate live report")
        
        if args.historical:
            result = generator.generate_historical_report()
            if result:
                print("✅ Historical report generated successfully")
            else:
                print("❌ Failed to generate historical report")
        
        if args.comprehensive:
            result = generator.generate_comprehensive_report()
            if result:
                print("✅ Comprehensive report generated successfully")
                
                # Print quick summary to console
                summary = result.get('summary', {})
                print("\n📊 Quick Summary:")
                print(f"   System Status: {summary.get('system_status', 'unknown').title()}")
                print(f"   Learning Effectiveness: {summary.get('learning_effectiveness', 'unknown').title()}")
                
                recommendations = summary.get('priority_recommendations', [])
                if recommendations:
                    print(f"   Top Recommendation: {recommendations[0]}")
            else:
                print("❌ Failed to generate comprehensive report")
        
    except KeyboardInterrupt:
        print("\n🛑 Report generation interrupted")
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 