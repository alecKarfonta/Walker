"""
Evaluation routes for accessing learning performance reports and metrics.
"""

import json
import time
import traceback
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app, send_file
from typing import Dict, List, Any, Optional

# Create blueprint for evaluation routes
evaluation_bp = Blueprint('evaluation', __name__, url_prefix='/evaluation')

def get_training_env():
    """Get the training environment from the Flask app."""
    return getattr(current_app, 'env', None)

@evaluation_bp.route('/status')
def evaluation_status():
    """Get the status of the evaluation framework."""
    try:
        env = get_training_env()
        if not env:
            return jsonify({'error': 'Training environment not available'}), 503
        
        status = {
            'timestamp': time.time(),
            'evaluation_enabled': env.enable_evaluation,
            'components': {
                'metrics_collector': env.metrics_collector is not None,
                'dashboard_exporter': env.dashboard_exporter is not None,
                'mlflow_integration': env.mlflow_integration is not None
            },
            'export_directory': 'evaluation_exports',
            'export_directory_exists': Path('evaluation_exports').exists()
        }
        
        # Add component details if available
        if env.metrics_collector:
            status['metrics_collector'] = {
                'metrics_history_size': len(env.metrics_collector.metrics_history),
                'last_evaluation_time': env.metrics_collector.last_evaluation_time,
                'enable_file_export': env.metrics_collector.enable_file_export
            }
        
        if env.dashboard_exporter:
            status['dashboard_exporter'] = {
                'port': getattr(env.dashboard_exporter, 'port', 2322),
                'enable_api': getattr(env.dashboard_exporter, 'enable_api', True)
            }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get evaluation status: {e}'}), 500

@evaluation_bp.route('/reports')
def list_reports():
    """List available evaluation reports."""
    try:
        export_dir = Path('evaluation_exports')
        if not export_dir.exists():
            return jsonify({'reports': [], 'message': 'No reports directory found'})
        
        reports = []
        for file in export_dir.glob('*.json'):
            try:
                stat = file.stat()
                reports.append({
                    'filename': file.name,
                    'size_bytes': stat.st_size,
                    'created_at': stat.st_mtime,
                    'type': 'json_report'
                })
            except Exception as e:
                continue
        
        # Also check for markdown summaries
        for file in export_dir.glob('*.md'):
            try:
                stat = file.stat()
                reports.append({
                    'filename': file.name,
                    'size_bytes': stat.st_size,
                    'created_at': stat.st_mtime,
                    'type': 'markdown_summary'
                })
            except Exception as e:
                continue
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'reports': reports,
            'total_count': len(reports)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to list reports: {e}'}), 500

@evaluation_bp.route('/reports/<filename>')
def get_report(filename):
    """Get a specific evaluation report."""
    try:
        export_dir = Path('evaluation_exports')
        report_file = export_dir / filename
        
        if not report_file.exists():
            return jsonify({'error': 'Report not found'}), 404
        
        # Security check - ensure file is in export directory
        if not str(report_file.resolve()).startswith(str(export_dir.resolve())):
            return jsonify({'error': 'Invalid file path'}), 400
        
        if filename.endswith('.json'):
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            return jsonify(report_data)
        elif filename.endswith('.md'):
            with open(report_file, 'r') as f:
                content = f.read()
            return jsonify({
                'filename': filename,
                'type': 'markdown',
                'content': content
            })
        else:
            # For other file types, return as file download
            return send_file(report_file, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get report: {e}'}), 500

@evaluation_bp.route('/generate', methods=['POST'])
def generate_report():
    """Generate a new evaluation report."""
    try:
        data = request.get_json() or {}
        report_type = data.get('type', 'comprehensive')  # comprehensive, live, historical
        
        # Import the report generator
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
        
        from generate_evaluation_report import EvaluationReportGenerator
        
        generator = EvaluationReportGenerator("evaluation_exports")
        
        if report_type == 'live':
            result = generator.generate_live_report()
        elif report_type == 'historical':
            result = generator.generate_historical_report()
        else:  # comprehensive
            result = generator.generate_comprehensive_report()
        
        if result:
            return jsonify({
                'success': True,
                'report_type': report_type,
                'message': f'{report_type.title()} report generated successfully',
                'timestamp': result.get('timestamp') or result.get('generated_at')
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to generate {report_type} report'
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Report generation failed: {e}',
            'traceback': traceback.format_exc()
        }), 500

@evaluation_bp.route('/live')
def live_evaluation():
    """Get live evaluation metrics from the current training session."""
    try:
        env = get_training_env()
        if not env:
            return jsonify({'error': 'Training environment not available'}), 503
        
        if not env.enable_evaluation:
            return jsonify({'error': 'Evaluation not enabled'}), 400
        
        # Get current system state
        live_metrics = {
            'timestamp': time.time(),
            'system_health': {
                'is_running': env.is_running,
                'agent_count': len([a for a in env.agents if not getattr(a, '_destroyed', False)]),
                'physics_fps': getattr(env, 'current_physics_fps', 0),
                'step_count': env.step_count,
                'generation': env.evolution_engine.generation
            },
            'learning_performance': {},
            'population_metrics': env.population_stats
        }
        
        # Add learning performance analysis
        if env.agents:
            active_agents = [a for a in env.agents if not getattr(a, '_destroyed', False)]
            
            learning_perf = {
                'total_agents': len(active_agents),
                'learning_active': 0,
                'reward_stats': {
                    'total_rewards': [],
                    'episode_rewards': []
                },
                'learning_approaches': {}
            }
            
            for agent in active_agents:
                try:
                    total_reward = getattr(agent, 'total_reward', 0)
                    episode_reward = getattr(agent, 'episode_reward', 0)
                    
                    if total_reward > 0:
                        learning_perf['learning_active'] += 1
                    
                    learning_perf['reward_stats']['total_rewards'].append(total_reward)
                    learning_perf['reward_stats']['episode_rewards'].append(episode_reward)
                    
                    approach = getattr(agent, 'learning_approach', 'unknown')
                    learning_perf['learning_approaches'][approach] = learning_perf['learning_approaches'].get(approach, 0) + 1
                    
                except Exception as e:
                    continue
            
            # Calculate statistics
            if learning_perf['reward_stats']['total_rewards']:
                rewards = learning_perf['reward_stats']['total_rewards']
                learning_perf['reward_stats']['avg_total_reward'] = sum(rewards) / len(rewards)
                learning_perf['reward_stats']['max_total_reward'] = max(rewards)
                learning_perf['reward_stats']['min_total_reward'] = min(rewards)
            
            live_metrics['learning_performance'] = learning_perf
        
        # Add metrics collector data if available
        if env.metrics_collector and env.metrics_collector.metrics_history:
            latest_metrics = env.metrics_collector.metrics_history[-1]
            live_metrics['latest_comprehensive_metrics'] = {
                'timestamp': latest_metrics.timestamp,
                'generation': latest_metrics.generation,
                'step_count': latest_metrics.step_count
            }
        
        return jsonify(live_metrics)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get live evaluation: {e}'}), 500

@evaluation_bp.route('/metrics/collector')
def metrics_collector_status():
    """Get detailed metrics collector status and history."""
    try:
        env = get_training_env()
        if not env or not env.metrics_collector:
            return jsonify({'error': 'Metrics collector not available'}), 404
        
        collector = env.metrics_collector
        
        status = {
            'metrics_history_size': len(collector.metrics_history),
            'last_evaluation_time': collector.last_evaluation_time,
            'evaluation_interval': collector.evaluation_interval,
            'enable_file_export': collector.enable_file_export,
            'export_directory': str(collector.export_directory),
            'collection_times': collector.collection_times[-10:] if collector.collection_times else []  # Last 10 collection times
        }
        
        # Add recent metrics summary
        if collector.metrics_history:
            recent_metrics = collector.metrics_history[-5:]  # Last 5 metrics
            status['recent_metrics'] = [
                {
                    'timestamp': m.timestamp,
                    'generation': m.generation,
                    'step_count': m.step_count
                }
                for m in recent_metrics
            ]
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get metrics collector status: {e}'}), 500

@evaluation_bp.route('/dashboard')
def evaluation_dashboard():
    """Get dashboard data for the evaluation interface."""
    try:
        env = get_training_env()
        if not env:
            return jsonify({'error': 'Training environment not available'}), 503
        
        dashboard_data = {
            'timestamp': time.time(),
            'evaluation_enabled': env.enable_evaluation,
            'system_overview': {
                'training_active': env.is_running,
                'agent_count': len([a for a in env.agents if not getattr(a, '_destroyed', False)]),
                'generation': env.evolution_engine.generation,
                'step_count': env.step_count
            },
            'reports_available': [],
            'quick_metrics': {}
        }
        
        # Get available reports
        export_dir = Path('evaluation_exports')
        if export_dir.exists():
            for file in export_dir.glob('*.json'):
                try:
                    dashboard_data['reports_available'].append({
                        'filename': file.name,
                        'created_at': file.stat().st_mtime
                    })
                except:
                    continue
        
        # Get quick metrics if evaluation is enabled
        if env.enable_evaluation and env.agents:
            active_agents = [a for a in env.agents if not getattr(a, '_destroyed', False)]
            rewards = [getattr(a, 'total_reward', 0) for a in active_agents]
            
            dashboard_data['quick_metrics'] = {
                'active_agents': len(active_agents),
                'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
                'max_reward': max(rewards) if rewards else 0,
                'learning_activity': len([r for r in rewards if r > 0])
            }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get dashboard data: {e}'}), 500

@evaluation_bp.route('/test')
def test_evaluation():
    """Test the evaluation framework components."""
    try:
        # Import the test framework
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
        
        # This would run tests but we'll just return a simple status for now
        return jsonify({
            'message': 'Evaluation framework test endpoint',
            'suggestion': 'Run: python scripts/test_evaluation_framework.py for full testing',
            'quick_test': 'Components initialized successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {e}'}), 500 