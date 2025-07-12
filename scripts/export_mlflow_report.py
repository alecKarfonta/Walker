#!/usr/bin/env python3
"""
Export MLflow data as an HTML report that can be viewed through the web interface.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to Python path  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Set up MLflow client pointing to the correct database
    db_path = "/app/experiments/walker_experiments.db" if os.path.exists("/app") else "experiments/walker_experiments.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = MlflowClient()
    
    print("🔬 Generating Walker Robot Training MLflow Report...")
    
    # Create output directory
    output_dir = Path("/app/static") if os.path.exists("/app") else Path("static")
    output_dir.mkdir(exist_ok=True)
    
    # Generate HTML report
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Walker Robot Training - MLflow Report</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 20px; 
            background: #f5f5f5; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        .header { 
            text-align: center; 
            color: #2c3e50; 
            margin-bottom: 30px; 
        }
        .experiment { 
            margin-bottom: 40px; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            overflow: hidden; 
        }
        .experiment-header { 
            background: #3498db; 
            color: white; 
            padding: 15px; 
            font-size: 1.2em; 
            font-weight: bold; 
        }
        .experiment-content { 
            padding: 20px; 
        }
        .runs-table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 10px 0; 
        }
        .runs-table th, .runs-table td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }
        .runs-table th { 
            background: #34495e; 
            color: white; 
        }
        .runs-table tr:nth-child(even) { 
            background: #f2f2f2; 
        }
        .metric-box { 
            display: inline-block; 
            background: #ecf0f1; 
            border: 1px solid #bdc3c7; 
            border-radius: 4px; 
            padding: 10px; 
            margin: 5px; 
            min-width: 150px; 
        }
        .metric-label { 
            font-weight: bold; 
            color: #7f8c8d; 
            font-size: 0.9em; 
        }
        .metric-value { 
            font-size: 1.2em; 
            color: #2c3e50; 
        }
        .status-running { color: #27ae60; font-weight: bold; }
        .status-finished { color: #2980b9; font-weight: bold; }
        .status-failed { color: #e74c3c; font-weight: bold; }
        .no-data { 
            text-align: center; 
            color: #7f8c8d; 
            font-style: italic; 
            padding: 20px; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Walker Robot Training - MLflow Report</h1>
            <p>Generated on {timestamp}</p>
        </div>
"""
    
    # Get all experiments
    experiments = client.search_experiments()
    
    total_runs = 0
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        total_runs += len(runs)
        
        html_content += f"""
        <div class="experiment">
            <div class="experiment-header">
                📊 Experiment: {exp.name}
            </div>
            <div class="experiment-content">
                <p><strong>Experiment ID:</strong> {exp.experiment_id}</p>
                <p><strong>Total Runs:</strong> {len(runs)}</p>
        """
        
        if runs:
            # Recent runs table
            recent_runs = runs[:10]  # Last 10 runs
            html_content += """
                <h3>📈 Recent Runs (Last 10)</h3>
                <table class="runs-table">
                    <thead>
                        <tr>
                            <th>Run ID</th>
                            <th>Status</th>
                            <th>Start Time</th>
                            <th>Generation</th>
                            <th>Avg Fitness</th>
                            <th>Best Fitness</th>
                            <th>Population</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for run in recent_runs:
                run_id = run.info.run_id[:8]
                status = run.info.status
                status_class = f"status-{status.lower()}"
                start_time = datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M')
                
                metrics = run.data.metrics
                generation = metrics.get('generation', 'N/A')
                avg_fitness = f"{metrics.get('avg_fitness', 0):.3f}" if metrics.get('avg_fitness') else 'N/A'
                best_fitness = f"{metrics.get('best_fitness', 0):.3f}" if metrics.get('best_fitness') else 'N/A'
                population = int(metrics.get('population_size', 0)) if metrics.get('population_size') else 'N/A'
                
                html_content += f"""
                        <tr>
                            <td>{run_id}</td>
                            <td class="{status_class}">{status}</td>
                            <td>{start_time}</td>
                            <td>{generation}</td>
                            <td>{avg_fitness}</td>
                            <td>{best_fitness}</td>
                            <td>{population}</td>
                        </tr>
                """
            
            html_content += """
                    </tbody>
                </table>
            """
            
            # Latest run details
            if runs:
                latest_run = runs[0]
                html_content += f"""
                <h3>🎯 Latest Run Details (ID: {latest_run.info.run_id[:8]})</h3>
                <div style="margin: 20px 0;">
                """
                
                # Show key metrics in boxes
                metrics = latest_run.data.metrics
                key_metrics = [
                    ('generation', 'Generation'),
                    ('avg_fitness', 'Avg Fitness'),
                    ('best_fitness', 'Best Fitness'),
                    ('diversity', 'Diversity'),
                    ('population_size', 'Population Size'),
                    ('step_count', 'Step Count')
                ]
                
                for metric_key, metric_label in key_metrics:
                    value = metrics.get(metric_key, 'N/A')
                    if isinstance(value, (int, float)):
                        if metric_key in ['avg_fitness', 'best_fitness', 'diversity']:
                            value_str = f"{value:.3f}"
                        else:
                            value_str = f"{value:.0f}"
                    else:
                        value_str = str(value)
                    
                    html_content += f"""
                    <div class="metric-box">
                        <div class="metric-label">{metric_label}</div>
                        <div class="metric-value">{value_str}</div>
                    </div>
                    """
                
                html_content += "</div>"
                
                # Individual robot performance
                robot_metrics = {}
                for key, value in metrics.items():
                    if key.startswith('robot_top_'):
                        parts = key.split('_')
                        if len(parts) >= 4:
                            robot_rank = parts[2]
                            robot_id = parts[3]
                            metric_name = '_'.join(parts[4:])
                            
                            if robot_rank not in robot_metrics:
                                robot_metrics[robot_rank] = {'id': robot_id}
                            robot_metrics[robot_rank][metric_name] = value
                
                if robot_metrics:
                    html_content += """
                    <h3>🏆 Top Robot Performance</h3>
                    <table class="runs-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Robot ID</th>
                                <th>Fitness</th>
                                <th>Position X</th>
                                <th>Total Reward</th>
                                <th>Steps</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    
                    for rank in sorted(robot_metrics.keys(), key=int):
                        robot = robot_metrics[rank]
                        robot_id = robot.get('id', 'Unknown')[:8]
                        fitness = robot.get('fitness', 0)
                        position_x = robot.get('position', {}).get('x', robot.get('position_x', 0))
                        total_reward = robot.get('total_reward', 0)
                        steps = robot.get('steps', 0)
                        
                        html_content += f"""
                            <tr>
                                <td>#{rank}</td>
                                <td>{robot_id}</td>
                                <td>{fitness:.3f}</td>
                                <td>{position_x:.1f}m</td>
                                <td>{total_reward:.3f}</td>
                                <td>{steps:.0f}</td>
                            </tr>
                        """
                    
                    html_content += """
                        </tbody>
                    </table>
                    """
        else:
            html_content += '<div class="no-data">No runs found in this experiment</div>'
        
        html_content += """
            </div>
        </div>
        """
    
    # Summary
    html_content += f"""
        <div class="experiment">
            <div class="experiment-header">
                📋 Summary
            </div>
            <div class="experiment-content">
                <div class="metric-box">
                    <div class="metric-label">Total Experiments</div>
                    <div class="metric-value">{len(experiments)}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Total Runs</div>
                    <div class="metric-value">{total_runs}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Database Size</div>
                    <div class="metric-value">{os.path.getsize(db_path) / 1024 / 1024:.1f} MB</div>
                </div>
                
                <h3>💡 Access Full MLflow UI</h3>
                <p>To access the full MLflow interface with interactive charts and comparisons:</p>
                <ol>
                    <li>Visit: <a href="http://localhost:5002" target="_blank">MLflow UI (PostgreSQL backend)</a></li>
                    <li>Access: <a href="http://localhost:7777" target="_blank">http://localhost:7777</a> (using the existing port mapping)</li>
                </ol>
            </div>
        </div>
    </body>
</html>
    """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Write HTML report
    report_path = output_dir / "mlflow_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    # Also create a JSON export for API access
    export_data = {
        "generated_at": datetime.now().isoformat(),
        "database_path": db_path,
        "database_size_mb": os.path.getsize(db_path) / 1024 / 1024,
        "experiments": []
    }
    
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=50)
        exp_data = {
            "name": exp.name,
            "experiment_id": exp.experiment_id,
            "total_runs": len(client.search_runs(exp.experiment_id)),
            "recent_runs": []
        }
        
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "metrics": run.data.metrics,
                "params": run.data.params
            }
            exp_data["recent_runs"].append(run_data)
        
        export_data["experiments"].append(exp_data)
    
    json_path = output_dir / "mlflow_data.json"
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✅ MLflow report generated!")
    print(f"📄 HTML Report: {report_path}")
    print(f"📊 JSON Data: {json_path}")
    print(f"🌐 View at: http://localhost:7777/static/mlflow_report.html")
    
except Exception as e:
    print(f"❌ Error generating MLflow report: {e}")
    import traceback
    traceback.print_exc() 