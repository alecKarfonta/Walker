import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import json

def create_app(training_environment):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

    app.title = "Walker AI Training"

    # --- Layout Definition ---
    app.layout = dbc.Container([
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
        
        # Header
        dbc.Row(dbc.Col(html.H1("Walker AI Training Dashboard"), width=12), className="mb-4"),

        # Main Content
        dbc.Row([
            # Left Sidebar (Leaderboard & Summary)
            dbc.Col([
                html.H4("🏆 Leaderboard"),
                html.Div(id='leaderboard-content'),
                html.Hr(),
                html.H4("📊 Population Summary"),
                html.Div(id='population-summary-content'),
            ], md=3, style={'height': '80vh', 'overflowY': 'auto', 'padding': '10px'}),

            # Center Panel (Visualization)
            dbc.Col([
                html.H4("Simulation View"),
                html.Img(id='simulation-canvas', style={'width': '100%'})
            ], md=6),

            # Right Sidebar (Controls)
            dbc.Col([
                html.H4("⚙️ Controls"),
                dbc.Accordion([
                    dbc.AccordionItem(
                        "Learning settings content will go here.",
                        title="Learning Settings"
                    ),
                    dbc.AccordionItem(
                        "Physical settings content will go here.",
                        title="Physical Settings"
                    ),
                    dbc.AccordionItem(
                        "Evolution settings content will go here.",
                        title="Evolution Settings"
                    ),
                ], start_collapsed=True),
            ], md=3),
        ]),
    ], fluid=True)


    # --- Callbacks ---
    @app.callback(
        [
            Output('leaderboard-content', 'children'),
            Output('population-summary-content', 'children'),
            Output('simulation-canvas', 'src')
        ],
        Input('interval-component', 'n_intervals')
    )
    def update_data(n):
        status = training_environment.get_status()
        
        # --- Leaderboard ---
        agents = status.get('agents', [])
        sorted_agents = sorted(agents, key=lambda a: a.get('performance', 0), reverse=True)
        
        leaderboard_items = []
        for rank, agent in enumerate(sorted_agents[:10]): # Top 10
            leaderboard_items.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"#{rank + 1}: Robot {agent['id']}", className="card-title"),
                        html.P(f"Performance: {agent.get('performance', 0):.2f}", className="card-text"),
                    ])
                ], className="mb-2")
            )

        # --- Population Summary ---
        pop_stats = status.get('population_stats', {})
        summary = [
            html.P(f"Generation: {pop_stats.get('generation', 1)}"),
            html.P(f"Average Fitness: {pop_stats.get('average_fitness', 0):.2f}"),
        ]

        # --- Simulation Image ---
        image_src = training_environment.render_to_base64_image()

        return leaderboard_items, summary, image_src

    return app

if __name__ == '__main__':
    # This part is for testing the UI component independently.
    # It won't be used when running from train_robots_web_visual.py
    class MockTrainingEnvironment:
        def get_status(self):
            return {
                'agents': [
                    {'id': 48, 'performance': 3.04},
                    {'id': 35, 'performance': 2.77},
                ],
                'population_stats': {
                    'generation': 1,
                    'average_fitness': -0.00,
                }
            }
            
        def render_to_base64_image(self):
            # Return a placeholder image for mock environment
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            
    mock_env = MockTrainingEnvironment()
    app = create_app(mock_env)
    app.run(debug=True) 