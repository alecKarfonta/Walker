import pytest
import json
import sys
import os

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from walker_python.train_robots_web_visual import app, TrainingEnvironment

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Swap out the real training environment with a mock or a lightweight version
    # to avoid starting the full simulation during tests.
    app.training_env = TrainingEnvironment(num_agents=1) # Use a small number of agents for testing
    with app.test_client() as client:
        yield client

def test_status_endpoint(client):
    """
    Tests if the /status endpoint returns a successful response and valid JSON
    with the expected high-level keys.
    """
    # Start the training loop in the background for the test
    app.training_env.start()
    
    try:
        response = client.get('/status')
        
        # 1. Check for successful response
        assert response.status_code == 200
        
        # 2. Check if the response is valid JSON
        data = json.loads(response.data)
        assert isinstance(data, dict)
        
        # 3. Check for the presence of essential top-level keys
        expected_keys = ['robots', 'leaderboard', 'statistics', 'camera', 'shapes']
        for key in expected_keys:
            assert key in data, f"Key '{key}' was not found in the /status response"
            
        print("\n✅ /status endpoint returned a valid structure.")
        
    finally:
        # Ensure the training loop is stopped after the test
        app.training_env.stop() 