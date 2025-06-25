import pytest
import json
import sys
import os

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from train_robots_web_visual import app, TrainingEnvironment

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Swap out the real training environment with a mock or a lightweight version
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Walker Training Visualizer' in response.data

def test_population_endpoint(client):
    response = client.get('/api/population')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'population' in data
    assert isinstance(data['population'], list)

def test_metrics_endpoint(client):
    response = client.get('/api/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'metrics' in data 