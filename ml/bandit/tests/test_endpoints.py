import requests

def test_knapsack_endpoint():
    response = requests.post(
        'http://localhost:8001/ml/discogs/knapsack/',
        json={
            'sellers': [{'name': 'kim_melody', 'shipping_min': 5.0, 'currency': 'USD'}],
            'budget': 100.0
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'knapsacks' in data
    assert data['budget'] == 100.0