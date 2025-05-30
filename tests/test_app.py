# tests/test_app.py

import pytest
import json
from unittest.mock import Mock, patch
import numpy as np
from app import app, normalizeWeights

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_db():
    with patch('app.get_db') as mock:
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value = cursor
        mock.return_value = conn
        yield cursor

@pytest.fixture
def mock_redis():
    with patch('app.r') as mock:
        mock.get.return_value = None
        yield mock

class TestScoreCalculation:
    def test_score_endpoint_basic(self, client, mock_db, mock_redis):
        """Test basic score calculation"""
        # Mock database response
        mock_db.fetchall.return_value = [
            {
                'id': 1,
                'geometry': {'type': 'Polygon', 'coordinates': [[[]]]},
                'score': 0.75,
                'rank': 1,
                'qtrmi_s': 0.8,
                'hwui_s': 0.7,
                'hagri_s': 0.6,
                'hvhsz_s': 0.9,
                'uphill_s': 0.5,
                'neigh1d_s': 0.4,
                'yearbuilt': 1990,
                'qtrmi_cnt': 5,
                'hlfmi_agri': 0.3,
                'hlfmi_wui': 0.4,
                'hlfmi_vhsz': 0.5,
                'num_neighb': 3
            }
        ]
        
        # Test request
        response = client.post('/api/score', 
            json={
                'weights': {
                    'qtrmi_s': 0.2,
                    'hwui_s': 0.2,
                    'hagri_s': 0.15,
                    'hvhsz_s': 0.15,
                    'uphill_s': 0.15,
                    'neigh1d_s': 0.15
                },
                'top_n': 500
            })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['type'] == 'FeatureCollection'
        assert len(data['features']) == 1
        assert data['features'][0]['properties']['score'] == 0.75
        assert data['features'][0]['properties']['top500'] == True
    
    def test_score_with_filters(self, client, mock_db, mock_redis):
        """Test score calculation with filters"""
        mock_db.fetchall.return_value = []
        
        response = client.post('/api/score',
            json={
                'weights': {'qtrmi_s': 1.0},
                'built_after_2011': True,
                'neigh1d_max': 50
            })
        
        # Verify SQL query includes filters
        call_args = mock_db.execute.call_args[0][0]
        assert "yearbuilt > 2011 OR yearbuilt IS NULL" in call_args
        assert "neigh1d_s <= " in call_args
    
    def test_score_normalization(self, client, mock_db, mock_redis):
        """Test that weights are normalized"""
        mock_db.fetchall.return_value = []
        
        response = client.post('/api/score',
            json={
                'weights': {
                    'qtrmi_s': 50,
                    'hwui_s': 30,
                    'hagri_s': 20,
                    'hvhsz_s': 0,
                    'uphill_s': 0,
                    'neigh1d_s': 0
                }
            })
        
        # Check that weights sum to 1 in SQL query
        call_args = mock_db.execute.call_args[0][1]
        weight_sum = sum(call_args[:6])  # First 6 params are weights
        assert abs(weight_sum - 1.0) < 0.001

class TestWeightInference:
    def test_infer_weights_basic(self, client, mock_db, mock_redis):
        """Test basic weight inference"""
        # Mock selected parcels with high qtrmi_s
        mock_db.fetchall.side_effect = [
            # Selected parcels
            [
                {'id': 1, 'qtrmi_s': 0.9, 'hwui_s': 0.3, 'hagri_s': 0.2, 
                 'hvhsz_s': 0.4, 'uphill_s': 0.1, 'neigh1d_s': 0.2},
                {'id': 2, 'qtrmi_s': 0.8, 'hwui_s': 0.2, 'hagri_s': 0.3,
                 'hvhsz_s': 0.5, 'uphill_s': 0.2, 'neigh1d_s': 0.1}
            ],
            # Non-selected parcels
            [
                {'id': 3, 'qtrmi_s': 0.2, 'hwui_s': 0.8, 'hagri_s': 0.7,
                 'hvhsz_s': 0.1, 'uphill_s': 0.9, 'neigh1d_s': 0.8},
                {'id': 4, 'qtrmi_s': 0.1, 'hwui_s': 0.9, 'hagri_s': 0.8,
                 'hvhsz_s': 0.2, 'uphill_s': 0.8, 'neigh1d_s': 0.9}
            ]
        ]
        
        response = client.post('/api/infer-weights',
            json={
                'selection': {
                    'type': 'Polygon',
                    'coordinates': [[[-119.7, 34.4], [-119.6, 34.4], 
                                   [-119.6, 34.5], [-119.7, 34.5], [-119.7, 34.4]]]
                }
            })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'weights' in data
        # Should have higher weight for qtrmi_s
        assert data['weights']['qtrmi_s'] > data['weights']['hwui_s']
    
    def test_infer_weights_no_selection(self, client):
        """Test inference with no selection"""
        response = client.post('/api/infer-weights', json={})
        assert response.status_code == 400
    
    def test_infer_weights_infeasible(self, client, mock_db, mock_redis):
        """Test handling of infeasible optimization"""
        # Mock parcels with no clear pattern
        mock_db.fetchall.side_effect = [
            [{'id': i, **{var: np.random.random() for var in 
              ['qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'uphill_s', 'neigh1d_s']}}
             for i in range(5)],
            [{'id': i+5, **{var: np.random.random() for var in 
              ['qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'uphill_s', 'neigh1d_s']}}
             for i in range(5)]
        ]
        
        response = client.post('/api/infer-weights',
            json={'selection': {'type': 'Polygon', 'coordinates': [[[]]]}})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        # Should return default equal weights
        weights = data['weights']
        assert all(abs(w - 16.7) < 1 for w in weights.values())

class TestFiltering:
    def test_get_parcels_with_filters(self, client, mock_db, mock_redis):
        """Test parcel filtering endpoint"""
        mock_db.fetchall.return_value = []
        
        response = client.get('/api/parcels?built_after_2011=true&neigh1d_max=50')
        
        assert response.status_code == 200
        # Check SQL query
        call_args = mock_db.execute.call_args[0][0]
        assert "yearbuilt > 2011 OR yearbuilt IS NULL" in call_args
        assert "neigh1d_s <= " in call_args

class TestPerformance:
    def test_caching(self, client, mock_db, mock_redis):
        """Test that results are cached"""
        mock_redis.get.return_value = json.dumps({
            'type': 'FeatureCollection',
            'features': []
        })
        
        response = client.post('/api/score', json={'weights': {}})
        
        # Should not hit database if cached
        mock_db.execute.assert_not_called()
    
    def test_response_time(self, client, mock_db, mock_redis):
        """Test response time meets requirements"""
        import time
        
        # Mock large dataset
        mock_db.fetchall.return_value = [
            {
                'id': i,
                'geometry': {'type': 'Polygon', 'coordinates': [[[]]]},
                'score': 0.5,
                'rank': i,
                **{var: 0.5 for var in ['qtrmi_s', 'hwui_s', 'hagri_s', 
                                        'hvhsz_s', 'uphill_s', 'neigh1d_s']},
                'yearbuilt': 2000,
                'qtrmi_cnt': 1,
                'hlfmi_agri': 0.1,
                'hlfmi_wui': 0.1,
                'hlfmi_vhsz': 0.1,
                'num_neighb': 1
            }
            for i in range(70000)
        ]
        
        start = time.time()
        response = client.post('/api/score', json={'weights': {}})
        duration = time.time() - start
        
        assert response.status_code == 200
        # Should complete in under 2 seconds
        assert duration < 2.0

class TestIntegration:
    def test_end_to_end_workflow(self, client, mock_db, mock_redis):
        """Test complete workflow from scoring to weight inference"""
        # Step 1: Initial score calculation
        mock_db.fetchall.return_value = [
            {
                'id': i,
                'geometry': {'type': 'Polygon', 'coordinates': [[[]]]},
                'score': 0.7 if i < 10 else 0.3,
                'rank': i + 1,
                'qtrmi_s': 0.8 if i < 10 else 0.2,
                'hwui_s': 0.5,
                'hagri_s': 0.5,
                'hvhsz_s': 0.5,
                'uphill_s': 0.5,
                'neigh1d_s': 0.5,
                'yearbuilt': 1990,
                'qtrmi_cnt': 1,
                'hlfmi_agri': 0.1,
                'hlfmi_wui': 0.1,
                'hlfmi_vhsz': 0.1,
                'num_neighb': 1
            }
            for i in range(20)
        ]
        
        response = client.post('/api/score',
            json={'weights': {var: 1/6 for var in 
                  ['qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'uphill_s', 'neigh1d_s']}})
        assert response.status_code == 200
        
        # Step 2: Draw selection and infer weights
        mock_db.fetchall.side_effect = [
            # Selected (high qtrmi_s)
            [{'id': i, 'qtrmi_s': 0.8, 'hwui_s': 0.5, 'hagri_s': 0.5,
              'hvhsz_s': 0.5, 'uphill_s': 0.5, 'neigh1d_s': 0.5} for i in range(5)],
            # Non-selected (low qtrmi_s)
            [{'id': i, 'qtrmi_s': 0.2, 'hwui_s': 0.5, 'hagri_s': 0.5,
              'hvhsz_s': 0.5, 'uphill_s': 0.5, 'neigh1d_s': 0.5} for i in range(5)]
        ]
        
        response = client.post('/api/infer-weights',
            json={'selection': {'type': 'Polygon', 'coordinates': [[[]]]}})
        assert response.status_code == 200
        
        # Should infer higher weight for qtrmi_s
        weights = json.loads(response.data)['weights']
        assert weights['qtrmi_s'] > weights['hwui_s']

# Run tests with: pytest tests/test_app.py -v