#!/usr/bin/env python3
"""
Integration tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from api_gateway.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_chat_endpoint_normal(self):
        """Test chat endpoint with normal message"""
        response = client.post("/chat", json={
            "user_input": "Xin chào",
            "session_id": "test_session_123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "bot_response" in data
        assert "risk_level" in data
        assert "confidence" in data
    
    def test_chat_endpoint_emergency(self):
        """Test chat endpoint with emergency message"""
        response = client.post("/chat", json={
            "user_input": "Tôi muốn tự tử",
            "session_id": "test_session_456"
        })
        assert response.status_code == 200
        data = response.json()
        assert "bot_response" in data
        assert "risk_level" in data
        # Emergency messages should have high risk level
        assert data["risk_level"] in ["high", "emergency"]
    
    def test_chat_endpoint_missing_input(self):
        """Test chat endpoint with missing input"""
        response = client.post("/chat", json={
            "session_id": "test_session_789"
        })
        assert response.status_code == 422  # Validation error
    
    def test_context_endpoint(self):
        """Test context management endpoints"""
        # Test getting context
        response = client.get("/context/test_session_123")
        assert response.status_code == 200
        
        # Test clearing context
        response = client.delete("/context/test_session_123")
        assert response.status_code == 200
    
    def test_emergency_endpoint(self):
        """Test emergency endpoint"""
        response = client.post("/emergency", json={
            "user_input": "Tôi muốn tự tử",
            "session_id": "test_session_emergency"
        })
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post("/chat", data="invalid json")
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        response = client.post("/chat", json={})
        assert response.status_code == 422
    
    def test_large_input(self):
        """Test handling of very large input"""
        large_input = "A" * 10000  # 10KB input
        response = client.post("/chat", json={
            "user_input": large_input,
            "session_id": "test_session_large"
        })
        # Should handle large input gracefully
        assert response.status_code in [200, 413]  # 413 if size limit enforced


if __name__ == "__main__":
    pytest.main([__file__]) 