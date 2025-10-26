import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_add_docs():
    """Test add documents endpoint"""
    print("Testing add documents...")
    data = {
        "docs": [
            {
                "content": "Test document content",
                "metadata": {"source": "test", "type": "test"}
            }
        ]
    }
    response = requests.post(f"{API_URL}/add_docs", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_search():
    """Test search endpoint"""
    print("Testing search...")
    data = {
        "query": "test search",
        "k": 5
    }
    response = requests.post(f"{API_URL}/search-similar", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_chat():
    """Test chat endpoint"""
    print("Testing chat...")
    data = {
        "message": "Xin chào",
        "top_k": 5
    }
    response = requests.post(f"{API_URL}/chat", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("=== RAG API Test Suite ===\n")
    
    try:
        test_health()
        test_add_docs()
        test_search()
        test_chat()
        print("All tests completed!")
    except Exception as e:
        print(f"Test failed: {e}")
