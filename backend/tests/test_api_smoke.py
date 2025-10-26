from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_run_train_endpoint():
    # This is a smoke test for the /api/v1/run_train endpoint
    # You may need to mock file upload or use a sample config file
    response = client.post(
        "/api/v1/run_train",
        files={"config_file": ("config.yaml", open("conf/config.yaml", "rb"), "application/x-yaml")}
    )
    assert response.status_code == 200
    assert "success" in response.json()
