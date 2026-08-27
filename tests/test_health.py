def test_health_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["asr_model"] == "Qwen/Qwen3-ASR-0.6B"
