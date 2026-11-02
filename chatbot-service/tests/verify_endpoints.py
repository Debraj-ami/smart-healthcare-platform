import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("1. Testing GET /health...")
resp_health = client.get("/health")
print("Status:", resp_health.status_code)
print("Response:", resp_health.json())
assert resp_health.status_code == 200

print("\n2. Testing POST /api/chat with sample symptom message...")
payload = {
    "user_id": "demo-user",
    "session_id": "demo-session",
    "message": "I've had pain below my right ribs for two days."
}
resp_chat = client.post("/api/chat", json=payload)
print("Status:", resp_chat.status_code)
print("Response JSON:\n", json.dumps(resp_chat.json(), indent=2))
assert resp_chat.status_code == 200

print("\n3. Testing POST /api/chat with urgent red-flag message...")
urgent_payload = {
    "user_id": "demo-user",
    "session_id": "urgent-session",
    "message": "I have severe crushing chest pain radiating to my left arm and severe shortness of breath."
}
resp_urgent = client.post("/api/chat", json=urgent_payload)
print("Status:", resp_urgent.status_code)
print("Response JSON:\n", json.dumps(resp_urgent.json(), indent=2))
assert resp_urgent.status_code == 200
assert resp_urgent.json()["safety_status"] == "URGENT_MEDICAL_EVALUATION"
assert resp_urgent.json()["next_action"] == "EMERGENCY_ROUTING"

print("\nAll endpoint verifications passed successfully!")
