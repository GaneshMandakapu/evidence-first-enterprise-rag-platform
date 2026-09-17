"""Smoke tests for the browser demo surface and its API wiring."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_demo_homepage_is_served():
    response = client.get("/")
    stylesheet = client.get("/static/style.css")

    assert response.status_code == 200
    assert "Evidence-First Enterprise RAG Platform" in response.text
    assert "/static/app.js" in response.text
    assert stylesheet.status_code == 200
    assert "--ink:" in stylesheet.text


def test_demo_query_requires_api_key():
    response = client.post("/query", json={"question": "What is the onboarding process?"})

    assert response.status_code == 401