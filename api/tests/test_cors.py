"""Regression test for the CORS middleware — without it, every browser
request from the deployed frontend (a different origin than the API) is
blocked before it reaches a route. See api/main.py.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_cross_origin_get_allowed():
    client = TestClient(app)
    resp = client.get("/healthz", headers={"Origin": "https://nova-bodhix.vercel.app"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_preflight_for_post_ask_allowed():
    client = TestClient(app)
    resp = client.options(
        "/ask",
        headers={
            "Origin": "https://nova-bodhix.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
