from fastapi.testclient import TestClient

from allelion.api.main import app

client = TestClient(app)


def test_classify_by_hgvs():
    r = client.post("/api/classify", json={"gene": "MYH7", "hgvs_c": "c.1208G>A"})
    assert r.status_code == 200
    body = r.json()
    assert body["classification"] in {"Pathogenic", "Likely pathogenic", "Uncertain significance", "Likely benign", "Benign"}
    assert "learned_verdict" in body and body["reference"]["note"].startswith("Shown for comparison")


def test_bad_hgvs_is_a_clear_error():
    r = client.post("/api/classify", json={"gene": "TP53", "hgvs_c": "nonsense"})
    assert r.status_code == 422


def test_copilot_cites_every_sentence():
    r = client.post("/api/copilot", json={"question": "At what strength should PM2 be applied?"}).json()
    assert r["sentences"] and all(s["cite"] for s in r["sentences"])
