import io
import time


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]["version"] == "test"


def test_ui(client):
    assert client.get("/").status_code == 200


def test_predict_csv(client, csv_bytes):
    r = client.post("/predict", files={"file": ("clients.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    lines = r.text.strip().split("\n")
    assert lines[0] == "id,deposit_probability,prediction"
    assert len(lines) == 201  # header + 200 строк
    cols = lines[1].split(",")
    assert 0.0 <= float(cols[1]) <= 1.0
    assert cols[2] in ("0", "1")


def test_predict_rejects_non_csv(client):
    r = client.post("/predict", files={"file": ("data.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_schema_error(client, csv_bytes):
    # портим значение month
    bad = csv_bytes.replace(b"jan", b"januar")
    r = client.post("/predict", files={"file": ("bad.csv", bad, "text/csv")})
    # стрим начался с 200, ошибка в теле ответа
    assert "error" in r.text or r.status_code == 422


def test_predict_batch_flow(client, csv_bytes):
    r = client.post("/predict/batch", files={"file": ("clients.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    for _ in range(50):
        status = client.get(f"/jobs/{job_id}").json()["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.1)
    assert status == "done"

    r = client.get(f"/jobs/{job_id}/result")
    assert r.status_code == 200
    assert r.text.split("\n")[0] == "id,deposit_probability,prediction"


def test_job_not_found(client):
    assert client.get("/jobs/nonexistent").status_code == 404


def test_reload_requires_key(client):
    assert client.post("/admin/reload-model").status_code == 403


def test_drift_report(client):
    r = client.get("/drift/report")
    assert r.status_code == 200
    assert r.json()["enabled"] is False  # в тестах baseline не загружен


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_request" in r.text or "python_info" in r.text


def test_upload_size_limit(client, monkeypatch):
    from src.api import main as api_main
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/predict", files={"file": ("big.csv", b"x" * 100, "text/csv")})
    assert r.status_code == 413
