"""Tests for serve.py. No network, no data.js and no dwave.conf needed."""

import base64
import json
import struct
import threading
import urllib.error
import urllib.request

import pytest

import serve

# a fragment of a puzzle: cells (0, 0) and (0, 1), each 1 or 2
# variables x0 = (0,0)=1, x1 = (0,0)=2, x2 = (0,1)=1, x3 = (0,1)=2
FAKE_DEMO = {
    "chip": {"solver": "fake_solver", "region": "na-west-1"},
    "puzzles": {"4": [{
        "variables": [[0, 0, 1], [0, 0, 2], [0, 1, 1], [0, 1, 2]],
        "groups": [[0, 1], [2, 3], [0, 2], [1, 3]],
        "qpu": {"chains": [[10], [11, 12], [13], [14, 15, 16]],
                "h": [[10, 0.5], [11, 0.25], [12, 0.25], [13, 0.5], [14, 0.2], [15, 0.2], [16, 0.1]],
                "J": [[11, 12, -2.0], [14, 15, -2.0], [15, 16, -2.0], [10, 13, 0.5]],
                "chain_strength": 2.0, "num_reads": 3, "annealing_time": 20.0},
    }]},
}

# spins by qubit for two distinct reads
READ_BAD = {10: -1, 11: 1, 12: -1, 13: 1, 14: -1, 15: -1, 16: 1}  # bits 0, 0 (tie), 1, 0: energy 2
READ_GOOD = {10: 1, 11: -1, 12: -1, 13: -1, 14: 1, 15: 1, 16: -1}  # bits 1, 0, 0, 1 (broken chain): energy 0


def test_majority():
    assert serve.majority({1: 1}, [1]) == 1
    assert serve.majority({1: -1}, [1]) == 0
    assert serve.majority({1: 1, 2: -1}, [1, 2]) == 0  # tie
    assert serve.majority({1: 1, 2: 1, 3: -1}, [1, 2, 3]) == 1  # broken chain
    assert serve.majority({1: -1, 2: 1, 3: -1}, [1, 2, 3]) == 0
    assert serve.majority([0, 1, -1, 1, 1], [1, 3, 4]) == 1  # a row of spins indexed by qubit


def test_energy():
    groups = FAKE_DEMO["puzzles"]["4"][0]["groups"]
    assert serve.energy([1, 0, 0, 1], groups) == 0
    assert serve.energy([0, 1, 1, 0], groups) == 0
    assert serve.energy([1, 1, 0, 0], groups) == 2
    assert serve.energy([0, 0, 0, 0], groups) == 4
    assert serve.energy([1, 1, 1, 1], groups) == 4
    assert serve.energy([1, 1], [[0, 1], [0], [1], []]) == 1 + 0 + 0 + 1


@pytest.fixture
def server(monkeypatch, tmp_path):
    (tmp_path / "hello.txt").write_text("hello")
    (tmp_path / "app.js").write_text("let x = 1;")
    monkeypatch.setattr(serve, "HERE", str(tmp_path))
    monkeypatch.setattr(serve, "CONF", str(tmp_path / "dwave.conf"))  # missing, so status stays offline
    monkeypatch.setattr(serve, "DEMO", FAKE_DEMO)
    monkeypatch.setattr(serve, "STATUS", {"live": False, "time": None})
    httpd = serve.make_server(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers, r.read()


def post(url, obj):
    request = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_sample(server, monkeypatch):
    calls = []

    def fake(qpu):
        calls.append(qpu)
        return [READ_BAD, READ_GOOD], [1, 2], 1234.5

    monkeypatch.setattr(serve, "dwave_sample", fake)
    code, body = post(server + "/api/sample", {"size": 4, "index": 0})
    assert code == 200
    assert body == {"source": "dwave", "energies": [2.0, 0.0, 0.0], "best": [1, 0, 0, 1],
                    "qpu_access_time": 1234.5}
    assert calls == [FAKE_DEMO["puzzles"]["4"][0]["qpu"]]


def test_sample_failure(server, monkeypatch):
    def fake(qpu):
        raise RuntimeError("no network")

    monkeypatch.setattr(serve, "dwave_sample", fake)
    serve.STATUS.update(live=True, time=None)
    code, body = post(server + "/api/sample", {"size": 4, "index": 0})
    assert code == 503
    assert "no network" in body["error"]
    assert serve.STATUS["live"] is False

    assert post(server + "/api/sample", {"size": 4, "index": 5})[0] == 503
    assert post(server + "/api/sample", {"size": 7, "index": 0})[0] == 503
    assert post(server + "/api/sample", {"size": 4})[0] == 503


def test_status(server, monkeypatch):
    code, _, body = get(server + "/api/status")
    assert code == 200
    assert json.loads(body) == {"live": False}

    monkeypatch.setattr(serve, "STATUS", {"live": False, "time": None})
    monkeypatch.setattr(serve, "check_live", lambda: (True, "fake"))
    assert json.loads(get(server + "/api/status")[2]) == {"live": True}
    monkeypatch.setattr(serve, "check_live", lambda: (False, "fake"))
    assert json.loads(get(server + "/api/status")[2]) == {"live": True}  # cached for 30 s


def test_static_files(server):
    code, headers, body = get(server + "/hello.txt")
    assert (code, body) == (200, b"hello")
    assert headers["Cache-Control"] == "no-store"
    assert get(server + "/app.js")[1]["Content-Type"].startswith("text/javascript")


def test_token_not_served(server):
    with open(serve.CONF, "w") as f:
        f.write("[defaults]\ntoken = secret\n")
    for path in ["/dwave.conf", "/dwave%2Econf", "/./dwave.conf", "/dwave.conf?x=1", "/DWAVE.CONF", "/dwave.conf."]:
        with pytest.raises(urllib.error.HTTPError) as e:
            get(server + path)
        assert e.value.code == 404


def qp_message(rows, counts, active, num_qubits):
    """A made-up SAPI answer in the 'qp' format: rows are the spins of the active qubits."""
    def b64(fmt, values):
        return base64.b64encode(struct.pack("<" + fmt * len(values), *values)).decode()

    packed = b""
    for row in rows:
        row_bytes = bytearray(-(-len(active) // 8))
        for i, s in enumerate(row):
            if s > 0:
                row_bytes[i // 8] |= 0x80 >> (i % 8)
        packed += bytes(row_bytes)
    return {"type": "ising", "answer": {
        "format": "qp", "num_variables": num_qubits, "active_variables": b64("i", active),
        "energies": b64("d", [0.0] * len(rows)), "num_occurrences": b64("i", counts),
        "solutions": base64.b64encode(packed).decode(), "timing": {"qpu_access_time": 15000.5}}}


@pytest.mark.parametrize("use_numpy", [True, False])
def test_read_future(monkeypatch, use_numpy):
    """Decode through the client's own Future and solver, with and without numpy."""
    pytest.importorskip("dwave.cloud")
    from dwave.cloud import solver as solver_module
    from dwave.cloud.computation import Future

    # the client picks its decoder from this flag, set by trying `import numpy`
    monkeypatch.setattr(solver_module, "_numpy", use_numpy)
    solver = solver_module.StructuredSolver(client=None, data={
        "identity": {"name": "fake", "version": {"graph_id": "0"}}, "status": "online",
        "description": "", "avg_load": 0.0,
        "properties": {"supported_problem_types": ["ising", "qubo"], "qubits": list(range(12)),
                       "couplers": [[1, 3], [3, 4]], "parameters": {}}})
    active = [1, 3, 4, 9]
    rows = [[1, -1, 1, -1], [-1, -1, 1, 1]]
    future = Future(solver, "fake-id")
    future._set_message(qp_message(rows, [7, 3], active, 12))

    spins, counts, qpu_access_time = serve.read_future(future)
    assert [[row[q] for q in active] for row in spins] == rows
    assert all(len(row) == 12 for row in spins)
    assert counts == [7, 3]
    assert qpu_access_time == 15000.5
    json.dumps([spins, counts, qpu_access_time])  # plain Python types only
