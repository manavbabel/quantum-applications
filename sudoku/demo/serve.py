"""Local server for the sudoku annealing demo.

    python serve.py                start the server and open the page
    python serve.py --no-browser   start the server only
    python serve.py --check        check the D-Wave connection and the chip mappings in data.js

Serves this folder at http://localhost:8765/ and sends puzzles to D-Wave for
the page. If .venv exists here (made by setup_dwave.py), the script re-runs
itself with that venv's Python. Without dwave-cloud-client or dwave.conf, the
page is still served and /api/status reports that live runs are off.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(HERE, "dwave.conf")
DATA = os.path.join(HERE, "data.js")
PORT = 8765
LABEL = "nqcc-sudoku-demo"
STATUS_TTL = 30  # seconds between D-Wave checks
CONNECT_TIMEOUT = 8  # seconds
SAMPLE_TIMEOUT = 18  # seconds; the page gives up after 20

DEMO = None  # contents of data.js, loaded by main()
DWAVE = {}  # the client and solver, kept once fetched
DWAVE_LOCK = threading.Lock()
STATUS = {"live": False, "time": None}
STATUS_LOCK = threading.Lock()


def log(text):
    print(time.strftime("%H:%M:%S"), text.encode("ascii", "replace").decode("ascii"), flush=True)


def describe(error):
    return f"{type(error).__name__}: {error}".encode("ascii", "replace").decode("ascii")


def load_demo(path):
    """data.js is one line, `window.DEMO = <json>;`."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return json.loads(text[text.index("=") + 1:text.rindex(";")])


def venv_python():
    if os.name == "nt":
        return os.path.join(HERE, ".venv", "Scripts", "python.exe")
    return os.path.join(HERE, ".venv", "bin", "python")


def in_venv():
    def norm(path):
        return os.path.normcase(os.path.realpath(path))
    return norm(sys.prefix) == norm(os.path.join(HERE, ".venv"))


def rerun_in_venv():
    # the variable stops a loop if the venv's Python fails to recognise itself
    env = dict(os.environ, SUDOKU_DEMO_VENV="1")
    proc = subprocess.Popen([venv_python(), os.path.abspath(__file__)] + sys.argv[1:], env=env)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            pass  # Ctrl+C reaches the child too, which stops itself


def majority(spins, chain):
    """Logical bit of one chain: 1 if most of its spins are +1, and 0 on a tie."""
    return 1 if sum(spins[q] for q in chain) > 0 else 0


def energy(bits, groups):
    """Sum over the "exactly one" groups of (sum x - 1)^2; 0 means the grid is solved."""
    return sum((sum(bits[i] for i in g) - 1) ** 2 for g in groups)


def call_with_timeout(fn, timeout):
    """Run fn in a daemon thread and give up waiting after `timeout` seconds."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except Exception as e:
            box["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"no answer within {timeout} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def connect():
    """Fetch the solver named in data.js, creating the client on first use."""
    from dwave.cloud import Client

    if not DWAVE_LOCK.acquire(timeout=CONNECT_TIMEOUT):
        raise TimeoutError("an earlier connection attempt is still running")
    try:
        if "client" not in DWAVE:
            # this folder's dwave.conf only; the short connect timeout and few retries stop a
            # dead network holding a thread for minutes (reads stay long for the 30 s long polls)
            DWAVE["client"] = Client.from_config(config_file=CONF, request_timeout=(5, 60),
                                                 http_retry_total=2)
        DWAVE["solver"] = DWAVE["client"].get_solver(name=DEMO["chip"]["solver"])
        return DWAVE["solver"]
    finally:
        DWAVE_LOCK.release()


def check_live():
    """(live, reason) for the staff console."""
    if DEMO is None:
        return False, "data.js is missing"
    if not os.path.exists(CONF):
        return False, "no dwave.conf (run setup_dwave.py)"
    try:
        import dwave.cloud  # noqa: F401
    except ImportError:
        return False, "dwave-cloud-client is not installed (run setup_dwave.py)"
    try:
        solver = call_with_timeout(connect, CONNECT_TIMEOUT)
    except Exception as e:
        return False, describe(e)
    if not solver.online:
        return False, f"{solver.name} is offline"
    return True, f"{solver.name} is online"


def status():
    """Whether the page should try D-Wave, re-checked at most every STATUS_TTL seconds."""
    with STATUS_LOCK:
        if STATUS["time"] is None or time.monotonic() - STATUS["time"] >= STATUS_TTL:
            live, reason = check_live()
            if STATUS["time"] is None or live != STATUS["live"]:
                log(f"[OK] D-Wave live: {reason}" if live else
                    f"[FAIL] D-Wave off, the page will show the recorded simulation: {reason}")
            STATUS.update(live=live, time=time.monotonic())
        return STATUS["live"]


def mark_offline():
    """After a failed job, send visitors straight to the recording until the next check."""
    with STATUS_LOCK:
        STATUS.update(live=False, time=time.monotonic())


def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def read_future(future):
    """Spins (one row per distinct read, indexed by qubit), occurrences and QPU access time (us).

    Reads the client's own decoded answer, so dimod is not needed. The client
    decodes with numpy when it is installed and converts back to lists; without
    numpy the counts come back as tuples. Qubits outside the problem read as 0
    with numpy and -1 without, which never matters here since every chain
    qubit is in h or J.
    """
    spins = as_list(future.samples)
    counts = [int(k) for k in as_list(future.num_occurrences)]
    return spins, counts, float(future.timing["qpu_access_time"])


def dwave_sample(qpu):
    """Send one puzzle's physical Ising problem to D-Wave; returns what read_future does."""
    start = time.monotonic()
    if not os.path.exists(CONF):
        raise RuntimeError("no dwave.conf (run setup_dwave.py)")
    solver = DWAVE.get("solver") or call_with_timeout(connect, CONNECT_TIMEOUT)
    h = {q: bias for q, bias in qpu["h"]}
    J = {(a, b): bias for a, b, bias in qpu["J"]}
    future = solver.sample_ising(h, J, num_reads=qpu["num_reads"],
                                 annealing_time=qpu["annealing_time"], label=LABEL)
    if not future.wait(max(0.1, SAMPLE_TIMEOUT - (time.monotonic() - start))):
        future.cancel()
        raise TimeoutError(f"no answer from D-Wave within {SAMPLE_TIMEOUT} s")
    return read_future(future)


def sample(size, index):
    """The /api/sample answer for one puzzle."""
    if DEMO is None:
        raise RuntimeError("data.js is missing")
    puzzles = DEMO["puzzles"][str(size)]
    if not 0 <= index < len(puzzles):
        raise IndexError(f"no puzzle {index} for size {size}")
    puzzle = puzzles[index]
    spins, counts, qpu_access_time = dwave_sample(puzzle["qpu"])
    energies, best, best_energy = [], None, None
    for row, k in zip(spins, counts):
        bits = [majority(row, chain) for chain in puzzle["qpu"]["chains"]]
        e = float(energy(bits, puzzle["groups"]))
        energies += [e] * k
        if best is None or e < best_energy:
            best, best_energy = bits, e
    return {"source": "dwave", "energies": energies, "best": best, "qpu_access_time": qpu_access_time}


class Handler(SimpleHTTPRequestHandler):
    # Windows can map .js to text/plain through the registry
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".html": "text/html; charset=utf-8",
                      ".js": "text/javascript; charset=utf-8",
                      ".css": "text/css; charset=utf-8"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def log_message(self, format, *args):
        pass

    def send_head(self):
        # the folder is served as it is, and that includes the token
        real = os.path.normcase(os.path.realpath(self.translate_path(self.path)))
        if real == os.path.normcase(os.path.realpath(CONF)):
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        # so a regenerated data.js or an edited app.js is picked up on reload
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            pass  # the page stopped waiting

    def do_GET(self):
        if self.path.split("?")[0] == "/api/status":
            self.send_json(200, {"live": status()})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/sample":
            self.send_json(404, {"error": "not found"})
            return
        size = index = None
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            size, index = int(body["size"]), int(body["index"])
            start = time.monotonic()
            result = sample(size, index)
        except Exception as e:
            log(f"[FAIL] D-Wave job for size {size} puzzle {index}: {describe(e)}")
            mark_offline()
            self.send_json(503, {"error": describe(e)})
            return
        solved = sum(e == 0 for e in result["energies"])
        log(f"[OK] D-Wave job for {size}x{size} puzzle {index}: {solved} of {len(result['energies'])} "
            f"reads solved it, {result['qpu_access_time'] / 1000:.1f} ms of QPU time, "
            f"{time.monotonic() - start:.1f} s in all")
        self.send_json(200, result)


def make_server(port):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler, bind_and_activate=False)
    # on Windows, address reuse would let a second copy of this server share the port
    server.allow_reuse_address = os.name != "nt"
    try:
        server.server_bind()
        server.server_activate()
    except OSError:
        server.server_close()
        raise
    return server


def run_server(open_browser):
    try:
        server = make_server(PORT)
    except OSError as e:
        print(f"[FAIL] cannot listen on port {PORT}; is serve.py already running? ({describe(e)})")
        return 1
    url = f"http://localhost:{PORT}/"
    print(f"Sudoku demo at {url} (Ctrl+C to stop)", flush=True)
    if DEMO is None:
        log("[FAIL] data.js not found; run precompute.py and pull the new data.js")
    status()
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    finally:
        server.server_close()
    return 0


def check():
    """Connect with ./dwave.conf and check every qubit and coupler in data.js is still working."""
    try:
        from dwave.cloud.package_info import __version__
    except ImportError:
        print("[FAIL] dwave-cloud-client is not installed: run setup_dwave.py")
        return 1
    print(f"[OK] dwave-cloud-client {__version__}")
    if DEMO is None:
        print("[FAIL] data.js not found: run precompute.py and pull the new data.js")
        return 1
    if not os.path.exists(CONF):
        print("[FAIL] dwave.conf not found: run setup_dwave.py")
        return 1
    name = DEMO["chip"]["solver"]
    try:
        solver = connect()
    except Exception as e:
        print(f"[FAIL] could not fetch {name}: {describe(e)}")
        print("       Check the internet connection, and the token and region in dwave.conf")
        print("       (python setup_dwave.py --token NEW_TOKEN replaces the token).")
        return 1
    ok = solver.online
    print(f"[OK] fetched {name}, which is online" if ok else f"[FAIL] fetched {name}, but it is offline")
    for size, puzzles in DEMO["puzzles"].items():
        problems = []
        for i, p in enumerate(puzzles):
            qpu = p["qpu"]
            qubits = {q for chain in qpu["chains"] for q in chain} | {q for q, _ in qpu["h"]}
            lost_qubits = qubits - solver.nodes
            lost_couplers = [(a, b) for a, b, _ in qpu["J"] if (a, b) not in solver.edges]
            if lost_qubits or lost_couplers:
                problems.append(f"puzzle {i} has {len(lost_qubits)} qubit(s) and "
                                f"{len(lost_couplers)} coupler(s) no longer working")
        if problems:
            ok = False
            print(f"[FAIL] {size}x{size}: " + "; ".join(problems))
        else:
            print(f"[OK] {size}x{size}: all {len(puzzles)} mappings fit the chip")
    if not ok:
        print("       To fix a mapping, re-run precompute.py in the repo's uv environment and pull the new data.js.")
    return 0 if ok else 1


def main():
    os.chdir(HERE)
    parser = argparse.ArgumentParser(description="Serve the sudoku annealing demo.")
    parser.add_argument("--check", action="store_true",
                        help="check the D-Wave connection and the chip mappings, then exit")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args = parser.parse_args()

    if os.path.exists(venv_python()) and not in_venv() and not os.environ.get("SUDOKU_DEMO_VENV"):
        return rerun_in_venv()

    global DEMO
    if os.path.exists(DATA):
        try:
            DEMO = load_demo(DATA)
        except (OSError, ValueError) as e:
            print(f"[FAIL] could not read data.js: {describe(e)}")
    if args.check:
        return check()
    return run_server(open_browser=not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
