# Sudoku annealing demo

An interactive sudoku demo for the NQCC stand at New Scientist Live. Visitors solve a puzzle against the clock, watch a recorded simulated quantum annealing run settle the same puzzle, then send it to a D-Wave quantum annealer.

## Running it

The demo laptop needs Python 3.10 or later and nothing else; 3.10 is the minimum for `dwave-cloud-client` 0.14.6.

```
cd quantum-applications/sudoku/demo
python setup_dwave.py
python serve.py
```

Run `setup_dwave.py` once, with internet, before the event. It creates `.venv` in this folder, installs `requirements.txt` into it, asks for the D-Wave API token (it's on the Leap dashboard) and saves it to `dwave.conf` here, then runs `python serve.py --check`. It's safe to re-run, and `python setup_dwave.py --token NEW_TOKEN` replaces the token. `dwave.conf` is git-ignored, and `serve.py` reads only that file, so the laptop's own D-Wave settings are neither used nor changed.

`serve.py` serves this folder at http://localhost:8765/ and opens it in the default browser. Press F11 for full screen, and Ctrl+C in the console to stop the server. It switches to `.venv` by itself, so there's nothing to activate. The console says whether D-Wave can be reached and prints one line per D-Wave job. `python serve.py --no-browser` starts the server without opening the page.

The visitor steps through "Send to quantum annealer" with a Next button, and nothing goes to D-Wave until they press Send on the third step.

Opening `index.html` directly also works, but then "Send to quantum annealer" always shows the recorded simulation. Through `serve.py`, the recording is also shown whenever D-Wave can't be used: no token, no internet, no answer within 18 s, or any error.

The small dot in the bottom-right corner tells staff where the last result came from: filled for D-Wave, hollow for the recorded simulation.

## If the check fails

`python serve.py --check` connects with `dwave.conf`, fetches the solver named in `data.js` (free; nothing is sent to the QPU) and checks that every qubit and coupler the puzzles use is still working. It prints `[OK]` or `[FAIL]` for each step and each puzzle size.

- can't connect: check the internet connection, then the token (`python setup_dwave.py --token NEW_TOKEN`)
- a mapping no longer fits, because D-Wave has taken qubits or couplers out of service: on a machine with the repo's uv environment, run `uv run python sudoku/demo/precompute.py` from the repo root, commit the new `data.js` and pull it onto the laptop

## Regenerating data.js

`data.js` holds the puzzles, the recorded annealing runs, the chip mappings and the fallback results. To rebuild it, from the repo root:

```
uv run python sudoku/demo/precompute.py
```

It fetches the solver's working graph from D-Wave (free), so it needs the D-Wave account set up for the repo (see the top-level README).

## Tests

```
uv run pytest sudoku
```
