"""One-off setup for live D-Wave runs in the sudoku demo; safe to re-run.

    python setup_dwave.py                 create .venv, install the client, save the token, check
    python setup_dwave.py --token TOKEN   the same, replacing the saved token

Standard library only. The token goes in dwave.conf in this folder, and
serve.py reads only that file, so the laptop's own D-Wave settings are neither
used nor changed.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGION = "na-west-1"


def venv_python():
    if os.name == "nt":
        return os.path.join(HERE, ".venv", "Scripts", "python.exe")
    return os.path.join(HERE, ".venv", "bin", "python")


def data_region():
    """Region of the solver in data.js, or the default region if there is no data.js."""
    try:
        with open("data.js", encoding="utf-8") as f:
            text = f.read()
        return json.loads(text[text.index("=") + 1:text.rindex(";")])["chip"]["region"]
    except (OSError, ValueError, KeyError):
        return DEFAULT_REGION


def main():
    os.chdir(HERE)
    parser = argparse.ArgumentParser(description="Set up live D-Wave runs for the sudoku demo.")
    parser.add_argument("--token", help="D-Wave API token; replaces the one in dwave.conf")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        print(f"[FAIL] this is Python {sys.version.split()[0]}; dwave-cloud-client needs 3.10 or later")
        return 1

    if os.path.exists(venv_python()):
        print("[OK] .venv already exists")
    else:
        print("Creating .venv ...", flush=True)
        made = subprocess.run([sys.executable, "-m", "venv", ".venv"]).returncode == 0
        if not made or not os.path.exists(venv_python()):
            print("[FAIL] could not create .venv")
            print("       Delete any half-made .venv folder here and run this again.")
            return 1
        print("[OK] created .venv")

    print("Installing requirements.txt into .venv (needs internet, can take a minute) ...", flush=True)
    pip = [venv_python(), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", "requirements.txt"]
    if subprocess.run(pip).returncode != 0:
        print("[FAIL] pip install failed")
        print("       Check the internet connection, then run this again.")
        return 1
    print("[OK] installed requirements.txt")

    if args.token or not os.path.exists("dwave.conf"):
        token = args.token
        if not token:
            print("Paste your D-Wave API token. It is on the Leap dashboard: https://cloud.dwavesys.com/leap/")
            try:
                token = input("Token: ")
            except EOFError:
                token = ""
        token = token.strip()
        if not token or any(c.isspace() for c in token):
            print("[FAIL] no valid token given, so dwave.conf was not written")
            print("       Run this again, or use: python setup_dwave.py --token YOUR_TOKEN")
            return 1
        region = data_region()
        with open("dwave.conf", "w", encoding="utf-8") as f:
            f.write(f"[defaults]\ntoken = {token}\nregion = {region}\n")
        print(f"[OK] saved the token to dwave.conf (region {region})")
    else:
        print("[OK] dwave.conf already exists (use --token to replace the token)")

    print("Checking the connection and the chip mappings ...", flush=True)
    if subprocess.run([venv_python(), "serve.py", "--check"]).returncode != 0:
        print("[FAIL] the check failed; see the lines above")
        return 1
    print("[OK] ready: run python serve.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
