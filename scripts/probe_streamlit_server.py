"""Script to launch Streamlit in background, probe HTTP 200, and gracefully terminate."""

import subprocess
import sys
import time
import urllib.request

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py",
    "--server.headless",
    "true",
    "--server.port",
    "8501",
]

print(f"Launching Streamlit on port 8501: {' '.join(cmd)}")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    print("Waiting for server startup...")
    server_ready = False
    for attempt in range(12):
        time.sleep(1.5)
        try:
            req = urllib.request.Request("http://localhost:8501")
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"[Probe Success] HTTP Status: {resp.status} | Content-Type: {resp.headers.get('Content-Type')}")
                server_ready = True
                break
        except Exception as e:
            print(f"Attempt {attempt + 1}: server not ready yet ({e})")

    if not server_ready:
        raise RuntimeError("Streamlit server failed to start within timeout.")
    print("Streamlit local web app probe verified successfully!")
finally:
    print("Terminating test server process...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Test server terminated cleanly.")

