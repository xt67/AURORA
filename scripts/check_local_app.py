"""Bounded loopback-only deployment health smoke check; always stops its server."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
               "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true",
               "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
    with (out / "streamlit-health.log").open("w") as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Streamlit exited; inspect outputs/verification/streamlit-health.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        if response.status == 200 and response.read() == b"ok":
                            print("PASS: Streamlit loopback health endpoint returned 200/ok")
                            return
                except OSError:
                    time.sleep(.25)
            raise TimeoutError("Streamlit health check timed out")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
