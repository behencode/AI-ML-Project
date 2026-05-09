"""Launch the Streamlit app through the Google Colab port proxy.

Run this from a Colab notebook cell with:

    %run ui/colab_launcher.py
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.request


def _wait_for_streamlit(port: int, timeout_seconds: int = 30) -> bool:
    """Wait until the Streamlit HTTP server answers locally.

    Args:
        port: Local Streamlit port.
        timeout_seconds: Maximum seconds to wait.

    Returns:
        True when the server responds, otherwise False.
    """
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def launch_streamlit(port: int = 8501) -> None:
    """Start Streamlit in the background and expose it through Colab.

    Args:
        port: Local port used by Streamlit.

    Returns:
        None.
    """
    try:
        from google.colab import output
    except Exception as exc:
        raise RuntimeError("This launcher only works inside a Google Colab notebook.") from exc

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_path = os.path.join(project_root, "streamlit.log")
    app_path = os.path.join(project_root, "ui", "app.py")
    command = [
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.Popen(command, cwd=project_root, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"Starting Streamlit on Colab port {port}...")
    if not _wait_for_streamlit(port):
        print("Streamlit did not pass the health check. Showing the latest log output:")
        try:
            with open(log_path, "r", encoding="utf-8") as log_file:
                print(log_file.read()[-4000:])
        except OSError as exc:
            print(f"Could not read log file: {exc}")
        return
    proxy_url = output.eval_js(f"google.colab.kernel.proxyPort({port})")
    print(f"Streamlit is ready: {proxy_url}")
    print(f"Logs: {log_path}")
    print("If the embedded window is blank, open the printed URL in a new browser tab.")
    output.serve_kernel_port_as_window(port)


if __name__ == "__main__":
    launch_streamlit()
