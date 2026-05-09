"""Launch the Streamlit app through the Google Colab port proxy.

Run this from a Colab notebook cell with:

    %run ui/colab_launcher.py
"""

from __future__ import annotations

import os
import subprocess
import time


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
    time.sleep(4)
    print(f"Streamlit is starting on Colab port {port}. Logs: {log_path}")
    print("If the window does not load, run: !cat streamlit.log")
    output.serve_kernel_port_as_window(port)


if __name__ == "__main__":
    launch_streamlit()
