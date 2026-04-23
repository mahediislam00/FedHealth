"""
run.py — convenience launcher
Usage:  python run.py
        python run.py --port 8080
"""
import sys, subprocess

port = 5000
for i, a in enumerate(sys.argv[1:]):
    if a in ("--port", "-p") and i + 1 < len(sys.argv[1:]):
        port = int(sys.argv[i + 2])

print(f"\nStarting FedHealth Analytics on port {port}...")
print(f"Open your browser at:  http://127.0.0.1:{port}\n")

subprocess.run([sys.executable, "app.py"], env={**__import__("os").environ, "PORT": str(port)})
