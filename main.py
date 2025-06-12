#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()  # pick up .env for MODEL, ITER_MODEL, etc.

SERVER_URL = "http://127.0.0.1:5000"
CHAT_ENDPOINT = f"{SERVER_URL}/v1/chat/completions"

def start_server():
    # Launch app.py in background
    p = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait until Flask signals it’s ready
    while True:
        line = p.stdout.readline()
        if not line:
            break
        print(line, end="")        # echo server log
        if "Running on" in line:
            break
    return p

def cli_loop():
    print("Chat CLI (type 'exit' or Ctrl-D to quit)\n")
    model = os.getenv("MODEL")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        r = requests.post(CHAT_ENDPOINT, json=payload)
        if r.ok:
            resp = r.json()
            # assume chat response format
            msg = resp["choices"][0]["message"]["content"]
            print("Assistant:", msg, "\n")
        else:
            print(f"Error {r.status_code}: {r.text}\n")

def main():
    server = start_server()
    try:
        cli_loop()
    finally:
        # gracefully shut down Flask
        server.send_signal(signal.SIGINT)
        server.wait()

if __name__ == "__main__":
    main()
