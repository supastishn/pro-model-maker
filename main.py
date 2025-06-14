#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()  # pick up .env for OPENAI_API_KEY

SERVER_URL = "http://127.0.0.1:5000/v1"
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=SERVER_URL
)

def start_server():
    p = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line-buffered output
    )
    while True:
        line = p.stdout.readline()
        if not line:
            break
        print(line, end="")
        # Changed to match the complete serving message including port number
        if ":5000" in line and "Running on" in line:
            break
    return p

def cli_loop():
    print("Chat CLI (type 'exit' or Ctrl-D to quit)\n")
    model = os.getenv("MODEL")
    
    # Ask for streaming preference
    streaming_input = input(
        "Do you want to stream responses? (y/n) [default=n]: "
    ).strip().lower()
    use_stream = streaming_input == 'y'
    stream_mode = "Streaming" if use_stream else "Non-streaming"

    print(f"\n{stream_mode} mode activated.\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break

        try:
            if use_stream:
                # Start assistant message prefix
                print("Assistant: ", end='', flush=True)
                
                # Process streaming response
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end='', flush=True)
                print("\n", flush=True)
            else:
                # Regular non-streaming flow
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                msg = resp.choices[0].message.content
                print("Assistant:", msg, "\n")
        except Exception as e:
            print("Error:", e, "\n")

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
