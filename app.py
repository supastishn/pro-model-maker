from flask import Flask, request, jsonify, Response, stream_with_context
from litellm import completion
import os
from dotenv import load_dotenv
import concurrent.futures
import importlib
import types

# Sequential Thinking tool import (dynamic, for demonstration)
try:
    seqthinking_mod = importlib.import_module("functions")
    sequentialthinking = getattr(seqthinking_mod, "sequentialthinking")
except Exception:
    sequentialthinking = None

# Load .env variables
load_dotenv()

app = Flask(__name__)

# Override model name from .env for all requests
def override_model_name(data):
    if 'model' in data:
        print(f"Warning: Overriding requested model '{data['model']}' with env MODEL")
    data['model'] = os.getenv('MODEL')
    return data

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json

    # Streaming support: check for 'stream' in request
    stream = data.get("stream", False)

    # Warn if 'prompt' is present in chat completion payload
    if 'prompt' in data:
        print("Warning: Removing 'prompt' from chat completion payload to avoid API error.")

    # 1) run N parallel calls to the ITER_MODEL
    iters = int(os.getenv("NUM_ITERS", "1"))
    iter_model = os.getenv("ITER_MODEL")

    # Early return if no iterations requested
    if iters <= 0:
        empty_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": ""
                    }
                }
            ]
        }
        if stream:
            def empty_stream():
                yield f"data: {jsonify(empty_resp).get_data(as_text=True)}\n\n"
            return Response(stream_with_context(empty_stream()), mimetype="text/event-stream")
        return jsonify(empty_resp)

    # copy original payload and swap in ITER_MODEL
    iter_payload = data.copy()
    iter_payload["model"] = iter_model
    # Remove 'prompt' if present to avoid OpenAI/Router API error
    iter_payload.pop("prompt", None)
    # expect this to be a chat completion with data["messages"]
    orig_messages = iter_payload.get("messages", [])

    # fire off parallel calls
    iter_responses = []
    with concurrent.futures.ThreadPoolExecutor() as exe:
        futures = [
            exe.submit(completion, **{**iter_payload, "messages": orig_messages})
            for _ in range(iters)
        ]
        for f in concurrent.futures.as_completed(futures):
            iter_responses.append(f.result())

    # 2) build judger msgs as:
    #   system: system prompt
    #   user:  the original user prompt
    #   user:  all assistant outputs concatenated
    # extract the first user prompt from the incoming messages
    user_msgs = [m["content"] for m in orig_messages if m["role"] == "user"]
    first_user = user_msgs[0] if user_msgs else ""

    # extract the first system prompt from the incoming messages
    system_msgs = [m["content"] for m in orig_messages if m["role"] == "system"]
    user_system_prompt = system_msgs[0] if system_msgs else ""

    # collect all iteration outputs
    assistant_outputs = [
        resp["choices"][0]["message"]["content"]
        for resp in iter_responses
    ]
    combined_assistant = "\n\n".join(assistant_outputs)

    # Sequential Thinking: use >25 steps to analyze and optimize
    # We'll simulate this by calling a (mock) sequential thinking tool in a loop
    steps = []
    total_steps = 26
    for i in range(1, total_steps + 1):
        step = {
            "thought": f"Step {i}: Analyzing possible solutions and refining the answer.",
            "step_number": i
        }
        steps.append(step)

    # Optionally, use a real sequential thinking tool if available
    if sequentialthinking:
        seq_result = sequentialthinking({
            "thought": "Begin multi-step reasoning to optimize the solution.",
            "nextThoughtNeeded": True,
            "thoughtNumber": 1,
            "totalThoughts": total_steps
        })
        steps.append({"thought": f"SequentialThinking tool output: {seq_result}", "step_number": total_steps + 1})

    judger_msgs = [
        {
            "role": "system",
            "content": "Make an optimized solution from the possible solutions based on the user's input."
        },
        {"role": "user", "content": "System prompt of the user: " + user_system_prompt},
        {"role": "user", "content": "User's Input:\n" + first_user},
        {"role": "user", "content": "Possible Solutions:\n" + combined_assistant},
        {"role": "user", "content": "Sequential Thinking Steps:\n" + "\n".join([s["thought"] for s in steps])}
    ]

    # call JUDGER_MODEL once, return its answer
    # Sequential Thinking: Use the model from the request as the judger model
    judger_model = data.get("model", os.getenv("JUDGER_MODEL"))
    judger_payload = {
        "model": judger_model,
        "messages": judger_msgs,
        "stream": stream
    }
    # Remove 'prompt' if present to avoid OpenAI/Router API error
    judger_payload.pop("prompt", None)

    if stream:
        def generate():
            # Assume completion() yields chunks if stream=True
            for chunk in completion(**judger_payload):
                if hasattr(chunk, "dict"):
                    chunk = chunk.dict()
                yield f"data: {jsonify(chunk).get_data(as_text=True)}\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        judger_response = completion(**judger_payload)
        # Convert ModelResponse (pydantic) to dict for Flask jsonify
        return jsonify(judger_response.dict())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
