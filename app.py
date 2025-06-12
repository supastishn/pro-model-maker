from flask import Flask, request, jsonify
from litellm import completion
import os
from dotenv import load_dotenv
import concurrent.futures

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

    # 1) run N parallel calls to the ITER_MODEL
    iters = int(os.getenv("NUM_ITERS", "1"))
    iter_model = os.getenv("ITER_MODEL")

    # copy original payload and swap in ITER_MODEL
    iter_payload = data.copy()
    iter_payload["model"] = iter_model
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

    # 2) build judger prompt: a system message + original messages + each iteration’s output
    judger_msgs = [
        {"role": "system", "content": "Judge between these solutions for the previous chat output"}
    ]
    judger_msgs.extend(orig_messages)
    for resp in iter_responses:
        # assume standard chat response shape
        content = resp["choices"][0]["message"]["content"]
        judger_msgs.append({"role": "assistant", "content": content})

    # call JUDGER_MODEL once, return its answer
    judger_payload = {
        "model": os.getenv("JUDGER_MODEL"),
        "messages": judger_msgs
    }
    judger_response = completion(**judger_payload)
    return jsonify(judger_response)

@app.route('/v1/completions', methods=['POST'])
def text_completions():
    data = request.json
    data = override_model_name(data)
    response = completion(**data)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
