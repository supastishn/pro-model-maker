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

    # Warn if 'prompt' is present in chat completion payload
    if 'prompt' in data:
        print("Warning: Removing 'prompt' from chat completion payload to avoid API error.")

    # 1) run N parallel calls to the ITER_MODEL
    iters = int(os.getenv("NUM_ITERS", "1"))
    iter_model = os.getenv("ITER_MODEL")

    # Early return if no iterations requested
    if iters <= 0:
        return jsonify({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": ""
                    }
                }
            ]
        })

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

    # collect all iteration outputs
    assistant_outputs = [
        resp["choices"][0]["message"]["content"]
        for resp in iter_responses
    ]
    combined_assistant = "\n\n".join(assistant_outputs)

    judger_msgs = [
        {
            "role": "system",
            "content": "Judge between these solutions for the previous chat output"
        },
        {"role": "user", "content": first_user},
        {"role": "user", "content": combined_assistant},
    ]

    # call JUDGER_MODEL once, return its answer
    # Sequential Thinking: Use the model from the request as the judger model
    judger_model = data.get("model", os.getenv("JUDGER_MODEL"))
    judger_payload = {
        "model": judger_model,
        "messages": judger_msgs
    }
    # Remove 'prompt' if present to avoid OpenAI/Router API error
    judger_payload.pop("prompt", None)
    judger_response = completion(**judger_payload)
    # Convert ModelResponse (pydantic) to dict for Flask jsonify
    return jsonify(judger_response.dict())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
