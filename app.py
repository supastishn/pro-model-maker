from flask import Flask, request, jsonify
from litellm import completion
import os
from dotenv import load_dotenv

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
    data = override_model_name(data)
    response = completion(**data)
    return jsonify(response)

@app.route('/v1/completions', methods=['POST'])
def text_completions():
    data = request.json
    data = override_model_name(data)
    response = completion(**data)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
