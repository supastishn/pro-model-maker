import os
import sys
import time
import unittest
import openai
from openai import OpenAI

SERVER_URL = "http://127.0.0.1:5000"

def wait_for_server(url, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            # Use openai client to check server readiness
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                base_url="http://127.0.0.1:5000/v1"
            )
            # Try a simple chat completion call
            client.chat.completions.create(
                model=os.getenv("MODEL", "test-model"),
                messages=[{"role": "user", "content": "ping"}]
            )
            return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set environment variables for the tests (does not affect running server)
        os.environ['MODEL'] = 'test-model'
        os.environ['ITER_MODEL'] = 'test-iter-model'
        os.environ['JUDGER_MODEL'] = 'test-judger-model'
        os.environ['NUM_ITERS'] = '2'
        # Wait for the server to be ready
        wait_for_server(SERVER_URL)
        time.sleep(1)  # Give a little extra time for server to be ready

        # Set up OpenAI client for local server
        cls.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            base_url="http://127.0.0.1:5000/v1"
        )
        cls.model = os.getenv("MODEL", "test-model")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_chat_completion(self):
        """Test /v1/chat/completions endpoint using openai lib"""
        request_data = {
            "messages": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "First user message"},
                {"role": "user", "content": "Second user message"}
            ]
        }
        print("\n[TEST] Sending input messages to server:")
        for msg in request_data["messages"]:
            print(f"  {msg['role']}: {msg['content']}")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=request_data["messages"]
        )
        print("[TEST] Server returned:")
        print(f"  assistant: {resp.choices[0].message.content}\n")
        self.assertTrue(hasattr(resp, "choices"))
        self.assertIsInstance(resp.choices, list)
        self.assertTrue(hasattr(resp.choices[0], "message"))
        self.assertTrue(hasattr(resp.choices[0].message, "content"))

    def test_chat_completion_empty_iterations(self):
        """Test with zero iterations using openai lib"""
        os.environ['NUM_ITERS'] = '0'
        request_data = {
            "messages": [
                {"role": "user", "content": "Test"}
            ]
        }
        print("\n[TEST] Sending input messages to server:")
        for msg in request_data["messages"]:
            print(f"  {msg['role']}: {msg['content']}")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=request_data["messages"]
        )
        print("[TEST] Server returned:")
        print(f"  assistant: {resp.choices[0].message.content}\n")
        self.assertTrue(hasattr(resp, "choices"))
        self.assertIsInstance(resp.choices, list)
        self.assertTrue(hasattr(resp.choices[0], "message"))
        self.assertTrue(hasattr(resp.choices[0].message, "content"))
        # Should return empty content for assistant
        self.assertEqual(resp.choices[0].message.content, "")

if __name__ == '__main__':
    unittest.main(verbosity=2)  # Show detailed test results
