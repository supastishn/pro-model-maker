import os
import sys
import time
import unittest
import subprocess
import requests

SERVER_URL = "http://127.0.0.1:5000"

def wait_for_server(url, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.post(url + "/v1/completions", json={"prompt": "ping", "model": os.getenv("MODEL", "test-model")})
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Server did not start in time")

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

    @classmethod
    def tearDownClass(cls):
        pass

    def test_text_completion(self):
        """Test /v1/completions endpoint"""
        payload = {
            "prompt": "Hello",
            "model": "should-be-overridden"
        }
        response = requests.post(SERVER_URL + "/v1/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("choices", data)
        self.assertIsInstance(data["choices"], list)

    def test_chat_completion(self):
        """Test /v1/chat/completions endpoint"""
        request_data = {
            "messages": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "First user message"},
                {"role": "user", "content": "Second user message"}
            ]
        }
        response = requests.post(SERVER_URL + "/v1/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("choices", data)
        self.assertIsInstance(data["choices"], list)

    def test_chat_completion_empty_iterations(self):
        """Test with zero iterations"""
        os.environ['NUM_ITERS'] = '0'
        request_data = {
            "messages": [
                {"role": "user", "content": "Test"}
            ]
        }
        response = requests.post(SERVER_URL + "/v1/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("choices", data)
        self.assertIsInstance(data["choices"], list)
        # Should return empty content for assistant
        self.assertEqual(data["choices"][0]["message"]["content"], "")

if __name__ == '__main__':
    unittest.main(verbosity=2)  # Show detailed test results
