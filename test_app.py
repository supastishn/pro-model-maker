import os
import unittest
from unittest.mock import patch, MagicMock
from app import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.orig_env = {}
        self.context = app.app_context()
        self.context.push()

    def tearDown(self):
        for k, v in self.orig_env.items():
            os.environ[k] = v
        self.context.pop()

    @patch('app.completion')
    def test_text_completion(self, mock_completion):
        """Test /v1/completions endpoint"""
        # Set up environment and mocks
        self.orig_env['MODEL'] = os.environ.get('MODEL', '')
        os.environ['MODEL'] = 'test-model'
        
        mock_response = {"choices": [{"text": "Mocked response"}]}
        mock_completion.return_value = mock_response.copy()
        
        # Make request
        response = self.app.post('/v1/completions', json={
            "prompt": "Hello",
            "model": "should-be-overridden"
        })
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, mock_response)
        
        # Verify model override
        self.assertEqual(mock_completion.call_args[1]['model'], 'test-model')

    @patch('app.completion')
    def test_chat_completion(self, mock_completion):
        """Test /v1/chat/completions endpoint"""
        # Set up environment
        self.orig_env.update({
            'NUM_ITERS': os.environ.get('NUM_ITERS', ''),
            'ITER_MODEL': os.environ.get('ITER_MODEL', ''),
            'JUDGER_MODEL': os.environ.get('JUDGER_MODEL', '')
        })
        os.environ.update({
            'NUM_ITERS': '2',
            'ITER_MODEL': 'test-iter-model',
            'JUDGER_MODEL': 'test-judger-model'
        })
        
        # Set up mock responses
        iter_responses = [
            {"choices": [{"message": {"content": f"Response {i}"}}]}
            for i in range(1, 3)
        ]
        judger_response = {"choices": [{"message": {"content": "Final response"}}]}
        
        mock_completion.side_effect = [*iter_responses, judger_response]
        
        # Make request
        request_data = {
            "messages": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "First user message"},
                {"role": "user", "content": "Second user message"}
            ]
        }
        response = self.app.post('/v1/chat/completions', json=request_data)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, judger_response)
        
        # Verify parallel calls
        iter_calls = mock_completion.call_args_list[:-1]
        self.assertEqual(len(iter_calls), 2)
        for call in iter_calls:
            kwargs = call[1]
            self.assertEqual(kwargs['model'], 'test-iter-model')
            self.assertEqual(kwargs['messages'], request_data['messages'])
        
        # Verify judger call
        judger_call = mock_completion.call_args_list[-1]
        self.assertEqual(judger_call[1]['model'], 'test-judger-model')
        self.assertEqual(judger_call[1]['messages'], [
            {"role": "system", "content": "Judge between these solutions for the previous chat output"},
            {"role": "user", "content": "First user message"},
            {"role": "user", "content": "Response 1\n\nResponse 2"}
        ])

    @patch('app.completion')
    def test_chat_completion_empty_iterations(self, mock_completion):
        """Test with zero iterations"""
        # Set up environment
        self.orig_env['NUM_ITERS'] = os.environ.get('NUM_ITERS', '')
        os.environ['NUM_ITERS'] = '0'
        
        mock_completion.return_value = {}
        self.app.post('/v1/chat/completions', json={
            "messages": [
                {"role": "user", "content": "Test"}
            ]
        })
        
        # Should skip parallel calls
        mock_completion.assert_not_called()

if __name__ == '__main__':
    unittest.main(verbosity=2)  # Show detailed test results
