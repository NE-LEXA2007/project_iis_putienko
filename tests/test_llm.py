import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import requests
import config
from llm_client import GigaChatClient, OllamaClient, LLMError, build_client
from contracts import Chunk, Message
import prompts


class LLMTests(unittest.TestCase):
    def test_ollama_transport_and_real_usage(self):
        client = OllamaClient()
        response = Mock()
        response.json.return_value = {'message': {'content': '<think>Скрытые мысли</think>Ответ'},
                                      'prompt_eval_count': 51, 'eval_count': 7}
        with patch('requests.post', return_value=response) as post:
            self.assertEqual(client.generate([{'role': 'user', 'content': 'Вопрос'}]), 'Ответ')
        payload = post.call_args.kwargs['json']
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['model'], config.OLLAMA_MODEL)
        self.assertGreater(payload['options']['temperature'], 0)
        self.assertEqual(client.last_usage, {'prompt_tokens': 51, 'completion_tokens': 7})

    def test_transport_error_is_not_a_fake_answer(self):
        with patch('requests.post', side_effect=requests.ConnectionError('sensitive internal text')):
            with self.assertRaises(LLMError) as err:
                OllamaClient().generate([])
        self.assertNotIn('sensitive', str(err.exception))

    def test_gigachat_sdk_payload_and_usage(self):
        sdk = Mock()
        sdk.chat.return_value = NS(choices=[NS(message=NS(content='Ответ'))],
                                   usage=NS(prompt_tokens=100, completion_tokens=20))
        sdk.tokens_count.return_value = [NS(tokens=13)]
        with patch.object(config, 'GIGACHAT_CREDENTIALS', 'test-only-not-a-real-key'), patch('gigachat.GigaChat', return_value=sdk):
            client = GigaChatClient()
        self.assertEqual(client.generate([{'role': 'user', 'content': 'Вопрос'}], 0), 'Ответ')
        self.assertGreater(sdk.chat.call_args.args[0]['temperature'], 0)
        self.assertEqual(client.count_tokens('Текст'), 13)
        self.assertEqual(client.last_usage['completion_tokens'], 20)
        sdk.tokens_count.side_effect = RuntimeError()
        self.assertGreater(client.count_tokens('Русский текст'), 0)

    def test_missing_key_fails_explicitly(self):
        with patch.object(config, 'GIGACHAT_CREDENTIALS', ''):
            with self.assertRaises(LLMError):
                GigaChatClient()

    def test_switching_backend(self):
        with patch.object(config, 'LLM_BACKEND', 'ollama'):
            self.assertIsInstance(build_client(), OllamaClient)

    def test_prompt_order_and_replacing_context(self):
        history = [Message('user', 'Первый вопрос'), Message('assistant', 'Первый ответ')]
        chunks = [Chunk('Учебная ставка A', 'cash_conditions.pdf', 4, 'rates')]
        messages = prompts.build_messages('analyst', 'Текущий вопрос?', chunks, history)
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
        self.assertTrue(messages[-1]['content'].endswith('Текущий вопрос?'))
        self.assertEqual(sum('Текущий вопрос?' in m['content'] for m in messages), 1)
        chunks[0].text = 'Другой исходный пункт B'
        replaced = prompts.build_messages('analyst', 'Текущий вопрос?', chunks, history)
        self.assertNotIn('Учебная ставка A', replaced[-1]['content'])
        self.assertIn('Другой исходный пункт B', replaced[-1]['content'])
        self.assertIn('[cash_conditions.pdf, стр. 4]', replaced[-1]['content'])

    def test_uncited_tail_is_not_shown_as_document_fact(self):
        chunks = [Chunk('Подтверждённое правило', 'cash_conditions.pdf', 2, 'terms')]
        answer = 'Подтверждённое правило. [cash_conditions.pdf, стр. 2] Остаток автоматически переносится.'
        self.assertEqual(prompts.finish_answer(answer, chunks), 'Подтверждённое правило. [cash_conditions.pdf, стр. 2]')

    def test_unknown_citation_is_rejected(self):
        chunks = [Chunk('Правило', 'cash_conditions.pdf', 2, 'terms')]
        answer = 'Правило. [unknown.pdf, стр. 99]'
        self.assertIn('не подтвердила ссылки', prompts.finish_answer(answer, chunks))


if __name__ == '__main__':
    unittest.main()
