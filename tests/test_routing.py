import unittest
from unittest.mock import patch
from orchestrator import Orchestrator, detect_product

CASES = [
    ('Как считается ПСК кредита наличными?', 'question', 'analyst'),
    ('Какие сроки у кредита наличными?', 'question', 'analyst'),
    ('Проверь размер штрафа', 'verify', 'verifier'),
    ('Можно ли досрочно погасить кредит наличными?', 'verify', 'verifier'),
    ('Что будет при просрочке?', 'verify', 'verifier'),
    ('Если заплачу больше, пересчитают?', 'verify', 'verifier'),
    ('Объясни простыми словами', 'rewrite', 'editor'),
    ('Напиши короче', 'rewrite', 'editor'),
    ('Здравствуйте!', 'question', 'analyst'),
    ('Абвгд', 'question', 'analyst'),
]


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch('orchestrator.MCPTools'), patch('orchestrator.build_client'),
                        patch('orchestrator.memory.create_schema')]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.core = Orchestrator()

    def test_domain_routes(self):
        for text, intent, role in CASES:
            with self.subTest(text=text):
                self.assertEqual(self.core.detect_intent(text), intent)
                self.assertEqual(self.core.route(text, intent), role)

    def test_forbidden_transition_logs_rollback(self):
        self.assertEqual(self.core.route('s', 'rewrite'), 'editor')
        with self.assertLogs('orchestrator', level='WARNING') as logs:
            self.assertEqual(self.core.route('s', 'rewrite'), 'analyst')
        self.assertIn('откат', logs.output[0])
        self.assertEqual(self.core.route('s', 'unknown'), 'analyst')

    def test_product_detection(self):
        for query, expected in [('Кредит наличными', 'cash'), ('Карта Платинум', 'card'),
                                ('Сравни кредитную карту и кредит наличными', 'both'), ('Какова ставка?', None), ('Как погасить кредит наличными картой другого банка?', 'cash'), ('Как снять наличные с карты Платинум?', 'card')]:
            self.assertEqual(detect_product(query), expected)


if __name__ == '__main__':
    unittest.main()
