"""Проверяет ядро и настоящую SQLite с моками MCP/LLM. Это не RAG-бенчмарк."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import memory
from contracts import Chunk
from orchestrator import Orchestrator, PRODUCT_QUESTION


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_patch = patch.object(memory.config, 'DB_PATH', Path(self.temp.name) / 'memory.db')
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.tools = Mock()
        self.tools.search_filtered.side_effect = self.search
        self.client = Mock()
        self.client.count_tokens.return_value = 10
        self.client.generate.return_value = 'Учебный ответ из контролируемого мока [cash_conditions.pdf, стр. 2]'
        self.client.last_usage = None
        for p in [patch('orchestrator.MCPTools', return_value=self.tools),
                  patch('orchestrator.build_client', return_value=self.client)]:
            p.start()
            self.addCleanup(p.stop)
        self.core = Orchestrator()

    @staticmethod
    def search(text, filters, k=None):
        file = filters['source'] if isinstance(filters['source'], str) else filters['source']['$in'][0]
        return [Chunk('Учебный фрагмент ' + text, file, 2, 'general')]

    def test_three_turn_memory_and_role_context(self):
        self.core.handle('s', 'Кредит наличными: как считается ПСК?')
        self.core.handle('s', 'Объясни простыми словами')
        self.assertEqual(self.core._state['s'], 'editor')
        self.assertEqual(len(self.core._context['s']), 1)  # редактор сохраняет исходный контекст
        self.core.handle('s', 'А если погасить досрочно?')
        payload = self.client.generate.call_args.args[0]
        self.assertEqual(payload[1]['content'], 'Кредит наличными: как считается ПСК?')
        self.assertEqual(sum('А если погасить досрочно?' in m['content'] for m in payload), 1)
        self.assertEqual(len(memory.window('s')), 6)
        self.assertEqual(self.core._state['s'], 'verifier')
        self.assertTrue(all(c.source in {'cash_conditions.pdf','cash_tariff.pdf','bank_conditions.pdf'} for c in self.core._context['s']))

    def test_question_across_categories_keeps_both_sections(self):
        self.core.handle('s', 'Кредит наличными: как меняются проценты при досрочном погашении?')
        filters = self.tools.search_filtered.call_args_list[0].args[1]
        self.assertEqual(filters['source']['$in'], ['cash_conditions.pdf', 'cash_tariff.pdf', 'bank_conditions.pdf'])
        self.assertNotIn('category', filters)

    def test_single_category_still_filtered(self):
        self.core.handle('s', 'Кредит наличными: можно ли погасить досрочно?')
        filters = self.tools.search_filtered.call_args_list[0].args[1]
        self.assertEqual(filters['category'], 'early_repayment')
        self.assertEqual(self.tools.search_filtered.call_args_list[1].args[1]['source'], 'cash_tariff.pdf')

    def test_out_of_scope_card_does_not_search(self):
        answer = self.core.handle('new', 'Карта Платинум: какой льготный период?')
        self.assertIn('В базе знаний нет ответа', answer)
        self.tools.search_filtered.assert_not_called()
        self.client.generate.assert_not_called()

    def test_single_product_does_not_require_repeated_selection(self):
        self.core.handle('s', 'Какой срок уведомления о досрочном погашении?')
        self.assertEqual(memory.get_product('s'), 'cash')
        self.assertEqual(self.tools.search_filtered.call_args.args[1]['product'], 'cash')

    def test_isolation_and_reset(self):
        self.core.handle('a', 'Кредит наличными: какой срок?')
        self.core.handle('b', 'Какой срок уведомления по кредиту наличными?')
        self.core.reset('a')
        self.assertEqual(memory.window('a'), [])
        self.assertIsNone(memory.get_product('a'))
        self.assertEqual(len(memory.window('b')), 2)
        self.assertNotIn('a', self.core._context)


if __name__ == '__main__':
    unittest.main()
