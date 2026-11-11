import unittest
import ingest


class IngestTests(unittest.TestCase):
    def test_financial_values_and_paragraphs(self):
        raw = 'Страница 3 из 15\n1. Процент-\nная ставка — 12,5 % годовых.\n\n2. Неустойка: 0,1 % в день.\n1000\n'
        result = ingest.clean_text(raw)
        self.assertIn('Процентная ставка — 12,5 % годовых.', result)
        self.assertIn('\n\n2.', result)
        self.assertIn('1000', result)  # это может быть значение тарифа, не номер страницы
        self.assertNotIn('Страница 3', result)

    def test_chunks_preserve_coverage(self):
        text = ' '.join(f'пункт_{i}' for i in range(100))
        chunks = ingest.chunk_text(text, 100, 20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(c) <= 100 for c in chunks))
        for word in text.split():
            self.assertTrue(any(word in c.split() for c in chunks), word)

    def test_bad_parameters(self):
        for size, overlap in [(0, 0), (100, 100), (100, -1), (40, 50)]:
            with self.assertRaises(ValueError):
                ingest.chunk_text('условия', size, overlap)
        self.assertEqual(ingest.chunk_text('', 100, 20), [])

    def test_categories(self):
        for text, category in [('Досрочное погашение', 'early_repayment'),
                               ('Штраф при просрочке', 'penalties'),
                               ('Процентная ставка', 'rates'), ('Срок договора', 'terms'),
                               ('Определения сторон', 'general')]:
            self.assertEqual(ingest.detect_category(text, 'cash_conditions.pdf'), category)


if __name__ == '__main__':
    unittest.main()
