import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'memory.db'
        self.patch = patch.object(memory.config, 'DB_PATH', self.path)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        memory.create_schema()

    def test_last_four_chronologically_and_isolation(self):
        for n in range(10):
            memory.save('a', 'user', str(n))
        memory.save('b', 'user', 'другая сессия')
        self.assertEqual([m.content for m in memory.window('a', 4)], ['6', '7', '8', '9'])
        self.assertEqual(len(memory.window('b', 4)), 1)
        self.assertEqual(memory.window('a', 0), [])
        with self.assertRaises(ValueError):
            memory.window('a', -1)

    def test_persistence_in_another_process(self):
        memory.save('persist', 'user', 'Сохранённое сообщение')
        memory.set_product('persist', 'card')
        code = "import memory; from pathlib import Path; memory.config.DB_PATH=Path(__import__('sys').argv[1]); assert memory.window('persist')[0].content=='Сохранённое сообщение'; assert memory.get_product('persist')=='card'"
        run = subprocess.run([sys.executable, '-c', code, str(self.path)], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_sql_parameters_and_clear(self):
        hostile = "x'; DROP TABLE messages;--"
        memory.save(hostile, 'user', 'Безопасный текст')
        memory.save('another', 'assistant', 'Не удалять')
        memory.set_product(hostile, 'cash')
        memory.clear(hostile)
        self.assertEqual(memory.window(hostile), [])
        self.assertIsNone(memory.get_product(hostile))
        self.assertEqual(len(memory.window('another')), 1)

    def test_prune(self):
        memory.save('old', 'user', 'old')
        memory.save('new', 'user', 'new')
        memory.set_product('old', 'card')
        with memory.connect() as db:
            db.execute("UPDATE messages SET timestamp='2000-01-01T00:00:00+00:00' WHERE session_id='old'")
        self.assertEqual(memory.prune(30), 1)
        self.assertEqual(len(memory.window('new')), 1)
        self.assertIsNone(memory.get_product('old'))


if __name__ == '__main__':
    unittest.main()
