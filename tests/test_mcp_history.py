"""Настоящий MCP-подпроцесс и SQLite; embedding-model для истории не нужна."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
import memory
from mcp_client import MCPTools


class MCPHistoryTests(unittest.TestCase):
    def test_real_mcp_reads_isolated_ordered_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            db_path = temp / 'history.db'
            root = Path(__file__).resolve().parents[1]
            with patch.object(config, 'DB_PATH', db_path):
                memory.create_schema()
                for i in range(10):
                    memory.save('first', 'user', f'Сообщение {i}')
                memory.save('second', 'user', 'Другая сессия')
                server = temp / 'server.py'
                server.write_text(
                    'import sys\nfrom pathlib import Path\n'
                    f'sys.path.insert(0, {str(root)!r})\n'
                    'import config\n'
                    f'config.DB_PATH = Path({str(db_path)!r})\n'
                    'from mcp_server import mcp\nmcp.run()\n', encoding='utf-8')
                tools = MCPTools(str(server), timeout=25)
                try:
                    messages = tools.get_history('first', 4)
                    self.assertEqual([m.content for m in messages],
                                     ['Сообщение 6', 'Сообщение 7', 'Сообщение 8', 'Сообщение 9'])
                    self.assertEqual([m.content for m in tools.get_history('second', 4)], ['Другая сессия'])
                    self.assertTrue(all(m.timestamp for m in messages))
                finally:
                    tools.close()
