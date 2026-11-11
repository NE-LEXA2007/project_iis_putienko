"""Восстановление публичных PDF по точным URL и SHA-256 снимка."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch():
    for item in json.loads((ROOT / 'sources.json').read_text(encoding='utf-8')):
        path = ROOT / 'data/raw' / item['file']
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
            print(item['file'], 'уже есть, SHA-256 OK')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(item['url'], timeout=60) as response:
            data = response.read()
        if not data.startswith(b'%PDF-') or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(item['file'] + ': PDF изменился относительно использованной редакции; источник нужно проверить заново')
        path.write_bytes(data)
        print(item['file'], 'скачан, SHA-256 OK')


if __name__ == '__main__':
    fetch()
