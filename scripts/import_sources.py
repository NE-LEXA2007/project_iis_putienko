"""Импорт переданных PDF по SHA-256, независимо от исходного имени файла."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path, help='PDF-файлы или каталог с тремя PDF')
    args = parser.parse_args()
    entries = json.loads((ROOT / 'sources.json').read_text(encoding='utf-8'))
    expected = {item['sha256']: item for item in entries}
    matched = set()
    for incoming in args.paths:
        files = sorted(incoming.glob('*.pdf')) if incoming.is_dir() else [incoming]
        for file in files:
            checksum = hashlib.sha256(file.read_bytes()).hexdigest()
            if checksum not in expected:
                continue
            item = expected[checksum]
            target = ROOT / 'data/raw' / item['file']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                raise ValueError('Существующий файл другой редакции не перезаписан: ' + item['file'])
            if file.resolve() != target.resolve():
                shutil.copyfile(file, target)
            matched.add(item['file'])
            print(item['file'], 'SHA-256 OK')
    missing = {item['file'] for item in entries} - matched
    if missing:
        raise SystemExit('Не найдены PDF нужной редакции: ' + ', '.join(sorted(missing)))


if __name__ == '__main__':
    main()
