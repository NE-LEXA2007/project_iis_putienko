"""Установка проверенной CPU-версии Ollama в каталог проекта (Linux x86_64)."""
import hashlib
from pathlib import Path
import platform
import tarfile
import urllib.request
import zstandard

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.40.16'
URL = f'https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-linux-amd64.tar.zst'
# Опубликован в sha256sum.txt этого официального релиза.
SHA256 = '87bfafb2acbc89b2e6a1be0a9043de6aa75fb450696f86bc426e65fc448f788d'


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install():
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'AMD64'}:
        raise SystemExit('Этот скрипт для Linux x86_64/WSL. На macOS установите Ollama с официального сайта.')
    archive = ROOT / '.cache/ollama-linux-amd64.tar.zst'
    archive.parent.mkdir(exist_ok=True)
    if not archive.is_file() or checksum(archive) != SHA256:
        with urllib.request.urlopen(URL, timeout=90) as response, archive.open('wb') as file:
            while block := response.read(4 * 1024 * 1024):
                file.write(block)
        if checksum(archive) != SHA256:
            archive.unlink(missing_ok=True)
            raise ValueError('SHA-256 Ollama не совпала с опубликованной')
    destination = ROOT / '.tools/ollama'
    destination.mkdir(parents=True, exist_ok=True)
    with zstandard.open(archive, 'rb') as compressed, tarfile.open(fileobj=compressed, mode='r|') as tar:
        for member in tar:
            name = member.name
            if not name.startswith(('bin/', 'lib/ollama/')):
                continue
            if any(word in name.lower() for word in ['cuda', 'rocm', 'vulkan', 'cublas', 'cudart', 'nvrtc', 'nvidia']):
                continue
            tar.extract(member, destination, filter='data')
    print('Ollama', VERSION, 'установлена, официальная SHA-256 проверена:', destination)


if __name__ == '__main__':
    install()
