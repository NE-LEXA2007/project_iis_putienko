"""Получение настоящих моделей из официального каталога Docker Models.

Docker daemon не нужен: модели опубликованы как OCI-артефакты.
Digest манифеста и SHA-256 каждого файла проверяются. Веса остаются вне Git.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = 'https://registry-1.docker.io/v2/'


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        redirected = super().redirect_request(request, fp, code, message, headers, new_url)
        if redirected and urllib.parse.urlsplit(request.full_url).netloc != urllib.parse.urlsplit(new_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


OPENER = urllib.request.build_opener(SafeRedirect())


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return 'sha256:' + h.hexdigest()


def fetch_model(model):
    repository = model['repository']
    auth = 'https://auth.docker.io/token?service=registry.docker.io&scope=repository:' + repository + ':pull'
    with OPENER.open(auth, timeout=30) as response:
        token = json.load(response)['token']
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/vnd.oci.image.manifest.v1+json'}
    request = urllib.request.Request(REGISTRY + repository + '/manifests/' + model['manifest'], headers=headers)
    with OPENER.open(request, timeout=30) as response:
        raw = response.read()
    if 'sha256:' + hashlib.sha256(raw).hexdigest() != model['manifest']:
        raise ValueError('Digest манифеста не совпал')
    manifest = json.loads(raw)
    destination = ROOT / model['directory']
    destination.mkdir(parents=True, exist_ok=True)
    for layer in manifest['layers']:
        name = layer['annotations']['org.cncf.model.filepath']
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Небезопасное имя в артефакте')
        target = destination / relative
        if target.is_file() and target.stat().st_size == layer['size'] and digest_file(target) == layer['digest']:
            print(model['name'], name, 'проверен, уже есть', flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.part')
        request = urllib.request.Request(REGISTRY + repository + '/blobs/' + layer['digest'], headers=headers)
        downloaded = 0
        last = time.monotonic()
        with OPENER.open(request, timeout=90) as response, temporary.open('wb') as out:
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last > 20:
                    print(model['name'], name, downloaded // 1024 // 1024, 'MiB', flush=True)
                    last = time.monotonic()
        if downloaded != layer['size'] or digest_file(temporary) != layer['digest']:
            temporary.unlink(missing_ok=True)
            raise ValueError('Размер или SHA-256 файла не совпал: ' + name)
        temporary.replace(target)
        print(model['name'], name, 'SHA-256 OK', flush=True)
    if model['name'] == 'embedding':
        # BF16 соответствует dtype опубликованных весов и поддерживается CPU.
        runtime = {'max_seq_length': 512, 'model_args': {'torch_dtype': 'auto'}}
        (destination / 'sentence_bert_config.json').write_text(json.dumps(runtime, indent=2))
    (destination / 'oci-provenance.json').write_text(json.dumps(model, indent=2), encoding='utf-8')


if __name__ == '__main__':
    selected = sys.argv[1:]
    models = json.loads((ROOT / 'models/registry-lock.json').read_text())
    for model in models:
        if not selected or model['name'] in selected:
            fetch_model(model)
