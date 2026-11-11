"""П4–П5: настоящие ответы, подмена контекста и память между процессами.

Запускается отдельно от быстрых unit-тестов; требует индекса и Ollama.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from contracts import Chunk
from llm_client import build_client
from orchestrator import Orchestrator
from prompts import build_messages


def normalize(text):
    return text.casefold().replace('ё', 'е').replace('.', ',')


def check_case(case_id, answer):
    text = normalize(answer)
    if case_id == 1:
        return '19,878' in text and '39,001' in text
    if case_id == 2:
        return ('30' in text or 'тридцат' in text) and ('соглас' in text or 'уменьш' in text)
    if case_id == 3:
        return '0,1' in text and any(s in text for s in ['ежеднев', 'каждый день', 'в день'])
    if case_id == 4:
        return 'досроч' in text and any(s in text for s in ['уведом', 'извест', 'заявк']) and 'срок' in text and 'платеж' in text and any(s in text for s in ['уменьш', 'измен'])
    return 'нет ответа' in text


def main():
    if config.LLM_BACKEND == 'fake':
        raise ValueError('Требуется настоящая модель')
    output = ROOT / 'reports/live-checks.json'
    cases = json.loads((ROOT / 'gold_dataset.json').read_text())
    rows = []
    core = Orchestrator()
    try:
        for case in cases:
            sid = 'live:' + uuid.uuid4().hex
            before = len(core.usage)
            answer = core.handle(sid, case['question'])
            generated = len(core.usage) > before
            row = {'id': case['id'], 'question': case['question'], 'answer': answer,
                   'chunks': [asdict(c) for c in core._context.get(sid, [])],
                   'provider_usage': dict(core.llm.last_usage) if generated else None,
                   'passed': check_case(case['id'], answer)}
            if generated and not row['provider_usage']:
                raise ValueError('Модель не вернула настоящие счётчики')
            rows.append(row)
            core.reset(sid)
            print('CASE',case['id'],'PASS' if row['passed'] else 'FAIL', answer, flush=True)
    finally:
        core.close()
    client = build_client()
    controls = []
    for amount in [17,29]:
        chunk = Chunk(f'Кредит наличными: максимальная сумма кредита составляет {amount} рублей.',
                      'cash_tariff.pdf', 1, 'rates')
        answer = client.generate(build_messages('analyst','Какая максимальная сумма кредита указана в приведённом фрагменте тарифа?', [chunk], []),temperature=0.1)
        passed = str(amount) in answer
        controls.append({'amount': amount, 'answer': answer, 'provider_usage': client.last_usage, 'passed': passed})
        print('CONTROL',amount,passed,answer,flush=True)
    sid = 'memory-live:' + uuid.uuid4().hex
    dialog = []
    core = Orchestrator()
    try:
        for question in ['Какой диапазон ПСК указан в тарифе КН 5.0?', 'Объясни этот ответ простыми словами.']:
            answer = core.handle(sid,question)
            dialog.append({'question':question,'answer':answer,'role':core._state[sid],'provider_usage':core.llm.last_usage})
            print('DIALOG',len(dialog),answer,flush=True)
    finally:
        core.close()
    # Третью реплику обрабатывает новый Python-процесс, который читает SQLite.
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--continue', sid],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
    if child.returncode:
        raise RuntimeError('Продолжение диалога завершилось ошибкой: '+child.stderr[-700:])
    continuation = json.loads(child.stdout)
    dialog.append(continuation)
    print('DIALOG 3 AFTER RESTART',continuation['answer'],flush=True)
    memory_ok = '19,878' in normalize(continuation['answer']) and '39,001' in normalize(continuation['answer'])
    result={'timestamp':datetime.now(timezone.utc).isoformat(),'backend':config.LLM_BACKEND,'model':config.OLLAMA_MODEL,'cases':rows,'context_controls':controls,
            'dialog':dialog,'memory_after_restart':memory_ok,
            'all_passed':all(r['passed'] for r in rows+controls) and memory_ok}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if not result['all_passed']:
        raise AssertionError('Есть неподтверждённые сценарии; подробности в reports/live-checks.json')
    print('ALL LIVE CHECKS PASSED',flush=True)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--continue':
        sid=sys.argv[2]
        core=Orchestrator()
        try:
            q='Повтори обе границы диапазона, о котором я спросил в первой реплике.'
            a=core.handle(sid,q)
            print(json.dumps({'question':q,'answer':a,'role':core._state[sid],'provider_usage':core.llm.last_usage},ensure_ascii=False))
            core.reset(sid)
        finally:core.close()
    else:main()
