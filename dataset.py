"""Общая проверка Gold Dataset: черновые ответы нельзя использовать для метрик."""
import json
from pathlib import Path


def load_cases(path: Path, *, count: int | None = None) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Gold Dataset должен быть непустым списком")
    if count is not None and len(cases) != count:
        raise ValueError(f"Нужно {count} вопросов, получено {len(cases)}")
    ids = set()
    for case in cases:
        if case["id"] in ids:
            raise ValueError("Повторяющиеся id в Gold Dataset")
        ids.add(case["id"])
        if not case.get("verified") or not case.get("expected"):
            raise ValueError(f"Вопрос {case['id']}: эталон ещё не сверен с документом")
        if not isinstance(case.get("expected_sources"), list) or not case.get("question", "").strip():
            raise ValueError(f"Вопрос {case['id']}: неверная схема")
        if case["type"] != "out_of_scope" and not case["expected_sources"]:
            raise ValueError(f"Вопрос {case['id']}: не указаны источники")
    return cases
