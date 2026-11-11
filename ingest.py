"""
ingest.py — сбор исходных данных и наполнение векторного индекса.

ПРАКТИКА 2. Что делает студент:
  1. clean_text()  — предварительная очистка: переносы, колонтитулы, мусорные пробелы
  2. chunk_text()  — нарезка на чанки фиксированного размера с перекрытием
  3. detect_category() — рубрикация чанка; словарь рубрик задаёте вы под свой домен

  4. черновик gold_dataset.json — 5 вопросов с указанием файлов-источников

Готово и трогать не нужно: обход data/raw/, чтение PDF и TXT, запись в ChromaDB,
замер hit-rate в evaluate_retrieval().

Каждый запуск пересобирает индекс с нуля — коллекция очищается перед наполнением.
Поэтому при подборе параметров достаточно поправить CHUNK_SIZE или CHUNK_OVERLAP
в config.py и запустить ingest.py снова: удалять data/chroma/ руками не нужно.

Критерии приёмки:
  - python ingest.py отрабатывает без ошибок, коллекция непуста
  - у каждого чанка заполнены source, page и category; пустых category нет
  - поиск с фильтром по category возвращает чанки только этой категории
  - в gold_dataset.json не меньше 5 вопросов, у каждого заполнено expected_sources
  - показаны не менее двух замеров с разными CHUNK_SIZE / CHUNK_OVERLAP,
    и выбор итоговых значений объяснён числом, а не вкусом
  - CHUNK_SIZE и CHUNK_OVERLAP берутся из config.py, а не зашиты в код

Про метрику. hit-rate@k отвечает на один вопрос: попал ли фрагмент с ответом
в первые k результатов. Считается механически, сверкой поля source с ожидаемым —
ни модели, ни судьи для этого не нужно, поэтому мерить можно уже здесь, на П2.
Не путайте её с Context Precision из триады RAG: та оценивает, насколько найденное
относится к делу, её считает LLM-судья, и появляется она только на П6. В плане
дисциплины обе назывались «точностью извлечения контекста», хотя это разные вещи.

Запуск:
    python ingest.py          # проиндексировать и сразу замерить
    python ingest.py --eval   # только замер, без переиндексации
"""

import hashlib
from pathlib import Path

import config


# --- Читатели форматов: готовы ---

def read_txt(path: Path) -> list[tuple[int, str]]:
    """Возвращает [(номер страницы, текст)]. У TXT страниц нет, поэтому страница 0."""
    return [(0, path.read_text(encoding="utf-8", errors="ignore"))]


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """Страница сохраняет исходный номер PDF. Удаляем только известный колонтитул."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    result = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # pypdf склеивает «1 из 3» и строку таблицы «1.» в «1 из 31.».
        # Поэтому номер удаляется по известному числу страниц, без жадного \d+.
        text = re.sub(r"^\s*" + str(i + 1) + r"\s+из\s+" + str(len(reader.pages)), "", text)
        result.append((i + 1, text))
    return result


READERS = {".txt": read_txt, ".md": read_txt, ".pdf": read_pdf}


# --- Ваша часть (П2) ---

import re
import unicodedata


def clean_text(text: str) -> str:
    """Очистка без потери номеров пунктов, единиц измерения и границ абзацев.

    Колонтитулы удаляются только по явным маркерам. Нельзя безусловно удалять
    любую повторяющуюся строку: ею может оказаться значимый пункт договора.
    """
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"(?<=[а-яёa-z])-\n\s*(?=[а-яёa-z])", "", text)
    text = re.sub(r"-\n\s*(?=[А-ЯЁA-Z])", "-", text)
    text = re.sub(r"(?:[TТ]-КН-\d+\.\d+-\d{6}|ФЗ-КН-\d{6})", "", text)
    paragraphs, current = [], []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if re.fullmatch(r"(?:стр(?:аница)?\.?\s*)\d+(?:\s*(?:из|/)\s*\d+)?", line, re.I):
            continue
        if re.fullmatch(r"(?:Т-Банк|АО «ТБанк»|www\.tbank\.ru)", line, re.I):
            continue
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if re.match(r"(?:\d+(?:\.\d+)*[.)]\s|Раздел\s+\d|Глава\s+\d|#{1,6}\s)", line, re.I) and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Окна с шагом size-overlap; по возможности граница между пунктами.

    Если размер пункта больше окна, режем его по словам. Реальное перекрытие
    может быть короче заданного на длину обрезанного слова.
    """
    if size < 32 or overlap < 0 or overlap >= size:
        raise ValueError("Нужно size >= 32 и 0 <= overlap < size")
    if not text.strip():
        return []
    result, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            paragraph = text.rfind("\n\n", start + size // 2, end)
            word = text.rfind(" ", start + size // 2, end)
            boundary = paragraph if paragraph != -1 else word
            if boundary > start + overlap:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
        if end == len(text):
            break
        next_start = end - overlap
        # Не начинать посреди слова: такой хвост мало полезен эмбеддингу.
        if next_start > start and next_start > 0 and not text[next_start - 1].isspace():
            space = re.search(r"\s", text[next_start:end])
            if space:
                next_start += space.end()
        start = max(start + 1, next_start)
    return result


def detect_category(text: str, source: str) -> str:
    """Второй уровень: продукт определяется именем source, рубрика — пунктом."""
    combined = (source + " " + text).lower().replace("ё", "е")
    rules = [
        ("early_repayment", r"досроч|early_repayment"),
        ("penalties", r"просроч|неустойк|штраф|пен[яию]|penalties"),
        ("rates", r"процент|ставк|полная стоимость|пск|rates|тариф"),
        ("terms", r"срок|дата платеж|платежн|платежей|terms"),
    ]
    return next((name for name, pattern in rules if re.search(pattern, combined)), "general")


# --- Сборка индекса: готова ---

def ingest() -> int:
    """Обходит исходные данные, нарезает и складывает в ChromaDB. Возвращает число чанков.

    Индекс собирается заново с нуля: коллекция очищается перед наполнением.
    Иначе после смены CHUNK_SIZE в коллекции остались бы чанки предыдущей
    нарезки, и hit-rate@5 замерялся бы по смеси двух — молча, без ошибки.
    """
    from scripts.check_sources import validate_sources
    errors = validate_sources()
    if errors:
        raise SystemExit("Корпус не готов:\n" + "\n".join(errors))
    if not Path(config.EMBEDDING_MODEL).is_dir():
        raise SystemExit("Нет модели эмбеддингов. Выполните python scripts/fetch_models.py embedding")
    collection = config.reset_collection()
    total = 0
    import json
    manifest = json.loads((config.BASE_DIR / "sources.json").read_text(encoding="utf-8"))
    ids, documents, metadatas = [], [], []
    seen = set()

    def flush():
        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            ids.clear(); documents.clear(); metadatas.clear()

    for source in manifest:
        path = config.RAW_DIR / source["file"]
        ranges = source.get("index_pages", [])
        for page, raw in READERS[path.suffix.lower()](path):
            if ranges and not any(start <= page <= end for start, end in ranges):
                continue
            cleaned = clean_text(raw)
            if not cleaned:
                continue
            part = "общие положения"
            if path.name == "bank_conditions.pdf" and page >= 40:
                part = "общие условия кредитования, редакция 31"
            # Специальные условия недвижимости, кредитной линии и образования
            # храним в индексе с другим продуктом; оркестратор их не выбирает.
            product = "other" if path.name == "bank_conditions.pdf" and page >= 45 else "cash"
            for body in chunk_text(cleaned):
                chunk_product = product
                if path.name == "bank_conditions.pdf" and re.search(
                        r"образовательн|недвижимост|автомобил|предмет\w* залог|партнерск", body, re.I):
                    chunk_product = "other"
                chunk = f"Корпус: кредит наличными. {source['title']}. {part}.\n{body}"
                chunk_id = hashlib.sha1(f"{path.name}:{page}:{body}".encode()).hexdigest()
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                ids.append(chunk_id); documents.append(chunk)
                metadatas.append({"source": path.name, "page": page,
                                  "category": detect_category(body, path.name), "product": chunk_product})
                total += 1
                if len(ids) >= 8:
                    flush()
        print(f"Обработан {path.name}; чанков всего: {total}", flush=True)
    flush()

    return total


# --- Замер качества поиска: готов ---

def evaluate_retrieval(k: int = config.TOP_K) -> float:
    """hit-rate@k по черновику Gold Dataset. Готово, менять не нужно.

    Для каждого вопроса выполняется поиск и проверяется, попал ли в первые k
    результатов хотя бы один чанк из ожидаемых источников.

    Вопросы с пустым expected_sources пропускаются: это проверка отказа отвечать,
    а она требует LLM и меряется на П6, а не здесь.

    Обращается к коллекции напрямую, а не через MCP: на П2 оркестратора ещё нет.
    """
    import json

    dataset_path = Path(__file__).parent / "gold_dataset.json"
    from dataset import load_cases
    cases = [c for c in load_cases(dataset_path)
             if c.get("expected_sources")]

    if not cases:
        print("В gold_dataset.json нет вопросов с expected_sources — замерять нечего (П2).")
        return 0.0

    collection = config.get_collection()
    hits = 0

    print(f"\nhit-rate@{k}  |  чанк {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP}")
    for case in cases:
        result = collection.query(query_texts=[case["question"]], n_results=k)
        found = {(m or {}).get("source", "") for m in (result.get("metadatas") or [[]])[0]}
        hit = bool(found & set(case["expected_sources"]))
        hits += hit
        print(f"  [{'+' if hit else '-'}] {case['question'][:70]}")
        if not hit:
            print(f"      ожидалось: {case['expected_sources']}, найдено: {sorted(found)}")

    rate = hits / len(cases)
    print(f"\nhit-rate@{k} = {rate:.0%}  ({hits} из {len(cases)})")
    skipped = len(json.loads(dataset_path.read_text(encoding="utf-8"))) - len(cases)
    if skipped:
        print(f"Пропущено вопросов без expected_sources: {skipped} — они меряются на П6.")
    return rate


if __name__ == "__main__":
    import sys

    if "--eval" not in sys.argv:
        count = ingest()
        print(f"Индекс пересобран заново. Проиндексировано чанков: {count}")
        print(f"Всего в коллекции: {config.get_collection().count()}")

    evaluate_retrieval()
