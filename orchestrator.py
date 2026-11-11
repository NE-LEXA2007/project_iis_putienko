"""П3. FSM ролей и продуктовый контекст. Одна точка входа для веба и VK."""
import logging
import config
import re
import threading

import guardrails
import memory
import prompts
from contracts import Chunk
from llm_client import build_client
from mcp_client import MCPTools

log = logging.getLogger(__name__)
BLOCKED_REPLY = "Запрос отклонён слоем безопасности. Уберите персональные данные и переформулируйте вопрос по условиям кредита."
NO_ANSWER = "В базе знаний нет ответа на этот вопрос."
PRODUCT_QUESTION = "В проекте рассматривается обычный кредит наличными Т-Банка."
PRODUCT_SOURCES = {
    "cash": ["cash_conditions.pdf", "cash_tariff.pdf", "bank_conditions.pdf"],
}


def detect_product(text: str) -> str | None:
    text = text.casefold().replace("ё", "е")
    cash = bool(re.search(r"кредит\w*\s+наличн\w*|потребительск\w*(?:\s+кредит\w*)?|\bcash\b", text))
    card = bool(re.search(r"кредитн\w*\s+карт|платинум|\bcard\b", text))
    if cash and card:
        return "both"
    return "cash" if cash else "card" if card else None


class Orchestrator:
    ROLES = {
        "analyst": {"description": "объясняет опубликованное условие выбранного продукта", "temperature": 0.2, "filters": {}},
        "verifier": {"description": "сверяет конкретное условие по продукту и рубрике", "temperature": 0.1, "filters": {}},
        "editor": {"description": "пересказывает найденный ответ понятнее, не добавляя условий", "temperature": 0.3, "filters": {}},
    }
    INITIAL_ROLE = "analyst"
    FALLBACK_ROLE = "analyst"
    # Повторное редактирование запрещено: сначала заново сверяем факты аналитиком.
    TRANSITIONS = {"analyst": {"analyst", "verifier", "editor"},
                   "verifier": {"analyst", "verifier", "editor"},
                   "editor": {"analyst", "verifier"}}
    INTENT_TO_ROLE = {"question": "analyst", "verify": "verifier", "rewrite": "editor"}

    def __init__(self):
        self.llm = build_client()
        self.tools = MCPTools()
        memory.create_schema()
        self._state: dict[str, str] = {}
        self._context: dict[str, list[Chunk]] = {}
        self._products: dict[str, str] = {}
        self.usage: list[dict] = []
        # Streamlit держит одно ядро. Сериализация защищает его общий usage/LLM
        # и смену состояния от одновременных обработчиков двух вкладок.
        self._lock = threading.RLock()

    def detect_intent(self, text: str) -> str:
        text = text.casefold()
        if re.search(r"проще|простыми словами|перефраз|перепиши|короче|кратко", text):
            return "rewrite"
        if re.search(r"сверь|проверь|уточни|досроч|просроч|если|штраф|неустойк", text):
            return "verify"
        return "question"

    def route(self, session_id: str, intent: str) -> str:
        current = self._state.get(session_id, self.INITIAL_ROLE)
        target = self.INTENT_TO_ROLE.get(intent, self.FALLBACK_ROLE)
        if target not in self.TRANSITIONS.get(current, {self.FALLBACK_ROLE}):
            log.warning("Переход %s -> %s запрещён, откат в %s", current, target, self.FALLBACK_ROLE)
            target = self.FALLBACK_ROLE
        if target != current:
            log.info("Сессия %s: роль %s -> %s (интент %s)", session_id, current, target, intent)
        self._state[session_id] = target
        return target

    def handle(self, session_id: str, text: str) -> str:
        with self._lock:
            return self._handle(session_id, text)

    def _handle(self, session_id: str, text: str) -> str:
        verdict = guardrails.check(text)
        if not verdict.allowed:
            guardrails.log_attempt(session_id, text, verdict)
            log.warning("Запрос заблокирован: %s", verdict.reason)
            return BLOCKED_REPLY
        if re.search(r"ипотек|автокредит|скоринг|одобрят|сбер|втб|альфа.?банк", text, re.I):
            return NO_ANSWER + " Другие банки, ипотека и одобрение вне проекта."

        previous_product = self._products.get(session_id) or memory.get_product(session_id)
        product = detect_product(text)
        if product in {"card", "both"}:
            return NO_ANSWER + " Кредитные карты вне выбранного продукта."
        history = memory.window(session_id)
        if product is None:
            product = previous_product
            if product is None:
                for message in reversed(history):
                    if message.role == "user" and detect_product(message.content) in PRODUCT_SOURCES:
                        product = detect_product(message.content)
                        break
        if product is None:
            product = "cash"

        changed_product = previous_product not in {None, product}
        if changed_product:
            # Не переносим в новый продукт ни найденные пункты, ни старый диалог.
            self._context.pop(session_id, None)
            self._state.pop(session_id, None)
            history = []
            memory.clear(session_id)
        self._products[session_id] = product
        memory.set_product(session_id, product)
        previous_role = self._state.get(session_id, self.INITIAL_ROLE)
        role = self.route(session_id, self.detect_intent(text))
        role_config = self.ROLES[role]
        filters = {"source": {"$in": PRODUCT_SOURCES[product]}, "product": "cash"}
        # У сверщика второй уровень иерархии. Отсутствие категории не подменяет
        # текст другого продукта: source-фильтр обязателен для любой роли.
        if role == "verifier":
            categories = [name for name, pattern in [
                ("early_repayment", r"досроч"),
                ("penalties", r"просроч|неустойк|штраф|пеня"),
                ("rates", r"процент|ставк|полная стоимость|пск"),
                ("terms", r"срок|дата платеж|обязательн.*платеж"),
            ] if re.search(pattern, text, re.I)]
            # Вопрос может соединять два раздела. Тогда оставляем фильтр продукта,
            # чтобы не потерять ставку из-за наличия слова «досрочно».
            if len(categories) == 1:
                filters["category"] = categories[0]
        query = text
        if re.search(r"больше|сверх", text, re.I) and re.search(r"запла|внес|плат", text, re.I):
            query += " Частичное досрочное погашение, уменьшение срока или платежа, уведомление банка."
        if role == "editor" and self._context.get(session_id):
            chunks = list(self._context[session_id])
        else:
            chunks = self.tools.search_filtered(query, filters)
            if role == "verifier":
                # Общая процедура и тариф дополняют друг друга. Рубрика может
                # быть смешанной, поэтому тариф проверяем отдельным MCP-запросом.
                tariff = self.tools.search_filtered(query, {"source": "cash_tariff.pdf", "product": "cash"}, k=8)
                known = {(c.source, c.page, c.text) for c in chunks}
                chunks = [c for c in tariff if (c.source, c.page, c.text) not in known] + chunks
        if role != previous_role and not changed_product:
            known = {(c.source, c.page, c.text) for c in chunks}
            chunks += [c for c in self._context.get(session_id, [])
                       if c.source in PRODUCT_SOURCES[product] and (c.source, c.page, c.text) not in known]
        chunks = chunks[:config.TOP_K * 2]
        self._context[session_id] = chunks
        if not chunks:
            return NO_ANSWER + " Для выбранного продукта подходящих фрагментов не найдено."
        messages = prompts.build_messages(role, text, chunks, history)
        prompt_size = self.llm.count_tokens("\n".join(m["content"] for m in messages))
        log.info("Продукт %s; роль %s; чанков %d; вход ~%d токенов", product, role, len(chunks), prompt_size)
        answer = self.llm.generate(messages, temperature=role_config["temperature"])
        if config.LLM_BACKEND != "fake":
            answer = prompts.finish_answer(answer, chunks)
        self.usage.append({"session_id": session_id, "role": role,
                           "prompt_tokens": prompt_size,
                           "completion_tokens": self.llm.count_tokens(answer)})
        # Сохраняем пару только после удачного вызова. Ошибка сети не оставляет
        # в истории одинокое сообщение, которое задублируется при повторе.
        memory.save(session_id, "user", text)
        memory.save(session_id, "assistant", answer)
        return answer

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._state.pop(session_id, None)
            self._context.pop(session_id, None)
            self._products.pop(session_id, None)
            memory.clear(session_id)

    def close(self) -> None:
        self.tools.close()
