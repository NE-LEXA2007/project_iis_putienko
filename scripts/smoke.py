"""Реальный сквозной запрос. Ошибка зависимости => exit 1, fake => exit 2."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from llm_client import LLMError
from orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', nargs='?', default='Кредит наличными: можно ли погасить его досрочно?')
    args = parser.parse_args()
    if config.LLM_BACKEND == 'fake':
        print('Это проверка настоящей LLM, fake не допускается.', file=sys.stderr)
        return 2
    core = None
    try:
        from scripts.check_sources import validate_sources
        errors = validate_sources()
        if errors:
            raise ValueError('Корпус не готов: ' + '; '.join(errors))
        core = Orchestrator()
        answer = core.handle('smoke', args.question)
        print(answer)
        if not core.usage:
            raise ValueError('Ответ не использовал модель: проверьте продукт, документы и вопрос')
        return 0
    except (LLMError, ValueError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if core is not None:
            core.reset('smoke')
            core.close()


if __name__ == '__main__':
    sys.exit(main())
