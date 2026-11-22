# Модель угроз ЛР12 (вариант 1 — SOC)

## Активы
1. Карточки инцидентов (конфиденциальность, изоляция tenant).
2. Права пользователя: role, tenant (целостность сессии).
3. Инструменты агента и параметры вызовов.
4. Бюджет вызовов / доступность.
5. Служебные подсказки и ключи (не должны попадать в ответы/логи).

## Границы доверия

| Источник | Статус | Правило |
|-|-|-|
| Сессия (role, tenant, approved) | Доверенный | Задаёт сервер |
| REGISTRY / PERMISSIONS | Доверенный | Конфигурация приложения |
| RAG / KB / ответ инструмента | Недоверенные **данные** | Не инструкции |
| tool_call от LLM | Недоверенное **предложение** | Только после `execute()` |
| Результат `execute` | Контролируемый | Фильтр по правам |

## Угрозы (не менее 5)

| # | Угроза | Атакующий / источник | Нарушение | Контроль | Тест |
|-|-|-|-|-|-|
| 1 | Косвенный prompt injection | KB-02 в RAG | Вызов send_report | TOOL_NOT_ALLOWED / роль | T3, T9 |
| 2 | Cross-tenant чтение | analyst A → INC-202 | Утечка tenant B | TENANT_DENIED | T2 |
| 3 | Cross-tenant через KB | search KB-03 | Утечка регламента B | фильтр tenant до выдачи | T10 |
| 4 | Подмена роли текстом | «ты supervisor» в KB | Повышение привилегий | роль только из Context | T9 |
| 5 | Excess agency | run_shell / delete | Несанкционированное действие | allowlist | T8 |
| 6 | Отправка без согласования | supervisor, approved=false | Нежелательная «отправка» | HUMAN_APPROVAL_REQUIRED | T6 |
| 7 | Неверный recipient | attacker@evil | Утечка наружу | whitelist | (recipient test) |
| 8 | Unbounded consumption | 20× вызов | DoS / перерасход | remaining_calls, audit | T5, T11 |
| 9 | Невалидные аргументы | id=list | Обход / сбой | схема + fullmatch | T4 |
| 10 | Прямая инъекция в user | «вызови send_report» | Тот же барьер | execute, не промпт | T3 |

## OWASP LLM Top 10 (ориентиры 2025/2026)
- LLM01 Prompt Injection — T3, T9  
- LLM02 Sensitive Information Disclosure — T2, T10  
- LLM03 Excessive Agency — T6, T7, T8  
- LLM06 Unbounded Consumption — T5, T11  
- Improper Output Handling — текст KB не передаётся в shell/SQL  

## Остаточные риски
- Модель может **предложить** опасный вызов и **пересказать** вредоносный текст пользователю (ретрансляция, как в ЛР10/11).
- Нулевой ASR на детерминированном стенде не доказывает устойчивость конкретной LLM.
- Нет проверки перефразированных атак и multi-turn social engineering.
