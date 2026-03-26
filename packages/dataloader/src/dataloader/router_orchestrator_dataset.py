"""History-aware seed dataset generator for router/orchestrator training.

The generator supports:
- current production agents
- an expanded catalog with future agents
- fully custom agents loaded from JSON, even if they do not exist in code yet

Commands:
    write-assets  Write prompt text, pretty JSON scenarios and SFT JSONL rows
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Final, Sequence, TypeVar

import yaml

T = TypeVar("T")


OUTPUT_DIR_NAME: Final[str] = "router_orchestrator"
DEFAULT_RANDOM_SEED: Final[int] = 17
DEFAULT_SAMPLES_PER_TEMPLATE: Final[int] = 5
DEFAULT_CATALOG_MODE: Final[str] = "current"
DEFAULT_BASIC_SAMPLES: Final[int] = 400
DEFAULT_REASONING_SAMPLES: Final[int] = 250

ROUTES: Final[tuple[str, ...]] = (
    "manage_notes",
    "personalize",
    "discovery_notes",
    "sage",
)

MESSAGE_PREFIXES: Final[tuple[str, ...]] = (
    "",
    "Hej, ",
    "Krótko: ",
    "Potrzebuję pomocy: ",
    "Proszę, ",
    "Mam prośbę: ",
)
MESSAGE_SUFFIXES: Final[tuple[str, ...]] = (
    "",
    " Proszę.",
    " Dzięki.",
    " To pilne.",
)
CORRECTION_PREFIXES: Final[tuple[str, ...]] = (
    "Nie, chodzi mi raczej o: ",
    "Poprawka, teraz chcę: ",
    "Zmiana planu. Zrób tak: ",
    "Jednak chodziło mi o to: ",
)
RETURN_PREFIXES: Final[tuple[str, ...]] = (
    "Wróćmy do poprzedniego tematu. ",
    "Jeszcze wracając do wcześniejszej sprawy: ",
    "Nie o to mi teraz chodzi, wróćmy do tamtego wątku. ",
)
NOTE_NAMES: Final[tuple[str, ...]] = (
    "Zakupy",
    "Plan weekendu",
    "Projekt Atlas",
    "Dom rodzinny",
    "Kredyt hipoteczny",
    "Notatki o RAG",
    "Strategia Q2",
)
CONFIG_VARIANTS: Final[tuple[str, ...]] = ("basic", "reasoning")


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    routing_hints: tuple[str, ...]
    direct_requests: tuple[str, ...]
    follow_ups: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "routing_hints": list(self.routing_hints),
            "direct_requests": list(self.direct_requests),
            "follow_ups": list(self.follow_ups),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AgentSpec":
        name = _require_non_empty_string(payload.get("name"), field_name="name")
        description = _require_non_empty_string(
            payload.get("description"),
            field_name=f"{name}.description",
        )
        routing_hints = _require_non_empty_strings(
            payload.get("routing_hints"),
            field_name=f"{name}.routing_hints",
        )
        direct_requests = _require_non_empty_strings(
            payload.get("direct_requests"),
            field_name=f"{name}.direct_requests",
        )
        follow_ups = _require_non_empty_strings(
            payload.get("follow_ups"),
            field_name=f"{name}.follow_ups",
        )
        return cls(
            name=name,
            description=description,
            routing_hints=tuple(routing_hints),
            direct_requests=tuple(direct_requests),
            follow_ups=tuple(follow_ups),
        )


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str

    def render(self) -> str:
        return f"<start_of_turn>{self.role}\n{self.content}\n<end_of_turn>"

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class RouterScenario:
    name: str
    history: tuple[ConversationTurn, ...]
    current_message: str
    expected_agent: str
    tags: tuple[str, ...] = ()

    def render_user_input(self) -> str:
        turns = [*self.history, ConversationTurn(role="user", content=self.current_message)]
        return "\n".join(turn.render() for turn in turns)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "history": [turn.as_dict() for turn in self.history],
            "current_message": self.current_message,
            "rendered_input": self.render_user_input(),
            "expected_agent": self.expected_agent,
            "tags": list(self.tags),
        }

    def as_jsonl_row(
        self,
        prompt_text: str,
        *,
        allowed_agents: Sequence[str],
    ) -> dict[str, object]:
        return {
            "messages": [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": self.render_user_input()},
                {"role": "assistant", "content": self.expected_agent},
            ],
            "metadata": {
                "scenario_name": self.name,
                "expected_agent": self.expected_agent,
                "allowed_agents": list(allowed_agents),
                "history_turn_count": len(self.history),
                "tags": list(self.tags),
                "source": "evaluation-inspired-synthetic-router",
            },
        }


CURRENT_AGENT_SPECS: Final[tuple[AgentSpec, ...]] = (
    AgentSpec(
        name="manage_notes",
        description="dodawanie, edycja, odczyt i listowanie notatek",
        routing_hints=(
            "tworzenia notatek",
            "edycji notatek",
            "odczytu notatek",
            "listowania notatek",
        ),
        direct_requests=(
            "Dodaj notatkę o nazwie Zakupy z treścią mleko i chleb.",
            "Odczytaj notatkę Projekt Atlas.",
            "Pokaż wszystkie notatki.",
            "Edytuj notatkę Plan weekendu i dopisz kino w sobotę.",
        ),
        follow_ups=(
            "Dopisz jeszcze jajka.",
            "Pokaż tę notatkę.",
            "Wyświetl ostatnią notatkę.",
            "Edytuj ją i dodaj porównanie kosztów.",
        ),
    ),
    AgentSpec(
        name="personalize",
        description="onboarding, ustawienia użytkownika, imię, vault i konfiguracja",
        routing_hints=(
            "imienia użytkownika",
            "vaulta",
            "ustawień użytkownika",
            "konfiguracji",
        ),
        direct_requests=(
            "Mam na imię Ania.",
            "Mój vault to SecondBrain.",
            "Ustaw moje imię na Mateusz.",
            "Chcę zmienić nazwę vaulta na Atlas.",
        ),
        follow_ups=(
            "Jednak zmień moje imię na Ola.",
            "Ustaw vault na KnowledgeHub.",
            "Będę używać vaulta Dom.",
            "Popraw tylko nazwę vaulta.",
        ),
    ),
    AgentSpec(
        name="discovery_notes",
        description="semantyczne wyszukiwanie wiedzy i znajdowanie informacji w notatkach",
        routing_hints=(
            "wyszukiwania w notatkach",
            "znajdowania informacji",
            "przeszukania bazy wiedzy",
            "semantycznego wyszukiwania",
        ),
        direct_requests=(
            "Znajdź w moich notatkach informacje o RAG.",
            "Wyszukaj notatki o kredycie hipotecznym.",
            "Szukaj wiedzy o linkowaniu notatek w Obsidianie.",
            "Znajdź informacje o embeddings i chunkingu.",
        ),
        follow_ups=(
            "Poszukaj też czegoś o chunkingu dokumentów.",
            "A teraz sprawdź też embeddings.",
            "Szukaj dalej w tym temacie.",
            "Dodaj też wyniki o analizie SWOT.",
        ),
    ),
    AgentSpec(
        name="sage",
        description="wsparcie decyzji, porównanie opcji, rekomendacje, ryzyka i plan działania",
        routing_hints=(
            "podejmowania decyzji",
            "porównania opcji",
            "rekomendacji",
            "analizy ryzyk",
        ),
        direct_requests=(
            "Nie wiem czy kupić dom czy mieszkanie, pomóż mi podjąć decyzję.",
            "Porównaj leasing i zakup auta.",
            "Pomóż mi zdecydować, czy zatrudnić freelancera, czy budować zespół wewnętrzny.",
            "Jaką strategię wejścia na rynek wybrać dla nowego produktu?",
        ),
        follow_ups=(
            "Uwzględnij jeszcze koszty początkowe.",
            "A co z wpływem na cash flow?",
            "Porównaj to też pod kątem ryzyka.",
            "Weź jeszcze pod uwagę czas wdrożenia.",
        ),
    ),
)

FUTURE_AGENT_SPECS: Final[tuple[AgentSpec, ...]] = (
    AgentSpec(
        name="calendar",
        description="kalendarz, spotkania, terminy i plan dnia",
        routing_hints=(
            "spotkań",
            "terminów",
            "kalendarza",
            "planu dnia",
        ),
        direct_requests=(
            "Umów spotkanie z zespołem na jutro o 10:00.",
            "Przełóż call z klientem na piątek.",
            "Sprawdź wolne sloty na przyszły tydzień.",
            "Zablokuj mi dwie godziny na pracę głęboką w środę.",
        ),
        follow_ups=(
            "Przesuń to na 14:00.",
            "Dodaj 30 minut buforu.",
            "Skróć to spotkanie do 25 minut.",
            "Pokaż mi alternatywy rano.",
        ),
    ),
    AgentSpec(
        name="tasks",
        description="zadania, checklisty, priorytety i plan wykonania",
        routing_hints=(
            "zadań",
            "todo",
            "priorytetów",
            "checklist",
        ),
        direct_requests=(
            "Dodaj zadanie: wysłać ofertę do klienta.",
            "Ułóż priorytety na dziś.",
            "Rozbij wdrożenie CRM na checklistę.",
            "Pokaż moje zadania na ten tydzień.",
        ),
        follow_ups=(
            "Dodaj do tego termin na czwartek.",
            "Podnieś priorytet tego zadania.",
            "Rozbij to jeszcze na mniejsze kroki.",
            "Oznacz to jako zrobione.",
        ),
    ),
    AgentSpec(
        name="writer",
        description="pisanie, redakcja, streszczenia i przeredagowanie tekstów",
        routing_hints=(
            "pisania tekstu",
            "redakcji",
            "maili",
            "streszczeń",
        ),
        direct_requests=(
            "Napisz mail do klienta z podsumowaniem spotkania.",
            "Przeredaguj opis oferty, żeby był bardziej konkretny.",
            "Skróć ten komunikat do 3 zdań.",
            "Napisz szkic posta o wdrożeniu AI w firmie.",
        ),
        follow_ups=(
            "Zrób to bardziej formalnie.",
            "Dodaj krótkie CTA.",
            "Skróć to jeszcze o połowę.",
            "Nadaj temu bardziej spokojny ton.",
        ),
    ),
    AgentSpec(
        name="researcher",
        description="research, zbieranie źródeł, benchmarki i porównanie informacji zewnętrznych",
        routing_hints=(
            "researchu",
            "źródeł",
            "benchmarków",
            "analizy rynku",
        ),
        direct_requests=(
            "Zbierz benchmark narzędzi do monitoringu aplikacji.",
            "Przygotuj research rynku coworków w Warszawie.",
            "Znajdź źródła o trendach w agentach AI.",
            "Porównaj trzy platformy do newsletterów.",
        ),
        follow_ups=(
            "Dodaj do tego 3 wiarygodne źródła.",
            "Zawęź to do rynku polskiego.",
            "Pokaż też różnice cenowe.",
            "Dodaj krótkie podsumowanie wniosków.",
        ),
    ),
    AgentSpec(
        name="finance",
        description="budżet, koszty, cash flow i scenariusze finansowe",
        routing_hints=(
            "budżetu",
            "kosztów",
            "cash flow",
            "analizy finansowej",
        ),
        direct_requests=(
            "Policz budżet projektu na 3 miesiące.",
            "Porównaj koszty etatu i freelancera.",
            "Przygotuj prostą analizę cash flow dla nowej usługi.",
            "Oszacuj opłacalność kampanii przy budżecie 20 tysięcy.",
        ),
        follow_ups=(
            "Uwzględnij VAT.",
            "Dodaj scenariusz pesymistyczny.",
            "Pokaż też wariant ostrożny.",
            "Rozpisz to miesięcznie.",
        ),
    ),
    AgentSpec(
        name="coding",
        description="kod, debugowanie, testy i refaktoryzacja",
        routing_hints=(
            "kodu",
            "bugów",
            "testów",
            "refaktoryzacji",
        ),
        direct_requests=(
            "Napraw błąd w endpointcie logowania.",
            "Napisz testy do parsera CSV.",
            "Zrefaktoruj ten moduł autoryzacji.",
            "Znajdź przyczynę regresji po ostatnim deployu.",
        ),
        follow_ups=(
            "Dodaj też test regresyjny.",
            "Pokaż minimalny patch.",
            "Zostaw komentarz przy trudniejszym fragmencie.",
            "Sprowadź to do mniejszej zmiany.",
        ),
    ),
)


def _package_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_output_dir() -> Path:
    return _package_dir() / "data" / OUTPUT_DIR_NAME


def _default_evaluation_dir() -> Path:
    return _package_dir().parent / "evaluation" / "src" / "evaluation"


def _choose(rng: random.Random, values: Sequence[T]) -> T:
    if not values:
        raise ValueError("Expected at least one value to choose from.")
    return values[rng.randrange(len(values))]


def _different_choice(
    rng: random.Random,
    values: Sequence[T],
    current: T,
) -> T:
    filtered = [value for value in values if value != current]
    return _choose(rng, filtered or values)


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field `{field_name}` must be a non-empty string.")
    return value.strip()


def _require_non_empty_strings(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Field `{field_name}` must be a non-empty list of strings.")
    items: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Field `{field_name}` can contain only non-empty strings.")
        items.append(raw.strip())
    return items


def _style_message(rng: random.Random, message: str) -> str:
    prefix = _choose(rng, MESSAGE_PREFIXES)
    suffix = _choose(rng, MESSAGE_SUFFIXES)
    styled = f"{prefix}{message}".strip()
    if suffix:
        styled = f"{styled}{suffix}"
    return styled


def _unique_specs(specs: Sequence[AgentSpec]) -> tuple[AgentSpec, ...]:
    ordered: list[AgentSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            for index, current in enumerate(ordered):
                if current.name == spec.name:
                    ordered[index] = spec
                    break
            continue
        seen.add(spec.name)
        ordered.append(spec)
    return tuple(ordered)


def _load_custom_agent_specs(agent_spec_path: Path | None) -> tuple[AgentSpec, ...]:
    if agent_spec_path is None:
        return ()
    payload = json.loads(agent_spec_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_agents = payload.get("agents")
        if not isinstance(raw_agents, list):
            raise ValueError("Custom agent spec JSON object must contain `agents` list.")
        payload = raw_agents
    if not isinstance(payload, list):
        raise ValueError("Custom agent spec must be a JSON list or an object with `agents`.")
    return tuple(AgentSpec.from_dict(item) for item in payload if isinstance(item, dict))


def load_agent_specs(
    *,
    catalog_mode: str = DEFAULT_CATALOG_MODE,
    agent_spec_path: Path | None = None,
) -> tuple[AgentSpec, ...]:
    if catalog_mode not in {"current", "expanded"}:
        raise ValueError("catalog_mode must be either `current` or `expanded`.")

    base_specs = (
        CURRENT_AGENT_SPECS
        if catalog_mode == "current"
        else (*CURRENT_AGENT_SPECS, *FUTURE_AGENT_SPECS)
    )
    custom_specs = _load_custom_agent_specs(agent_spec_path)
    return _unique_specs([*base_specs, *custom_specs])


def _find_next_route(messages: list[object], start_index: int) -> str | None:
    for raw_message in messages[start_index:]:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role", "")).strip()
        content = str(raw_message.get("content", "")).strip()
        if role == "assistant" and content:
            return content
        if role == "user":
            return None
    return None


def _load_reference_scenarios(
    *,
    evaluation_dir: Path | None,
    allowed_agents: set[str],
) -> tuple[RouterScenario, ...]:
    resolved_evaluation_dir = evaluation_dir or _default_evaluation_dir()
    flows_path = resolved_evaluation_dir / "router_flows.json"
    if not flows_path.is_file():
        return ()

    payload = json.loads(flows_path.read_text(encoding="utf-8"))
    scenarios: list[RouterScenario] = []
    for flow_payload in payload:
        if not isinstance(flow_payload, dict):
            continue
        messages = flow_payload.get("messages", [])
        if not isinstance(messages, list):
            continue

        history: list[ConversationTurn] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not content:
                continue

            if role == "user":
                expected_agent = _find_next_route(messages, index + 1)
                if expected_agent in allowed_agents:
                    source_name = str(
                        message.get("source_scenario_name")
                        or flow_payload.get("name")
                        or f"scenario_{len(scenarios) + 1}"
                    )
                    scenarios.append(
                        RouterScenario(
                            name=f"reference::{source_name}",
                            history=tuple(history),
                            current_message=content,
                            expected_agent=str(expected_agent),
                            tags=("reference", "router_flows"),
                        )
                    )
                history.append(ConversationTurn(role="user", content=content))
                continue

            if role == "assistant" and content in allowed_agents:
                history.append(ConversationTurn(role="model", content=content))

    return tuple(scenarios)


def _render_rules(agent_specs: Sequence[AgentSpec]) -> str:
    return "\n".join(
        f"- Jeśli wiadomość dotyczy {', '.join(spec.routing_hints)} -> `{spec.name}`"
        for spec in agent_specs
    )


def _build_prompt_examples(agent_specs: Sequence[AgentSpec]) -> tuple[RouterScenario, ...]:
    specs = list(agent_specs)
    examples: list[RouterScenario] = []

    for spec in specs[:3]:
        examples.append(
            RouterScenario(
                name=f"prompt::direct::{spec.name}",
                history=(),
                current_message=spec.direct_requests[0],
                expected_agent=spec.name,
            )
        )

    if specs:
        first = specs[0]
        examples.append(
            RouterScenario(
                name=f"prompt::follow_up::{first.name}",
                history=(
                    ConversationTurn(role="user", content=first.direct_requests[0]),
                    ConversationTurn(role="model", content=first.name),
                ),
                current_message=first.follow_ups[0],
                expected_agent=first.name,
            )
        )

    if len(specs) >= 2:
        previous = specs[0]
        target = specs[1]
        examples.append(
            RouterScenario(
                name=f"prompt::switch::{previous.name}_to_{target.name}",
                history=(
                    ConversationTurn(role="user", content=previous.direct_requests[0]),
                    ConversationTurn(role="model", content=previous.name),
                ),
                current_message=target.direct_requests[0],
                expected_agent=target.name,
            )
        )

    if len(specs) >= 5:
        extra = specs[4]
        examples.append(
            RouterScenario(
                name=f"prompt::future::{extra.name}",
                history=(),
                current_message=extra.direct_requests[0],
                expected_agent=extra.name,
            )
        )

    return tuple(examples[:6])


def build_history_aware_decision_prompt(
    *,
    agent_specs: Sequence[AgentSpec] | None = None,
) -> str:
    resolved_specs = tuple(agent_specs or CURRENT_AGENT_SPECS)
    examples = _build_prompt_examples(resolved_specs)
    example_blocks = "\n\n".join(
        "\n".join(
            [
                f"Przykład {index}:",
                "Wejście:",
                scenario.render_user_input(),
                f"Odpowiedź: {scenario.expected_agent}",
            ]
        )
        for index, scenario in enumerate(examples, start=1)
    )

    allowed_agents = ", ".join(f"`{spec.name}`" for spec in resolved_specs)
    prompt = dedent(
        f"""
        Wybierz 1 agenta na podstawie całego kontekstu rozmowy. Odpowiedz tylko nazwą agenta.

        Jesteś routerem/orchestratorem. Otrzymujesz historię rozmowy w formacie:
        - `<start_of_turn>user ... <end_of_turn>` dla wiadomości użytkownika
        - `<start_of_turn>model ... <end_of_turn>` dla poprzedniej decyzji routera

        Zasady interpretacji kontekstu:
        - Ostatni turn `user` to bieżąca wiadomość, którą masz zroute'ować.
        - Wcześniejsze turny `user` i `model` to historia oraz poprzednie decyzje routera.
        - Jeśli bieżąca wiadomość jest skrótowa, zależna od kontekstu lub zawiera odwołania typu: `to`, `ją`, `tę`, `jeszcze`, `a teraz`, `w tym temacie`, użyj ostatniej wiadomości użytkownika i ostatniej decyzji routera.
        - Jeśli bieżąca wiadomość jasno zmienia temat, wybierz agenta na podstawie bieżącej wiadomości, nawet jeśli poprzednia decyzja była inna.
        - Jeśli użytkownik wraca do starszego wątku, a treść to sygnalizuje, nie trzymaj się ślepo ostatniej decyzji.
        - Decyzję podejmuj całościowo: bieżąca wiadomość jest najważniejsza, ale historia rozstrzyga niejednoznaczności.

        Ważne:
        - Lista agentów poniżej jest źródłem prawdy.
        - Agent może być nowy lub przyszły i nie musi jeszcze istnieć w kodzie produkcyjnym.
        - Masz wybrać dokładnie jednego agenta z listy dostępnych agentów.

        Dostępni agenci:
        {chr(10).join(f"- `{spec.name}` - {spec.description}" for spec in resolved_specs)}

        Reguły wyboru:
        {_render_rules(resolved_specs)}
        - W przypadku niejasności wybierz najbardziej dopasowanego agenta na bazie bieżącego turnu i wcześniejszych decyzji.

        Odpowiedź musi być dokładnie jednym słowem z listy:
        {allowed_agents}
        Bez dodatkowych wyjaśnień.
        """
    ).strip()

    return f"{prompt}\n\nPrzykłady:\n\n{example_blocks}"


def _scenario_name(kind: str, agent_name: str, sample_index: int) -> str:
    return f"synthetic::{kind}::{agent_name}::{sample_index:03d}"


def _direct_request(
    *,
    spec: AgentSpec,
    rng: random.Random,
    sample_index: int,
) -> RouterScenario:
    return RouterScenario(
        name=_scenario_name("direct_request", spec.name, sample_index),
        history=(),
        current_message=_style_message(rng, _choose(rng, spec.direct_requests)),
        expected_agent=spec.name,
        tags=("synthetic", "direct_request", spec.name),
    )


def _same_agent_follow_up(
    *,
    spec: AgentSpec,
    rng: random.Random,
    sample_index: int,
) -> RouterScenario:
    opening = _choose(rng, spec.direct_requests)
    return RouterScenario(
        name=_scenario_name("same_agent_follow_up", spec.name, sample_index),
        history=(
            ConversationTurn(role="user", content=_style_message(rng, opening)),
            ConversationTurn(role="model", content=spec.name),
        ),
        current_message=_style_message(rng, _choose(rng, spec.follow_ups)),
        expected_agent=spec.name,
        tags=("synthetic", "same_agent_follow_up", spec.name),
    )


def _same_agent_correction(
    *,
    spec: AgentSpec,
    rng: random.Random,
    sample_index: int,
) -> RouterScenario:
    opening = _choose(rng, spec.direct_requests)
    replacement = _different_choice(rng, spec.direct_requests, opening)
    return RouterScenario(
        name=_scenario_name("same_agent_correction", spec.name, sample_index),
        history=(
            ConversationTurn(role="user", content=_style_message(rng, opening)),
            ConversationTurn(role="model", content=spec.name),
        ),
        current_message=_style_message(
            rng,
            f"{_choose(rng, CORRECTION_PREFIXES)}{replacement}",
        ),
        expected_agent=spec.name,
        tags=("synthetic", "same_agent_correction", spec.name),
    )


def _deep_history_same_agent(
    *,
    spec: AgentSpec,
    rng: random.Random,
    sample_index: int,
) -> RouterScenario:
    return RouterScenario(
        name=_scenario_name("deep_history_same_agent", spec.name, sample_index),
        history=(
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, spec.direct_requests))),
            ConversationTurn(role="model", content=spec.name),
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, spec.follow_ups))),
            ConversationTurn(role="model", content=spec.name),
        ),
        current_message=_style_message(rng, _different_choice(rng, spec.follow_ups, spec.follow_ups[0])),
        expected_agent=spec.name,
        tags=("synthetic", "deep_history_same_agent", spec.name),
    )


def _switch_from_previous_agent(
    *,
    target: AgentSpec,
    previous: AgentSpec,
    rng: random.Random,
    sample_index: int,
) -> RouterScenario:
    return RouterScenario(
        name=_scenario_name(
            f"switch_from_{previous.name}",
            target.name,
            sample_index,
        ),
        history=(
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, previous.direct_requests))),
            ConversationTurn(role="model", content=previous.name),
        ),
        current_message=_style_message(rng, _choose(rng, target.direct_requests)),
        expected_agent=target.name,
        tags=("synthetic", "topic_switch", previous.name, target.name),
    )


def _recent_context_wins(
    *,
    dominant: AgentSpec,
    recent: AgentSpec,
    rng: random.Random,
    sample_index: int,
) -> RouterScenario:
    return RouterScenario(
        name=_scenario_name(
            f"recent_context_wins_over_{dominant.name}",
            recent.name,
            sample_index,
        ),
        history=(
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, dominant.direct_requests))),
            ConversationTurn(role="model", content=dominant.name),
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, recent.direct_requests))),
            ConversationTurn(role="model", content=recent.name),
        ),
        current_message=_style_message(rng, _choose(rng, recent.follow_ups)),
        expected_agent=recent.name,
        tags=("synthetic", "recent_context_wins", dominant.name, recent.name),
    )


def _older_topic_returns(
    *,
    older: AgentSpec,
    recent: AgentSpec,
    rng: random.Random,
    sample_index: int,
) -> RouterScenario:
    return RouterScenario(
        name=_scenario_name(
            f"return_to_{older.name}_after_{recent.name}",
            older.name,
            sample_index,
        ),
        history=(
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, older.direct_requests))),
            ConversationTurn(role="model", content=older.name),
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, recent.direct_requests))),
            ConversationTurn(role="model", content=recent.name),
        ),
        current_message=_style_message(
            rng,
            f"{_choose(rng, RETURN_PREFIXES)}{_choose(rng, older.follow_ups)}",
        ),
        expected_agent=older.name,
        tags=("synthetic", "older_topic_returns", older.name, recent.name),
    )


def _long_history_topic_switch(
    *,
    target: AgentSpec,
    dominant: AgentSpec,
    rng: random.Random,
    sample_index: int,
) -> RouterScenario:
    return RouterScenario(
        name=_scenario_name(
            f"long_history_switch_from_{dominant.name}",
            target.name,
            sample_index,
        ),
        history=(
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, dominant.direct_requests))),
            ConversationTurn(role="model", content=dominant.name),
            ConversationTurn(role="user", content=_style_message(rng, _choose(rng, dominant.follow_ups))),
            ConversationTurn(role="model", content=dominant.name),
            ConversationTurn(role="user", content=_style_message(rng, _different_choice(rng, dominant.follow_ups, dominant.follow_ups[0]))),
            ConversationTurn(role="model", content=dominant.name),
        ),
        current_message=_style_message(rng, _choose(rng, target.direct_requests)),
        expected_agent=target.name,
        tags=("synthetic", "long_history_topic_switch", dominant.name, target.name),
    )


def build_seed_scenarios(
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    count_per_template: int = DEFAULT_SAMPLES_PER_TEMPLATE,
    catalog_mode: str = DEFAULT_CATALOG_MODE,
    agent_spec_path: Path | None = None,
    evaluation_dir: Path | None = None,
    agent_specs: Sequence[AgentSpec] | None = None,
) -> tuple[RouterScenario, ...]:
    rng = random.Random(seed)
    specs = tuple(
        agent_specs
        if agent_specs is not None
        else load_agent_specs(
            catalog_mode=catalog_mode,
            agent_spec_path=agent_spec_path,
        )
    )
    allowed_agents = {spec.name for spec in specs}

    scenarios: list[RouterScenario] = [
        *_load_reference_scenarios(
            evaluation_dir=evaluation_dir,
            allowed_agents=allowed_agents,
        )
    ]

    if not specs:
        return tuple(scenarios)

    for spec in specs:
        for sample_index in range(1, count_per_template + 1):
            scenarios.extend(
                [
                    _direct_request(spec=spec, rng=rng, sample_index=sample_index),
                    _same_agent_follow_up(spec=spec, rng=rng, sample_index=sample_index),
                    _same_agent_correction(spec=spec, rng=rng, sample_index=sample_index),
                    _deep_history_same_agent(spec=spec, rng=rng, sample_index=sample_index),
                ]
            )

    for sample_index in range(1, count_per_template + 1):
        for target in specs:
            previous = _choose(rng, [spec for spec in specs if spec.name != target.name] or [target])
            scenarios.append(
                _switch_from_previous_agent(
                    target=target,
                    previous=previous,
                    rng=rng,
                    sample_index=sample_index,
                )
            )
            dominant = _choose(rng, [spec for spec in specs if spec.name != target.name] or [target])
            scenarios.append(
                _recent_context_wins(
                    dominant=dominant,
                    recent=target,
                    rng=rng,
                    sample_index=sample_index,
                )
            )
            recent_for_return = _choose(rng, [spec for spec in specs if spec.name != target.name] or [target])
            scenarios.append(
                _older_topic_returns(
                    older=target,
                    recent=recent_for_return,
                    rng=rng,
                    sample_index=sample_index,
                )
            )
            dominant = _choose(rng, [spec for spec in specs if spec.name != target.name] or [target])
            scenarios.append(
                _long_history_topic_switch(
                    target=target,
                    dominant=dominant,
                    rng=rng,
                    sample_index=sample_index,
                )
            )

    unique_names: set[str] = set()
    deduped: list[RouterScenario] = []
    for scenario in scenarios:
        if scenario.name in unique_names:
            continue
        unique_names.add(scenario.name)
        deduped.append(scenario)
    return tuple(deduped)


def _render_seed_scenarios_for_prompt(
    scenarios: Sequence[RouterScenario],
    *,
    limit: int = 12,
) -> str:
    rendered: list[str] = []
    for index, scenario in enumerate(scenarios[:limit], start=1):
        rendered.append(
            "\n".join(
                [
                    f"Seed {index}: {scenario.name}",
                    "Wejście:",
                    scenario.render_user_input(),
                    f"Oczekiwana odpowiedź: {scenario.expected_agent}",
                ]
            )
        )
    return "\n\n".join(rendered)


def _build_topics_prompt(agent_specs: Sequence[AgentSpec]) -> str:
    agent_lines = "\n".join(
        f"- {spec.name}: {spec.description}; sygnały: {', '.join(spec.routing_hints)}"
        for spec in agent_specs
    )
    return dedent(
        f"""
        Wygeneruj różnorodne tematy datasetu dla routera/orchestratora agentów.
        Dataset ma uczyć wyboru jednego agenta na podstawie historii rozmowy po polsku.

        Dostępni agenci:
        {agent_lines}

        Szukaj tematów obejmujących:
        - pojedyncze, bezpośrednie prośby
        - follow-upy zależne od poprzedniej decyzji
        - krótkie doprecyzowania i korekty
        - zmianę tematu mimo długiej historii
        - powrót do starszego wątku
        - nowe lub przyszłe agenty z listy
        - niejednoznaczne rozmowy, gdzie historia rozstrzyga routing

        Każdy temat ma prowadzić do jednego finalnego wyboru agenta.
        Wszystkie przykłady muszą być po polsku.
        """
    ).strip()


def _build_generation_system_prompt(
    *,
    prompt_text: str,
    agent_specs: Sequence[AgentSpec],
    seed_scenarios: Sequence[RouterScenario],
) -> str:
    catalog_json = json.dumps(
        [spec.as_dict() for spec in agent_specs],
        ensure_ascii=False,
        indent=2,
    )
    seed_block = _render_seed_scenarios_for_prompt(seed_scenarios)
    return (
        "Jesteś ekspertem od tworzenia danych treningowych dla routera/orchestratora.\n\n"
        "<router-prompt>\n"
        f"{prompt_text}\n"
        "</router-prompt>\n\n"
        "<agent-catalog>\n"
        f"{catalog_json}\n"
        "</agent-catalog>\n\n"
        "<seed-scenarios>\n"
        f"{seed_block}\n"
        "</seed-scenarios>"
    )


def _build_generation_instructions(
    *,
    variant: str,
    allowed_agents: Sequence[str],
) -> str:
    route_list = ", ".join(allowed_agents)
    if variant == "reasoning":
        return dedent(
            f"""
            Wygeneruj prosty wieloetapowy przykład dla routera/orchestratora po polsku.
            Przykład ma zawierać krótką historię rozmowy w formacie turnów:
            `<start_of_turn>user ... <end_of_turn>` oraz `<start_of_turn>model ... <end_of_turn>`.

            Wymagania:
            - historia ma mieć 2-6 wcześniejszych turnów
            - ostatni turn `user` ma być bieżącą wiadomością do zroute'owania
            - przypadek ma wymagać prostego rozumowania kontekstowego, ale bez bardzo złożonej logiki
            - finalna odpowiedź modelu ma być dokładnie jedną nazwą agenta z listy: {route_list}
            - nie dodawaj wyjaśnień po finalnej odpowiedzi
            - całość po polsku
            """
        ).strip()

    return dedent(
        f"""
        Wygeneruj praktyczny przykład dla routera/orchestratora po polsku.
        Przykład ma zawierać rozmowę w formacie turnów:
        `<start_of_turn>user ... <end_of_turn>` oraz opcjonalnie wcześniejsze `<start_of_turn>model ... <end_of_turn>`.

        Wymagania:
        - finalny input ma prowadzić do wyboru dokładnie jednego agenta z listy: {route_list}
        - przykłady mają być różnorodne: bezpośrednie prośby, follow-upy, zmiany tematu, powroty do wcześniejszego wątku
        - finalna odpowiedź modelu ma być dokładnie jedną nazwą agenta
        - żadnych dodatkowych wyjaśnień w finalnej odpowiedzi
        - całość po polsku
        """
    ).strip()


def build_deepfabric_config(
    *,
    output_dir: Path,
    variant: str,
    prompt_text: str,
    agent_specs: Sequence[AgentSpec],
    seed_scenarios: Sequence[RouterScenario],
    num_samples: int,
) -> dict[str, object]:
    if variant not in CONFIG_VARIANTS:
        raise ValueError(f"Unsupported config variant: {variant}")

    allowed_agents = [spec.name for spec in agent_specs]
    suffix = "simple_reasoning" if variant == "reasoning" else "basic"

    return {
        "llm": {
            "provider": "openrouter",
            "model": "openrouter/hunter-alpha",
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.2,
            "max_tokens": 1400,
        },
        "topics": {
            "prompt": _build_topics_prompt(agent_specs),
            "mode": "graph",
            "depth": 3,
            "degree": 3,
            "save_as": str(output_dir / f"router_orchestrator_{suffix}_topics.jsonl"),
            "llm": {
                "provider": "openrouter",
                "model": "openrouter/hunter-alpha",
                "base_url": "https://openrouter.ai/api/v1",
                "temperature": 0.2,
                "max_tokens": 1400,
            },
        },
        "generation": {
            "system_prompt": _build_generation_system_prompt(
                prompt_text=prompt_text,
                agent_specs=agent_specs,
                seed_scenarios=seed_scenarios,
            ),
            "instructions": _build_generation_instructions(
                variant=variant,
                allowed_agents=allowed_agents,
            ),
            "conversation": (
                {"type": "cot", "reasoning_style": "freetext"}
                if variant == "reasoning"
                else {"type": "basic"}
            ),
            "llm": {
                "provider": "openrouter",
                "model": "openrouter/hunter-alpha",
                "base_url": "https://openrouter.ai/api/v1",
                "temperature": 0.3,
            },
        },
        "output": {
            "system_prompt": prompt_text,
            "include_system_message": True,
            "num_samples": num_samples,
            "batch_size": 4,
            "save_as": str(output_dir / f"router_orchestrator_{suffix}.jsonl"),
        },
    }


def write_configs(
    output_dir: Path,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    count_per_template: int = DEFAULT_SAMPLES_PER_TEMPLATE,
    catalog_mode: str = DEFAULT_CATALOG_MODE,
    agent_spec_path: Path | None = None,
    evaluation_dir: Path | None = None,
    basic_samples: int = DEFAULT_BASIC_SAMPLES,
    reasoning_samples: int = DEFAULT_REASONING_SAMPLES,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_specs = load_agent_specs(
        catalog_mode=catalog_mode,
        agent_spec_path=agent_spec_path,
    )
    prompt_text = build_history_aware_decision_prompt(agent_specs=agent_specs)
    seed_scenarios = build_seed_scenarios(
        seed=seed,
        count_per_template=count_per_template,
        catalog_mode=catalog_mode,
        agent_spec_path=agent_spec_path,
        evaluation_dir=evaluation_dir,
        agent_specs=agent_specs,
    )

    config_paths: dict[str, Path] = {}
    for variant, num_samples in (
        ("basic", basic_samples),
        ("reasoning", reasoning_samples),
    ):
        config = build_deepfabric_config(
            output_dir=output_dir,
            variant=variant,
            prompt_text=prompt_text,
            agent_specs=agent_specs,
            seed_scenarios=seed_scenarios,
            num_samples=num_samples,
        )
        suffix = "simple_reasoning" if variant == "reasoning" else "basic"
        config_path = output_dir / f"router_orchestrator_{suffix}.yaml"
        config_path.write_text(
            yaml.dump(
                config,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            ),
            encoding="utf-8",
        )
        config_paths[variant] = config_path

    return config_paths


def write_assets(
    output_dir: Path,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    count_per_template: int = DEFAULT_SAMPLES_PER_TEMPLATE,
    catalog_mode: str = DEFAULT_CATALOG_MODE,
    agent_spec_path: Path | None = None,
    evaluation_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_specs = load_agent_specs(
        catalog_mode=catalog_mode,
        agent_spec_path=agent_spec_path,
    )
    allowed_agents = [spec.name for spec in agent_specs]
    prompt_text = build_history_aware_decision_prompt(agent_specs=agent_specs)
    scenarios = build_seed_scenarios(
        seed=seed,
        count_per_template=count_per_template,
        catalog_mode=catalog_mode,
        agent_spec_path=agent_spec_path,
        evaluation_dir=evaluation_dir,
        agent_specs=agent_specs,
    )

    prompt_path = output_dir / "router_orchestrator_prompt.txt"
    scenarios_path = output_dir / "router_orchestrator_scenarios.json"
    jsonl_path = output_dir / "router_orchestrator_sft.jsonl"
    catalog_path = output_dir / "router_orchestrator_agent_catalog.json"

    prompt_path.write_text(prompt_text + "\n", encoding="utf-8")
    scenarios_path.write_text(
        json.dumps([scenario.as_dict() for scenario in scenarios], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps([spec.as_dict() for spec in agent_specs], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(
                json.dumps(
                    scenario.as_jsonl_row(
                        prompt_text,
                        allowed_agents=allowed_agents,
                    ),
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    return {
        "prompt": prompt_path,
        "scenarios": scenarios_path,
        "jsonl": jsonl_path,
        "catalog": catalog_path,
    }


def cmd_write_assets(args: argparse.Namespace) -> None:
    agent_spec_path = Path(args.agent_spec) if args.agent_spec is not None else None
    evaluation_dir = Path(args.evaluation_dir) if args.evaluation_dir is not None else None
    written = write_assets(
        Path(args.output_dir),
        seed=args.seed,
        count_per_template=args.samples_per_template,
        catalog_mode=args.catalog_mode,
        agent_spec_path=agent_spec_path,
        evaluation_dir=evaluation_dir,
    )
    scenarios = build_seed_scenarios(
        seed=args.seed,
        count_per_template=args.samples_per_template,
        catalog_mode=args.catalog_mode,
        agent_spec_path=agent_spec_path,
        evaluation_dir=evaluation_dir,
    )
    agent_specs = load_agent_specs(
        catalog_mode=args.catalog_mode,
        agent_spec_path=agent_spec_path,
    )

    print(f"Wrote prompt: {written['prompt']}")
    print(f"Wrote scenarios: {written['scenarios']}")
    print(f"Wrote SFT JSONL: {written['jsonl']}")
    print(f"Wrote agent catalog: {written['catalog']}")
    print(f"Agents: {', '.join(spec.name for spec in agent_specs)}")
    print(f"Scenarios: {len(scenarios)}")


def cmd_write_configs(args: argparse.Namespace) -> None:
    agent_spec_path = Path(args.agent_spec) if args.agent_spec is not None else None
    evaluation_dir = Path(args.evaluation_dir) if args.evaluation_dir is not None else None
    written = write_configs(
        Path(args.output_dir),
        seed=args.seed,
        count_per_template=args.samples_per_template,
        catalog_mode=args.catalog_mode,
        agent_spec_path=agent_spec_path,
        evaluation_dir=evaluation_dir,
        basic_samples=args.basic_samples,
        reasoning_samples=args.reasoning_samples,
    )
    print(f"Wrote basic config: {written['basic']}")
    print(f"Wrote simple reasoning config: {written['reasoning']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="router-orchestrator-dataset",
        description="Generate history-aware Polish router/orchestrator datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    write_assets_parser = sub.add_parser(
        "write-assets",
        help="Write prompt text, agent catalog, pretty scenarios and SFT JSONL rows",
    )
    write_assets_parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory for generated prompt and datasets",
    )
    write_assets_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for synthetic scenario generation",
    )
    write_assets_parser.add_argument(
        "--samples-per-template",
        type=int,
        default=DEFAULT_SAMPLES_PER_TEMPLATE,
        help="How many synthetic scenarios to generate per template and agent",
    )
    write_assets_parser.add_argument(
        "--catalog-mode",
        choices=("current", "expanded"),
        default=DEFAULT_CATALOG_MODE,
        help="Use only current agents or an expanded catalog with future agents",
    )
    write_assets_parser.add_argument(
        "--agent-spec",
        default=None,
        help="Path to JSON agent catalog with additional or overriding agents",
    )
    write_assets_parser.add_argument(
        "--evaluation-dir",
        default=str(_default_evaluation_dir()),
        help="Path to evaluation assets used as reference scenarios",
    )

    write_configs_parser = sub.add_parser(
        "write-configs",
        help="Write DeepFabric YAML configs for basic and simple reasoning routing data",
    )
    write_configs_parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory for generated YAML configs",
    )
    write_configs_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for synthetic scenario generation",
    )
    write_configs_parser.add_argument(
        "--samples-per-template",
        type=int,
        default=DEFAULT_SAMPLES_PER_TEMPLATE,
        help="How many synthetic scenarios to generate per template and agent for prompt context",
    )
    write_configs_parser.add_argument(
        "--catalog-mode",
        choices=("current", "expanded"),
        default=DEFAULT_CATALOG_MODE,
        help="Use only current agents or an expanded catalog with future agents",
    )
    write_configs_parser.add_argument(
        "--agent-spec",
        default=None,
        help="Path to JSON agent catalog with additional or overriding agents",
    )
    write_configs_parser.add_argument(
        "--evaluation-dir",
        default=str(_default_evaluation_dir()),
        help="Path to evaluation assets used as reference scenarios",
    )
    write_configs_parser.add_argument(
        "--basic-samples",
        type=int,
        default=DEFAULT_BASIC_SAMPLES,
        help="Target DeepFabric sample count for the basic config",
    )
    write_configs_parser.add_argument(
        "--reasoning-samples",
        type=int,
        default=DEFAULT_REASONING_SAMPLES,
        help="Target DeepFabric sample count for the simple reasoning config",
    )

    args = parser.parse_args()
    if args.command == "write-assets":
        cmd_write_assets(args)
    elif args.command == "write-configs":
        cmd_write_configs(args)


__all__ = [
    "AgentSpec",
    "ConversationTurn",
    "CURRENT_AGENT_SPECS",
    "DEFAULT_CATALOG_MODE",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_SAMPLES_PER_TEMPLATE",
    "FUTURE_AGENT_SPECS",
    "OUTPUT_DIR_NAME",
    "ROUTES",
    "RouterScenario",
    "build_history_aware_decision_prompt",
    "build_deepfabric_config",
    "build_seed_scenarios",
    "load_agent_specs",
    "main",
    "write_assets",
    "write_configs",
]


if __name__ == "__main__":
    main()
