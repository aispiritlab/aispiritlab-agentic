from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson
import pytest
from deepeval.errors import DeepEvalError
from deepeval.metrics import (
    GEval,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ToolCall,
    ToolCallParams,
)

from agentic.models import ModelProvider as AgenticModelProvider
import agentic.prompts as agentic_prompts
from personal_assistant.agents.manage_notes.evaluation import NOTES_EVALUATION, NOTES_TOOL_SCENARIOS
from evaluation import ConversationScenario, ToolScenario, build_conversation_scenarios
from evaluation.deepeval_providers.deepeval_openrouter import FixedOpenRouterModel
from registry.prompts import GREETING_PROMPT, MANAGE_NOTES_PROMPT, Prompts


DEFAULT_NOTES_OPENROUTER_MODELS: tuple[str, ...] = (
    "mlx-community/Phi-4-mini-reasoning-bf16",
    "mlx-community/Phi-4-mini-instruct-8bit",
    "google/gemma-3-4b-it",
    "google/gemma-3-1b-it",
)

DEFAULT_OPENROUTER_JUDGE_MODEL = "minimax/minimax-m2.5"
AVAILABLE_TOOLS = ("add_note", "edit_note", "get_note", "list_notes")
HF_URL_PREFIX = "https://huggingface.co/"


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OPENROUTER_BENCHMARKS", "").lower() not in {"1", "true", "yes"}
    or not os.getenv("OPENROUTER_API_KEY"),
    reason=(
        "OpenRouter benchmark is opt-in and paid. Set RUN_OPENROUTER_BENCHMARKS=1 "
        "and OPENROUTER_API_KEY to run."
    ),
)


@dataclass(frozen=True)
class ScenarioBenchmarkResult:
    scenario_name: str
    model_id: str
    requested_model_id: str
    latency_ms: float
    raw_model_output: str | None
    final_output: str | None
    called_tool_name: str | None
    called_parameters: dict[str, Any]
    expected_tool_name: str
    expected_parameters: dict[str, Any]
    exact_tool_match: bool
    tool_correctness_score: float | None
    step_efficiency_score: float | None
    task_completion_score: float | None
    error: str | None = None


@dataclass(frozen=True)
class ModelBenchmarkSummary:
    requested_model_id: str
    model_id: str
    scenarios_total: int
    scenarios_completed: int
    scenarios_failed: int
    exact_tool_match_count: int
    exact_tool_match_rate: float | None
    avg_tool_correctness: float | None
    avg_step_efficiency: float | None
    avg_task_completion: float | None
    avg_latency_ms: float | None
    composite_score: float | None
    status: str
    error: str | None
    failed_scenarios: tuple[str, ...]


@dataclass(frozen=True)
class ConversationBenchmarkResult:
    conversation_name: str
    requested_model_id: str
    model_id: str
    turns_total: int
    turns_completed: int
    exact_turn_match_count: int
    exact_turn_match_rate: float | None
    latency_ms: float
    tool_correctness_score: float | None
    step_efficiency_score: float | None
    task_completion_score: float | None
    final_output: str | None
    error: str | None = None


@dataclass(frozen=True)
class ConversationModelBenchmarkSummary:
    requested_model_id: str
    model_id: str
    conversations_total: int
    conversations_completed: int
    conversations_failed: int
    avg_exact_turn_match_rate: float | None
    avg_tool_correctness: float | None
    avg_step_efficiency: float | None
    avg_task_completion: float | None
    avg_latency_ms: float | None
    composite_score: float | None
    status: str
    error: str | None
    failed_conversations: tuple[str, ...]


@dataclass(frozen=True)
class PromptStyleCase:
    name: str
    prompt_name: Prompts
    user_input: str
    expected_output: str


PROMPT_STYLE_LLM_AS_JUDGE_CRITERIA = """
Oceń odpowiedź w języku POLSKIM. Porównaj rzeczywistą odpowiedź (actual output) z oczekiwaną odpowiedzią (expected output) jako wzorcem referencyjnym.

Domena: asystent do zarządzania notatkami i personalizacji bota.

To jest ocena pojedynczego przypadku testowego (single-case), nie ranking wielu uczestników.
Przyznaj ocenę na podstawie tego, jak dobrze actual output dopasowuje się do expected output pod względem:

     1. POPRAWNE UŻYCIE NARZĘDZI (najważniejsze):
     - Czy wywołuje właściwe narzędzie (szczególnie: update_personalization, add_note, edit_note, get_note, list_notes)?
     - Czy wywołuje narzędzie we właściwym momencie (jak w expected output)?
     - Czy przekazuje poprawne parametry do narzędzia (np. name, vault_name, note_name, note)?
     - Czy zachowuje poprawny format wywołania (czysty JSON albo blok <tool_call>...</tool_call>, zgodnie z expected output)?
     - Jeśli expected output nie wymaga wywołania narzędzia, czy odpowiedź poprawnie tego nie robi?

  2. PRZESTRZEGANIE REGUŁ:
     - Czy odpowiedź wykonuje WSZYSTKIE kroki wymienione w instrukcji?
     - Czy zachowuje wymagany format odpowiedzi (JSON / <tool_call>, bez dodatkowego tekstu jeśli jest zabroniony)?
     - Czy przestrzega ograniczeń (np. język polski, brak dodatkowych sekcji, brak komentarzy)?
     - Czy kroki są wykonane w podanej kolejności?
     - Czy odpowiedź jest zgodna z przykładami i szablonami z instrukcji (format, styl, struktura)?

  3. POPRAWNOŚĆ MAPOWANIA INTENCJI:
     - Czy intencja użytkownika została poprawnie rozpoznana?
     - Czy nazwa notatki / treść / dane personalizacji zostały zachowane 1:1 bez nieuprawnionych zmian?
     - Czy model nie normalizuje, nie poprawia i nie zgaduje danych, jeśli expected output zachowuje oryginalne wartości?

  4. UWZGLĘDNIENIE KONTEKSTU / HISTORII KONWERSACJI:
     - Jeśli input zawiera wcześniejsze turny, czy odpowiedź uwzględnia informacje z poprzednich tur?
     - Czy model poprawnie rozwiązuje odwołania kontekstowe typu „ostatnia notatka” / „ta notatka”?
     - Czy model nie ignoruje kontekstu i nie zaczyna od zera?

  5. PRZESTRZEGANIE PERSONALIZACJI:
     - Czy odpowiedź utrzymuje wymagany styl/ton i język?
     - Czy przestrzega wszystkich ograniczeń z instrukcji systemowej?

  6. JAKOŚĆ ODPOWIEDZI:
     - Czy odpowiedź jest użyteczna i kompletna względem expected output?
     - Czy informacje są dokładne?
     - Czy język polski jest poprawny gramatycznie?

  Priorytet: narzędzia > przestrzeganie reguł > mapowanie intencji > kontekst/historia > personalizacja > jakość.
""".strip()


PROMPT_STYLE_CASES: tuple[PromptStyleCase, ...] = (
    PromptStyleCase(
        name="greeting_confirm_personalization_json",
        prompt_name=Prompts.GREETING,
        user_input="""
<start_of_turn>user
Cześć
<end_of_turn>
<start_of_turn>model
Cześć! Jak mam się do Ciebie zwracać?
<end_of_turn>
<start_of_turn>user
Mateusz
<end_of_turn>
<start_of_turn>model
Dobrze, Mateusz. Podaj nazwę vaulta Obsidian (taką, jak w aplikacji Obsidian).
<end_of_turn>
<start_of_turn>user
MyVault
<end_of_turn>
<start_of_turn>model
Podsumowanie:
- Nazwa: Mateusz
- Vault: MyVault
Czy potwierdzasz? (Tak/Nie)
<end_of_turn>
<start_of_turn>user
Tak
<end_of_turn>
""".strip(),
        expected_output="""
{"name":"update_personalization","parameters":{"name":"Mateusz","vault_name":"MyVault"}}
""".strip(),
    ),
    PromptStyleCase(
        name="note_edit_last_note_from_history",
        prompt_name=Prompts.MANAGE_NOTES,
        user_input="""
<start_of_turn>user
Dodaj notatkę o nazwie Zakupy z treścią Mleko i chleb
<end_of_turn>
<start_of_turn>model
<tool_call>
{"name":"add_note","parameters":{"note_name":"Zakupy","note":"Mleko i chleb"}}
</tool_call>
<end_of_turn>
<start_of_turn>user
Dopisz do ostatniej notatki Jajka
<end_of_turn>
""".strip(),
        expected_output="""
<tool_call>
{"name":"edit_note","parameters":{"note_name":"Zakupy","note":"Jajka"}}
</tool_call>
""".strip(),
    ),
)


def _is_full_turn_text(text: str) -> bool:
    candidate = text.strip()
    return candidate.startswith("<start_of_turn>") and "<end_of_turn>" in candidate


def _as_turn(role: str, text: str) -> str:
    return f"<start_of_turn>{role}\n{text.strip()}\n<end_of_turn>"


def _prompt_style_system_prompt(prompt_name: Prompts) -> str:
    if prompt_name == Prompts.GREETING:
        return GREETING_PROMPT
    if prompt_name == Prompts.MANAGE_NOTES:
        return MANAGE_NOTES_PROMPT
    raise ValueError(f"Unsupported prompt style case prompt: {prompt_name}")


def _build_prompt_style_case_input(case: PromptStyleCase) -> str:
    system_prompt = _prompt_style_system_prompt(case.prompt_name).strip()
    system_turn = (
        system_prompt if _is_full_turn_text(system_prompt) else _as_turn("system", system_prompt)
    )
    user_turns = (
        case.user_input.strip()
        if _is_full_turn_text(case.user_input)
        else _as_turn("user", case.user_input)
    )
    return f"{system_turn}\n{user_turns}\n<start_of_turn>model\n"


def _registry_prompt_loader(name: str) -> str:
    if name == Prompts.MANAGE_NOTES:
        return MANAGE_NOTES_PROMPT
    if name == Prompts.GREETING:
        return GREETING_PROMPT
    return MANAGE_NOTES_PROMPT


def _strict_parse_json(text: str) -> Any | None:
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        return None


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _canonicalize_model_id(raw_model_id: str) -> tuple[str, str]:
    requested = raw_model_id.strip()
    model_id = requested
    if model_id.startswith(HF_URL_PREFIX):
        model_id = model_id.removeprefix(HF_URL_PREFIX)
    model_id = model_id.strip()
    return requested, model_id


def _candidate_models() -> tuple[tuple[str, str], ...]:
    raw = os.getenv("NOTES_OPENROUTER_MODELS")
    if raw and raw.strip():
        candidates = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        candidates = list(DEFAULT_NOTES_OPENROUTER_MODELS)
    return tuple(_canonicalize_model_id(model_id) for model_id in candidates)


def _judge_model_id() -> str:
    return (
        os.getenv("NOTES_OPENROUTER_JUDGE_MODEL", DEFAULT_OPENROUTER_JUDGE_MODEL)
        .strip()
        .lower()
    )


def _benchmark_settings() -> tuple[str, tuple[tuple[str, str], ...], bool]:
    return (
        _judge_model_id(),
        _candidate_models(),
        _bool_env("NOTES_OPENROUTER_WRITE_REPORT", default=True),
    )


def _selected_scenarios() -> tuple[ToolScenario, ...]:
    selected_names_raw = os.getenv("NOTES_OPENROUTER_SCENARIOS", "")
    selected_names = {
        name.strip()
        for name in selected_names_raw.split(",")
        if name.strip()
    }
    limit = _parse_int_env("NOTES_OPENROUTER_MAX_SCENARIOS")

    scenarios = NOTES_TOOL_SCENARIOS
    if selected_names:
        scenarios = tuple(s for s in scenarios if s.name in selected_names)
    if limit is not None and limit >= 0:
        scenarios = scenarios[:limit]
    return scenarios


def _selected_conversations() -> tuple[ConversationScenario, ...]:
    selected_names_raw = os.getenv("NOTES_OPENROUTER_CONVERSATIONS", "")
    selected_names = {
        name.strip()
        for name in selected_names_raw.split(",")
        if name.strip()
    }
    limit = _parse_int_env("NOTES_OPENROUTER_MAX_CONVERSATIONS")

    conversations = build_conversation_scenarios(NOTES_EVALUATION)
    if selected_names:
        conversations = tuple(
            conversation
            for conversation in conversations
            if conversation.name in selected_names
        )
    if limit is not None and limit >= 0:
        conversations = conversations[:limit]
    return conversations


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _build_notes_full_rank_list(
    ranked: list[ModelBenchmarkSummary],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, summary in enumerate(ranked, start=1):
        items.append(
            {
                "rank": index,
                "requested_model_id": summary.requested_model_id,
                "model_id": summary.model_id,
                "status": summary.status,
                "composite_score": summary.composite_score,
                "exact_tool_match_rate": summary.exact_tool_match_rate,
                "avg_tool_correctness": summary.avg_tool_correctness,
                "avg_step_efficiency": summary.avg_step_efficiency,
                "avg_task_completion": summary.avg_task_completion,
                "avg_latency_ms": summary.avg_latency_ms,
                "scenarios_completed": summary.scenarios_completed,
                "scenarios_total": summary.scenarios_total,
            }
        )
    return items


def _build_prompt_case_full_rank_list(
    ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, summary in enumerate(ranked, start=1):
        items.append(
            {
                "rank": index,
                "requested_model_id": summary.get("requested_model_id"),
                "model_id": summary.get("model_id"),
                "status": summary.get("status"),
                "composite_score": summary.get("composite_score"),
                "avg_g_eval": summary.get("avg_g_eval"),
                "avg_task_completion": summary.get("avg_task_completion"),
                "avg_latency_ms": summary.get("avg_latency_ms"),
                "cases_completed": summary.get("cases_completed"),
                "cases_total": summary.get("cases_total"),
            }
        )
    return items


def _build_conversation_full_rank_list(
    ranked: list[ConversationModelBenchmarkSummary],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, summary in enumerate(ranked, start=1):
        items.append(
            {
                "rank": index,
                "requested_model_id": summary.requested_model_id,
                "model_id": summary.model_id,
                "status": summary.status,
                "composite_score": summary.composite_score,
                "avg_exact_turn_match_rate": summary.avg_exact_turn_match_rate,
                "avg_tool_correctness": summary.avg_tool_correctness,
                "avg_step_efficiency": summary.avg_step_efficiency,
                "avg_task_completion": summary.avg_task_completion,
                "avg_latency_ms": summary.avg_latency_ms,
                "conversations_completed": summary.conversations_completed,
                "conversations_total": summary.conversations_total,
            }
        )
    return items


class _LocalCandidateAgentAdapter:
    def __init__(self, model_id: str):
        self._model_id = model_id
        self._provider = AgenticModelProvider(model_id)
        self.last_response: str | None = None

    def response(self, prompt: str) -> str:
        with self._provider.session("model") as model:
            if model is None:
                raise RuntimeError(
                    f"Nie udało się załadować lokalnego modelu MLX/HF: {self._model_id}"
                )
            text = model.response(prompt).text
        self.last_response = text.strip()
        return self.last_response

    def probe_load_error(self) -> str | None:
        model = self._provider.get("model")
        if model is not None:
            return None
        hint = ""
        if "flash" in self._model_id and "reasoning" in self._model_id:
            hint = (
                " (to wygląda jak identyfikator modelu hosted/OpenRouter, a nie lokalny "
                "repozytorium MLX/HF)"
            )
        return (
            f"Nie udało się załadować lokalnego modelu MLX/HF: {self._model_id}{hint}"
        )


class _LenientOpenRouterJudge(DeepEvalBaseLLM):
    """OpenRouter judge wrapper that tolerates non-strict JSON outputs.

    Some judge models (e.g. `minimax/minimax-m2.5`) may ignore Deepeval's JSON-only
    instructions and return code fences / extra prose / malformed JSON. Deepeval's
    default OpenRouterModel then raises DeepEvalError. This wrapper retries by:
    1) extracting a JSON object from raw text,
    2) asking the same judge to repair JSON (plain text mode).
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._inner = FixedOpenRouterModel(
            model=model,
            temperature=temperature,
            generation_kwargs=generation_kwargs or {"max_tokens": 1024},
        )
        super().__init__(model=model)

    def load_model(self) -> "_LenientOpenRouterJudge":
        return self

    def get_model_name(self) -> str:
        return f"{self.name} (OpenRouter lenient-json judge)"

    def generate(self, prompt: str, schema=None) -> Any:
        if schema is None:
            return self._generate_text(prompt)

        try:
            # Fast path: if model behaves, let the provider handle structured output.
            return self._inner.generate(prompt, schema=schema)
        except DeepEvalError:
            pass
        except Exception:
            # Fall through to lenient mode for any provider-specific parsing issue.
            pass

        raw_output = self._generate_text(prompt)
        parsed = self._parse_schema_from_text(raw_output, schema)
        if parsed is not None:
            return parsed

        repaired_output = self._generate_text(
            self._build_json_repair_prompt(
                original_prompt=prompt,
                bad_output=raw_output,
                schema=schema,
            )
        )
        parsed = self._parse_schema_from_text(repaired_output, schema)
        if parsed is not None:
            return parsed

        raise DeepEvalError(
            "Evaluation LLM outputted an invalid JSON (even after lenient parsing/repair). "
            "Try another judge model or lower temperature."
        )

    async def a_generate(self, prompt: str, schema=None) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def _generate_text(self, prompt: str) -> str:
        output = self._inner.generate(prompt)
        return output if isinstance(output, str) else str(output)

    def _parse_schema_from_text(self, text: str, schema) -> Any | None:
        for candidate in _json_candidates(text):
            data = _json_loads_lenient(candidate)
            if not isinstance(data, dict):
                continue
            try:
                return schema.model_validate(data)
            except Exception:
                continue
        return None

    def _build_json_repair_prompt(self, *, original_prompt: str, bad_output: str, schema) -> str:
        schema_name = getattr(schema, "__name__", "Schema")
        schema_json = "{}"
        try:
            schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        except Exception:
            pass

        return (
            "You are fixing a JSON response for an evaluation tool.\n"
            "Return ONLY one valid JSON object matching the schema.\n"
            "Do not include markdown, explanations, or extra text.\n\n"
            f"Schema name: {schema_name}\n"
            f"JSON Schema: {schema_json}\n\n"
            "Original prompt (context only):\n"
            f"{original_prompt}\n\n"
            "Broken model output:\n"
            f"{bad_output}\n\n"
            "Return fixed JSON now:"
        )


def _build_judge_llm(judge_model_id: str) -> _LenientOpenRouterJudge:
    return _LenientOpenRouterJudge(
        model=judge_model_id,
        temperature=0.0,
        generation_kwargs={"max_tokens": 1024},
    )


def _available_tool_calls() -> list[ToolCall]:
    return [ToolCall(name=name) for name in AVAILABLE_TOOLS]


def _build_tool_correctness_metric(
    judge_llm: DeepEvalBaseLLM,
    *,
    threshold: float = 1.0,
) -> ToolCorrectnessMetric:
    return ToolCorrectnessMetric(
        available_tools=_available_tool_calls(),
        threshold=threshold,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        should_exact_match=True,
        should_consider_ordering=True,
        async_mode=False,
        model=judge_llm,
    )


def _build_step_efficiency_metric(
    judge_llm: DeepEvalBaseLLM,
    *,
    threshold: float,
) -> StepEfficiencyMetric:
    return StepEfficiencyMetric(
        threshold=threshold,
        async_mode=False,
        model=judge_llm,
    )


def _build_task_completion_metric(
    judge_llm: DeepEvalBaseLLM,
    *,
    threshold: float,
) -> TaskCompletionMetric:
    return TaskCompletionMetric(
        threshold=threshold,
        async_mode=False,
        model=judge_llm,
    )


def _write_json_report(
    payload: dict[str, Any],
    *,
    write_report: bool,
    report_path: Path,
) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if not write_report:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")


def _json_candidates(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    candidates: list[str] = [text]

    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match:
        candidates.append(generic_fence_match.group(1).strip())

    # Extract outermost JSON object/array spans if the model wraps JSON in prose.
    candidates.extend(_extract_brace_spans(text, "{", "}"))
    candidates.extend(_extract_brace_spans(text, "[", "]"))

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _extract_brace_spans(text: str, open_ch: str, close_ch: str) -> list[str]:
    spans: list[str] = []
    depth = 0
    start: int | None = None
    for idx, ch in enumerate(text):
        if ch == open_ch:
            if depth == 0:
                start = idx
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : idx + 1])
                start = None
    return spans


def _json_loads_lenient(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Common cleanup: trailing commas before } or ]
    text_without_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(text_without_trailing_commas)
    except json.JSONDecodeError:
        return None


@pytest.fixture()
def notes_manager(monkeypatch, tmp_path):
    monkeypatch.setattr(agentic_prompts, "get_prompt", _registry_prompt_loader)

    import agentic.tools as agentic_tools

    monkeypatch.setattr(
        agentic_tools.JsonParser,
        "parse_json",
        staticmethod(_strict_parse_json),
    )

    import agentic_runtime.trace as trace_module

    monkeypatch.setattr(trace_module, "init_tracing", lambda: "test-run")

    from personal_assistant.agents.manage_notes import tools as note_tools

    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    personalization_file = tmp_path / "personalization.json"
    personalization_file.write_text(
        json.dumps({"vault_path": str(vault_path)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(note_tools, "PERSONALIZATION_FILE", personalization_file)

    import agentic_runtime as runtime_module

    runtime_module = importlib.reload(runtime_module)
    runtime_module._reset_runtime()
    runtime = runtime_module.get_runtime()
    runtime.note_workflow._agent.reset()
    runtime.note_workflow._agent.start()
    return runtime


def _install_model_for_notes_agent(
    monkeypatch,
    notes_manager,
    adapter: _LocalCandidateAgentAdapter,
) -> None:
    @contextmanager
    def fake_session(name: str = "model"):  # noqa: ARG001
        yield adapter

    monkeypatch.setattr(
        notes_manager.note_workflow._agent._model_provider,
        "session",
        fake_session,
    )


def _reset_and_prefill(notes_manager, scenario: ToolScenario) -> None:
    notes_manager.note_workflow._agent.reset()
    notes_manager.note_workflow._agent.start()
    for prefill_message in scenario.prefill_messages:
        notes_manager.note_workflow._agent.call(prefill_message)


def _build_tool_test_case(
    scenario: ToolScenario,
    final_output: str | None,
    called_tool_name: str | None,
    called_parameters: dict[str, Any],
) -> LLMTestCase:
    tools_called = []
    if called_tool_name is not None:
        tools_called.append(
            ToolCall(name=called_tool_name, input_parameters=called_parameters),
        )

    return LLMTestCase(
        input=scenario.user_message_pl,
        actual_output=final_output or "",
        tools_called=tools_called,
        expected_tools=[
            ToolCall(
                name=scenario.tool_name,
                input_parameters=scenario.parameters,
            ),
        ],
    )


def _build_trace(
    scenario: ToolScenario,
    final_output: str | None,
    called_tool_name: str | None,
    called_parameters: dict[str, Any],
) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    if called_tool_name is not None:
        children.append(
            {
                "name": called_tool_name,
                "type": "tool",
                "input": {"inputParameters": called_parameters},
                "output": final_output or "",
                "children": [],
            }
        )

    return {
        "name": "notes_agent",
        "type": "agent",
        "input": {"input": scenario.user_message_pl},
        "output": final_output or "",
        "available_tools": list(AVAILABLE_TOOLS),
        "children": children,
    }


def _build_conversation_trace(
    conversation: ConversationScenario,
    *,
    turn_outputs: list[str],
    called_tools: list[ToolCall],
) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    for idx, (step, output) in enumerate(zip(conversation.steps, turn_outputs, strict=False), start=1):
        matched_call = called_tools[idx - 1] if idx - 1 < len(called_tools) else None
        if matched_call is not None and matched_call.name != "__missing_tool__":
            children.append(
                {
                    "name": f"{matched_call.name}#{idx}",
                    "type": "tool",
                    "input": {"inputParameters": matched_call.input_parameters or {}},
                    "output": output,
                    "children": [],
                }
            )
        else:
            # Preserve the turn even when tool detection failed.
            children.append(
                {
                    "name": f"turn_{idx}_llm_only",
                    "type": "llm",
                    "input": {"prompt": step.user_message_pl},
                    "output": output,
                    "children": [],
                }
            )

    return {
        "name": "notes_conversation_agent",
        "type": "agent",
        "input": {
            "conversation": [
                {"turn": idx, "user": step.user_message_pl}
                for idx, step in enumerate(conversation.steps, start=1)
            ]
        },
        "output": {
            "turns_total": len(conversation.steps),
            "turns_completed": len(turn_outputs),
            "final_output": turn_outputs[-1] if turn_outputs else "",
        },
        "available_tools": list(AVAILABLE_TOOLS),
        "children": children,
    }


def _run_conversation(
    notes_manager,
    conversation: ConversationScenario,
    *,
    adapter: _LocalCandidateAgentAdapter,
    requested_model_id: str,
    tool_metric: ToolCorrectnessMetric,
    step_metric: StepEfficiencyMetric,
    task_metric: TaskCompletionMetric,
) -> ConversationBenchmarkResult:
    notes_manager.note_workflow._agent.reset()
    notes_manager.note_workflow._agent.start()

    start = time.perf_counter()
    called_tools: list[ToolCall] = []
    expected_tools: list[ToolCall] = []
    turn_outputs: list[str] = []
    exact_turn_match_count = 0

    try:
        for step in conversation.steps:
            expected_tools.append(
                ToolCall(name=step.tool_name, input_parameters=step.parameters)
            )
            model_reply = notes_manager.note_workflow._agent._agent.run(step.user_message_pl)
            output = model_reply.content or ""
            turn_outputs.append(output)

            if model_reply.tool_call is not None:
                called_name, called_params = model_reply.tool_call
                called_tool = ToolCall(
                    name=called_name,
                    input_parameters=called_params,
                )
                called_tools.append(called_tool)
                if called_name == step.tool_name and called_params == step.parameters:
                    exact_turn_match_count += 1
            else:
                # Keep index alignment for trace/tool-order scoring.
                called_tools.append(
                    ToolCall(name="__missing_tool__", input_parameters={})
                )
    except Exception as error:  # pragma: no cover - environment/model-dependent
        elapsed_ms = (time.perf_counter() - start) * 1000
        completed_turns = len(turn_outputs)
        turns_total = len(conversation.steps)
        return ConversationBenchmarkResult(
            conversation_name=conversation.name,
            requested_model_id=requested_model_id,
            model_id=adapter._model_id,
            turns_total=turns_total,
            turns_completed=completed_turns,
            exact_turn_match_count=exact_turn_match_count,
            exact_turn_match_rate=_round_or_none(
                exact_turn_match_count / completed_turns if completed_turns else None
            ),
            latency_ms=round(elapsed_ms, 2),
            tool_correctness_score=None,
            step_efficiency_score=None,
            task_completion_score=None,
            final_output=turn_outputs[-1] if turn_outputs else adapter.last_response,
            error=str(error),
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    turns_total = len(conversation.steps)
    # Remove synthetic placeholders for metric input; ordering penalties are still captured by exact_turn_match_rate.
    metric_tools_called = [
        tool for tool in called_tools if tool.name != "__missing_tool__"
    ]
    test_case = LLMTestCase(
        input="\n".join(
            f"Turn {idx}. {step.user_message_pl}"
            for idx, step in enumerate(conversation.steps, start=1)
        ),
        actual_output=turn_outputs[-1] if turn_outputs else "",
        tools_called=metric_tools_called,
        expected_tools=expected_tools,
    )
    test_case._trace_dict = _build_conversation_trace(
        conversation,
        turn_outputs=turn_outputs,
        called_tools=called_tools,
    )

    tool_score = tool_metric.measure(test_case)
    step_score = step_metric.measure(test_case)
    task_score = task_metric.measure(test_case)

    return ConversationBenchmarkResult(
        conversation_name=conversation.name,
        requested_model_id=requested_model_id,
        model_id=adapter._model_id,
        turns_total=turns_total,
        turns_completed=turns_total,
        exact_turn_match_count=exact_turn_match_count,
        exact_turn_match_rate=_round_or_none(exact_turn_match_count / turns_total),
        latency_ms=round(elapsed_ms, 2),
        tool_correctness_score=float(tool_score),
        step_efficiency_score=float(step_score),
        task_completion_score=float(task_score),
        final_output=turn_outputs[-1] if turn_outputs else "",
        error=None,
    )


def _summarize_conversation_model(
    *,
    requested_model_id: str,
    resolved_model_id: str,
    results: list[ConversationBenchmarkResult],
) -> ConversationModelBenchmarkSummary:
    completed = [result for result in results if result.error is None]
    failures = [result for result in results if result.error is not None]

    avg_exact_turn = _avg(
        [
            result.exact_turn_match_rate
            for result in completed
            if result.exact_turn_match_rate is not None
        ]
    )
    avg_tool = _avg(
        [
            result.tool_correctness_score
            for result in completed
            if result.tool_correctness_score is not None
        ]
    )
    avg_step = _avg(
        [
            result.step_efficiency_score
            for result in completed
            if result.step_efficiency_score is not None
        ]
    )
    avg_task = _avg(
        [
            result.task_completion_score
            for result in completed
            if result.task_completion_score is not None
        ]
    )
    avg_latency = _avg([result.latency_ms for result in completed])

    composite = None
    if (
        avg_exact_turn is not None
        and avg_tool is not None
        and avg_step is not None
        and avg_task is not None
    ):
        composite = (
            0.35 * avg_exact_turn
            + 0.30 * avg_tool
            + 0.20 * avg_task
            + 0.15 * avg_step
        )

    status = "ok" if completed else "failed"
    return ConversationModelBenchmarkSummary(
        requested_model_id=requested_model_id,
        model_id=resolved_model_id,
        conversations_total=len(results),
        conversations_completed=len(completed),
        conversations_failed=len(failures),
        avg_exact_turn_match_rate=_round_or_none(avg_exact_turn),
        avg_tool_correctness=_round_or_none(avg_tool),
        avg_step_efficiency=_round_or_none(avg_step),
        avg_task_completion=_round_or_none(avg_task),
        avg_latency_ms=_round_or_none(avg_latency, digits=2),
        composite_score=_round_or_none(composite),
        status=status,
        error=failures[0].error if failures and not completed else None,
        failed_conversations=tuple(result.conversation_name for result in failures),
    )


def _rank_conversation_summaries(
    summaries: list[ConversationModelBenchmarkSummary],
) -> list[ConversationModelBenchmarkSummary]:
    def sort_key(summary: ConversationModelBenchmarkSummary) -> tuple[int, float, float, float]:
        if summary.status != "ok":
            return (1, 0.0, 0.0, float("inf"))
        return (
            0,
            -(summary.avg_exact_turn_match_rate or 0.0),
            -(summary.avg_task_completion or 0.0),
            summary.avg_latency_ms or float("inf"),
        )

    return sorted(summaries, key=sort_key)


def _run_single_scenario(
    notes_manager,
    scenario: ToolScenario,
    *,
    adapter: _LocalCandidateAgentAdapter,
    requested_model_id: str,
    tool_metric: ToolCorrectnessMetric,
    step_metric: StepEfficiencyMetric,
    task_metric: TaskCompletionMetric,
) -> ScenarioBenchmarkResult:
    start = time.perf_counter()
    try:
        _reset_and_prefill(notes_manager, scenario)
        model_reply = notes_manager.note_workflow._agent._agent.run(scenario.user_message_pl)
    except Exception as error:  # pragma: no cover - network/provider failures are environment-dependent
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ScenarioBenchmarkResult(
            scenario_name=scenario.name,
            model_id=adapter._model_id,
            requested_model_id=requested_model_id,
            latency_ms=round(elapsed_ms, 2),
            raw_model_output=adapter.last_response,
            final_output=None,
            called_tool_name=None,
            called_parameters={},
            expected_tool_name=scenario.tool_name,
            expected_parameters=scenario.parameters,
            exact_tool_match=False,
            tool_correctness_score=None,
            step_efficiency_score=None,
            task_completion_score=None,
            error=str(error),
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    final_output = model_reply.content or ""
    called_tool_name: str | None = None
    called_parameters: dict[str, Any] = {}
    if model_reply.tool_call is not None:
        called_tool_name, called_parameters = model_reply.tool_call

    exact_match = (
        called_tool_name == scenario.tool_name
        and called_parameters == scenario.parameters
    )

    test_case = _build_tool_test_case(
        scenario=scenario,
        final_output=final_output,
        called_tool_name=called_tool_name,
        called_parameters=called_parameters,
    )
    test_case._trace_dict = _build_trace(
        scenario=scenario,
        final_output=final_output,
        called_tool_name=called_tool_name,
        called_parameters=called_parameters,
    )

    tool_score = tool_metric.measure(test_case)
    step_score = step_metric.measure(test_case)
    task_score = task_metric.measure(test_case)

    return ScenarioBenchmarkResult(
        scenario_name=scenario.name,
        model_id=adapter._model_id,
        requested_model_id=requested_model_id,
        latency_ms=round(elapsed_ms, 2),
        raw_model_output=adapter.last_response,
        final_output=final_output,
        called_tool_name=called_tool_name,
        called_parameters=called_parameters,
        expected_tool_name=scenario.tool_name,
        expected_parameters=scenario.parameters,
        exact_tool_match=exact_match,
        tool_correctness_score=float(tool_score),
        step_efficiency_score=float(step_score),
        task_completion_score=float(task_score),
        error=None,
    )


def _summarize_model(
    requested_model_id: str,
    resolved_model_id: str,
    scenario_results: list[ScenarioBenchmarkResult],
) -> ModelBenchmarkSummary:
    completed = [result for result in scenario_results if result.error is None]
    failures = [result for result in scenario_results if result.error is not None]

    exact_match_count = sum(1 for result in completed if result.exact_tool_match)
    exact_match_rate = (
        exact_match_count / len(completed)
        if completed
        else None
    )

    avg_tool = _avg(
        [
            result.tool_correctness_score
            for result in completed
            if result.tool_correctness_score is not None
        ]
    )
    avg_step = _avg(
        [
            result.step_efficiency_score
            for result in completed
            if result.step_efficiency_score is not None
        ]
    )
    avg_task = _avg(
        [
            result.task_completion_score
            for result in completed
            if result.task_completion_score is not None
        ]
    )
    avg_latency = _avg([result.latency_ms for result in completed])

    composite = None
    if exact_match_rate is not None and avg_task is not None and avg_step is not None:
        composite = (
            0.60 * exact_match_rate
            + 0.25 * avg_task
            + 0.15 * avg_step
        )

    status = "ok" if completed else "failed"
    error = failures[0].error if failures and not completed else None
    failed_scenarios = tuple(result.scenario_name for result in failures)

    return ModelBenchmarkSummary(
        requested_model_id=requested_model_id,
        model_id=resolved_model_id,
        scenarios_total=len(scenario_results),
        scenarios_completed=len(completed),
        scenarios_failed=len(failures),
        exact_tool_match_count=exact_match_count,
        exact_tool_match_rate=_round_or_none(exact_match_rate),
        avg_tool_correctness=_round_or_none(avg_tool),
        avg_step_efficiency=_round_or_none(avg_step),
        avg_task_completion=_round_or_none(avg_task),
        avg_latency_ms=_round_or_none(avg_latency, digits=2),
        composite_score=_round_or_none(composite),
        status=status,
        error=error,
        failed_scenarios=failed_scenarios,
    )


def _rank_summaries(summaries: list[ModelBenchmarkSummary]) -> list[ModelBenchmarkSummary]:
    def sort_key(summary: ModelBenchmarkSummary) -> tuple[int, float, float, float]:
        # Failed models last. Higher scores first. Lower latency breaks ties.
        if summary.status != "ok":
            return (1, 0.0, 0.0, float("inf"))
        return (
            0,
            -(summary.exact_tool_match_rate or 0.0),
            -(summary.avg_task_completion or 0.0),
            summary.avg_latency_ms or float("inf"),
        )

    return sorted(summaries, key=sort_key)


def _default_report_path() -> Path:
    report_env = os.getenv("NOTES_OPENROUTER_REPORT_PATH")
    if report_env and report_env.strip():
        return Path(report_env).expanduser()
    return Path("packages/evaluation/src/evaluation/notes_openrouter_benchmark_report.json")


def _make_report_payload(
    judge_model_id: str,
    scenarios: tuple[ToolScenario, ...],
    summaries: list[ModelBenchmarkSummary],
    detailed_results: dict[str, list[ScenarioBenchmarkResult]],
) -> dict[str, Any]:
    ranked = _rank_summaries(summaries)
    top_models = [summary.model_id for summary in ranked if summary.status == "ok"][:2]
    return {
        "judge_model_id": judge_model_id,
        "scenarios": [scenario.name for scenario in scenarios],
        "top_models": top_models,
        "full_rank_list": _build_notes_full_rank_list(ranked),
        "ranked_summaries": [asdict(summary) for summary in ranked],
        "per_model_results": {
            model_id: [asdict(result) for result in results]
            for model_id, results in detailed_results.items()
        },
    }


def _selected_prompt_cases() -> tuple[PromptStyleCase, ...]:
    selected_names_raw = os.getenv("NOTES_OPENROUTER_PROMPT_CASES", "")
    selected_names = {
        name.strip()
        for name in selected_names_raw.split(",")
        if name.strip()
    }
    limit = _parse_int_env("NOTES_OPENROUTER_MAX_PROMPT_CASES")

    cases = PROMPT_STYLE_CASES
    if selected_names:
        cases = tuple(case for case in cases if case.name in selected_names)
    if limit is not None and limit >= 0:
        cases = cases[:limit]
    return cases


def _default_prompt_cases_report_path() -> Path:
    report_env = os.getenv("NOTES_OPENROUTER_PROMPT_REPORT_PATH")
    if report_env and report_env.strip():
        return Path(report_env).expanduser()
    return Path(
        "packages/evaluation/src/evaluation/notes_openrouter_prompt_cases_report.json"
    )


def _default_conversation_report_path() -> Path:
    report_env = os.getenv("NOTES_OPENROUTER_CONVERSATION_REPORT_PATH")
    if report_env and report_env.strip():
        return Path(report_env).expanduser()
    return Path(
        "packages/evaluation/src/evaluation/notes_openrouter_conversation_benchmark_report.json"
    )


def _run_prompt_style_case(
    *,
    adapter: _LocalCandidateAgentAdapter,
    requested_model_id: str,
    case: PromptStyleCase,
    g_eval_metric: GEval,
    task_metric: TaskCompletionMetric,
) -> dict[str, Any]:
    benchmark_input = _build_prompt_style_case_input(case)
    start = time.perf_counter()
    try:
        actual_output = adapter.response(benchmark_input)
    except Exception as error:  # pragma: no cover - environment/model-dependent
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "case_name": case.name,
            "prompt_name": str(case.prompt_name),
            "requested_model_id": requested_model_id,
            "model_id": adapter._model_id,
            "latency_ms": round(elapsed_ms, 2),
            "actual_output": adapter.last_response,
            "g_eval_score": None,
            "task_completion_score": None,
            "error": str(error),
        }

    elapsed_ms = (time.perf_counter() - start) * 1000
    test_case = LLMTestCase(
        input=benchmark_input,
        actual_output=actual_output,
        expected_output=case.expected_output,
    )
    g_eval_score = g_eval_metric.measure(test_case)
    task_score = task_metric.measure(test_case)
    return {
        "case_name": case.name,
        "prompt_name": str(case.prompt_name),
        "requested_model_id": requested_model_id,
        "model_id": adapter._model_id,
        "latency_ms": round(elapsed_ms, 2),
        "actual_output": actual_output,
        "g_eval_score": float(g_eval_score),
        "task_completion_score": float(task_score),
        "error": None,
    }


def _rank_prompt_case_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[int, float, float, float]:
        if item["status"] != "ok":
            return (1, 0.0, 0.0, float("inf"))
        return (
            0,
            -(item.get("avg_g_eval") or 0.0),
            -(item.get("avg_task_completion") or 0.0),
            item.get("avg_latency_ms") or float("inf"),
        )

    return sorted(items, key=sort_key)


def _failed_scenario_results(
    *,
    scenarios: tuple[ToolScenario, ...],
    requested_model_id: str,
    model_id: str,
    error: str,
) -> list[ScenarioBenchmarkResult]:
    return [
        ScenarioBenchmarkResult(
            scenario_name=scenario.name,
            model_id=model_id,
            requested_model_id=requested_model_id,
            latency_ms=0.0,
            raw_model_output=None,
            final_output=None,
            called_tool_name=None,
            called_parameters={},
            expected_tool_name=scenario.tool_name,
            expected_parameters=scenario.parameters,
            exact_tool_match=False,
            tool_correctness_score=None,
            step_efficiency_score=None,
            task_completion_score=None,
            error=error,
        )
        for scenario in scenarios
    ]


def _failed_prompt_case_results(
    *,
    cases: tuple[PromptStyleCase, ...],
    requested_model_id: str,
    model_id: str,
    error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "case_name": case.name,
            "prompt_name": str(case.prompt_name),
            "requested_model_id": requested_model_id,
            "model_id": model_id,
            "latency_ms": 0.0,
            "actual_output": None,
            "g_eval_score": None,
            "task_completion_score": None,
            "error": error,
        }
        for case in cases
    ]


def _summarize_prompt_case_model(
    *,
    requested_model_id: str,
    model_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    ok_results = [result for result in results if result["error"] is None]
    failed_results = [result for result in results if result["error"] is not None]
    avg_g_eval = _avg(
        [
            result["g_eval_score"]
            for result in ok_results
            if result["g_eval_score"] is not None
        ]
    )
    avg_task = _avg(
        [
            result["task_completion_score"]
            for result in ok_results
            if result["task_completion_score"] is not None
        ]
    )
    avg_latency = _avg([result["latency_ms"] for result in ok_results])
    composite = None
    if avg_g_eval is not None and avg_task is not None:
        composite = 0.7 * avg_g_eval + 0.3 * avg_task

    return {
        "requested_model_id": requested_model_id,
        "model_id": model_id,
        "cases_total": len(results),
        "cases_completed": len(ok_results),
        "cases_failed": len(failed_results),
        "avg_g_eval": _round_or_none(avg_g_eval),
        "avg_task_completion": _round_or_none(avg_task),
        "avg_latency_ms": _round_or_none(avg_latency, digits=2),
        "composite_score": _round_or_none(composite),
        "status": "ok" if ok_results else "failed",
        "failed_cases": [result["case_name"] for result in failed_results],
        "error": (
            failed_results[0]["error"]
            if failed_results and not ok_results
            else None
        ),
    }


def _failed_conversation_results(
    *,
    conversations: tuple[ConversationScenario, ...],
    requested_model_id: str,
    model_id: str,
    error: str,
) -> list[ConversationBenchmarkResult]:
    return [
        ConversationBenchmarkResult(
            conversation_name=conversation.name,
            requested_model_id=requested_model_id,
            model_id=model_id,
            turns_total=len(conversation.steps),
            turns_completed=0,
            exact_turn_match_count=0,
            exact_turn_match_rate=None,
            latency_ms=0.0,
            tool_correctness_score=None,
            step_efficiency_score=None,
            task_completion_score=None,
            final_output=None,
            error=error,
        )
        for conversation in conversations
    ]


def test_notes_candidate_models_benchmark(notes_manager, monkeypatch) -> None:
    scenarios = _selected_scenarios()
    assert scenarios, "Brak scenariuszy do uruchomienia benchmarku."

    judge_model_id, candidate_models, write_report = _benchmark_settings()
    judge_llm = _build_judge_llm(judge_model_id)
    summaries: list[ModelBenchmarkSummary] = []
    detailed_results: dict[str, list[ScenarioBenchmarkResult]] = {}

    for requested_model_id, model_id in candidate_models:
        adapter = _LocalCandidateAgentAdapter(model_id=model_id)
        _install_model_for_notes_agent(monkeypatch, notes_manager, adapter)
        load_error = adapter.probe_load_error()
        if load_error is not None:
            scenario_results = _failed_scenario_results(
                scenarios=scenarios,
                requested_model_id=requested_model_id,
                model_id=model_id,
                error=load_error,
            )
            detailed_results[model_id] = scenario_results
            summaries.append(
                _summarize_model(
                    requested_model_id=requested_model_id,
                    resolved_model_id=model_id,
                    scenario_results=scenario_results,
                )
            )
            continue

        tool_metric = _build_tool_correctness_metric(judge_llm)
        step_metric = _build_step_efficiency_metric(judge_llm, threshold=0.9)
        task_metric = _build_task_completion_metric(judge_llm, threshold=0.5)

        scenario_results: list[ScenarioBenchmarkResult] = []
        for scenario in scenarios:
            scenario_results.append(
                _run_single_scenario(
                notes_manager=notes_manager,
                scenario=scenario,
                adapter=adapter,
                requested_model_id=requested_model_id,
                tool_metric=tool_metric,
                step_metric=step_metric,
                task_metric=task_metric,
                )
            )

        detailed_results[model_id] = scenario_results
        summaries.append(
            _summarize_model(
                requested_model_id=requested_model_id,
                resolved_model_id=model_id,
                scenario_results=scenario_results,
            )
        )

    report_payload = _make_report_payload(
        judge_model_id=judge_model_id,
        scenarios=scenarios,
        summaries=summaries,
        detailed_results=detailed_results,
    )

    _write_json_report(
        report_payload,
        write_report=write_report,
        report_path=_default_report_path(),
    )

    successful_summaries = [
        summary for summary in summaries if summary.status == "ok"
    ]
    assert successful_summaries, (
        "Żaden model nie został poprawnie oceniony. "
        "Sprawdź lokalne ładowanie modeli MLX/HF (np. zgodność z `mlx_lm`) oraz "
        "klucz/konto OpenRouter dla modelu sędziego."
    )

    best = _rank_summaries(summaries)[0]
    assert best.status == "ok"
    assert best.exact_tool_match_rate is None or 0.0 <= best.exact_tool_match_rate <= 1.0


def test_prompt_style_cases_with_judge() -> None:
    if not _bool_env("RUN_OPENROUTER_PROMPT_CASES", default=False):
        pytest.skip(
            "Prompt-style benchmark is optional. Set RUN_OPENROUTER_PROMPT_CASES=1 to run."
        )

    cases = _selected_prompt_cases()
    assert cases, "Brak prompt-case do uruchomienia benchmarku."

    judge_model_id, candidate_models, write_report = _benchmark_settings()
    judge_llm = _build_judge_llm(judge_model_id)

    summaries: list[dict[str, Any]] = []
    per_model_results: dict[str, list[dict[str, Any]]] = {}

    for requested_model_id, model_id in candidate_models:
        adapter = _LocalCandidateAgentAdapter(model_id=model_id)
        load_error = adapter.probe_load_error()
        if load_error is not None:
            results = _failed_prompt_case_results(
                cases=cases,
                requested_model_id=requested_model_id,
                model_id=model_id,
                error=load_error,
            )
            per_model_results[model_id] = results
            summaries.append(
                _summarize_prompt_case_model(
                    requested_model_id=requested_model_id,
                    model_id=model_id,
                    results=results,
                )
            )
            continue

        g_eval_metric = GEval(
            name="PromptCase Fidelity",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            criteria=PROMPT_STYLE_LLM_AS_JUDGE_CRITERIA,
            threshold=0.5,
            async_mode=False,
            model=judge_llm,
        )
        task_metric = _build_task_completion_metric(judge_llm, threshold=0.5)

        results = [
            _run_prompt_style_case(
                adapter=adapter,
                requested_model_id=requested_model_id,
                case=case,
                g_eval_metric=g_eval_metric,
                task_metric=task_metric,
            )
            for case in cases
        ]
        per_model_results[model_id] = results
        summaries.append(
            _summarize_prompt_case_model(
                requested_model_id=requested_model_id,
                model_id=model_id,
                results=results,
            )
        )

    ranked = _rank_prompt_case_summaries(summaries)
    payload = {
        "judge_model_id": judge_model_id,
        "prompt_cases": [case.name for case in cases],
        "top_models": [item["model_id"] for item in ranked if item["status"] == "ok"][:2],
        "full_rank_list": _build_prompt_case_full_rank_list(ranked),
        "ranked_summaries": ranked,
        "per_model_results": per_model_results,
    }

    _write_json_report(
        payload,
        write_report=write_report,
        report_path=_default_prompt_cases_report_path(),
    )

    assert any(item["status"] == "ok" for item in ranked), (
        "Żaden model nie został oceniony w prompt-case benchmarku. "
        "Sprawdź lokalne ładowanie modeli oraz OpenRouter judge."
    )


def test_notes_conversations_with_judge(notes_manager, monkeypatch) -> None:
    if not _bool_env("RUN_OPENROUTER_CONVERSATION_BENCHMARKS", default=False):
        pytest.skip(
            "Conversation benchmark is optional. Set RUN_OPENROUTER_CONVERSATION_BENCHMARKS=1 to run."
        )

    conversations = _selected_conversations()
    assert conversations, "Brak konwersacji do uruchomienia benchmarku."

    judge_model_id, candidate_models, write_report = _benchmark_settings()
    judge_llm = _build_judge_llm(judge_model_id)

    summaries: list[ConversationModelBenchmarkSummary] = []
    per_model_results: dict[str, list[dict[str, Any]]] = {}

    for requested_model_id, model_id in candidate_models:
        adapter = _LocalCandidateAgentAdapter(model_id=model_id)
        _install_model_for_notes_agent(monkeypatch, notes_manager, adapter)
        load_error = adapter.probe_load_error()
        if load_error is not None:
            failed_results = _failed_conversation_results(
                conversations=conversations,
                requested_model_id=requested_model_id,
                model_id=model_id,
                error=load_error,
            )
            per_model_results[model_id] = [asdict(result) for result in failed_results]
            summaries.append(
                _summarize_conversation_model(
                    requested_model_id=requested_model_id,
                    resolved_model_id=model_id,
                    results=failed_results,
                )
            )
            continue

        tool_metric = _build_tool_correctness_metric(judge_llm)
        step_metric = _build_step_efficiency_metric(judge_llm, threshold=0.5)
        task_metric = _build_task_completion_metric(judge_llm, threshold=0.5)

        results = [
            _run_conversation(
                notes_manager,
                conversation,
                adapter=adapter,
                requested_model_id=requested_model_id,
                tool_metric=tool_metric,
                step_metric=step_metric,
                task_metric=task_metric,
            )
            for conversation in conversations
        ]
        per_model_results[model_id] = [asdict(result) for result in results]
        summaries.append(
            _summarize_conversation_model(
                requested_model_id=requested_model_id,
                resolved_model_id=model_id,
                results=results,
            )
        )

    ranked = _rank_conversation_summaries(summaries)
    payload = {
        "judge_model_id": judge_model_id,
        "conversation_count": len(conversations),
        "conversations": [conversation.name for conversation in conversations],
        "top_models": [summary.model_id for summary in ranked if summary.status == "ok"][:2],
        "full_rank_list": _build_conversation_full_rank_list(ranked),
        "ranked_summaries": [asdict(summary) for summary in ranked],
        "per_model_results": per_model_results,
    }

    _write_json_report(
        payload,
        write_report=write_report,
        report_path=_default_conversation_report_path(),
    )

    assert any(summary.status == "ok" for summary in ranked), (
        "Żaden model nie został oceniony w conversation benchmarku. "
        "Sprawdź lokalne ładowanie modeli oraz OpenRouter judge."
    )
