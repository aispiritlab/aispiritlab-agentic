# Resumable research agent

Ten przykład realizuje research internetowy jako trwałą maszynę stanów. Można
zatrzymać proces po planowaniu, po dowolnym zapytaniu albo podczas zapisu i
kontynuować go przez `resume`.

## Dwa źródła prawdy o rozłącznych odpowiedzialnościach

- SQLite Event Store przechowuje historię procesu: starty prób, ukończone kroki,
  błędy i pozycję workflow.
- Plik `*.research.json` przechowuje materiał badawczy: pytanie, zapytania,
  wyniki, URL-e i końcowe podsumowanie.

Wyniki wyszukiwania i tekst podsumowania nie są kopiowane do eventów. Agent do
kontynuacji potrzebuje zarówno historii procesu, jak i pliku badawczego.

Aktualizacja pliku używa protokołu odzyskiwania:

1. atomowy zapis pliku z operacją `pending`,
2. append eventu z optimistic concurrency,
3. oznaczenie operacji plikowej jako `committed`.

Po awarii między krokami 1–3 `resume` najpierw uzgadnia operację. Nie wykonuje
ponownie zakończonego wyszukiwania ani podsumowania.

## Uruchomienie

W `.env` ustaw `LANGSEARCH_API_KEY` oraz konfigurację modelu używaną przez
pozostałe warsztaty. Następnie:

```bash
uv run --package workshops python -m workshops.resumable_research start \
  "Jakie są najnowsze wzorce budowy niezawodnych agentów AI?"
```

Ograniczenie liczby zewnętrznych kroków pozwala zasymulować kontrolowaną pauzę:

```bash
uv run --package workshops python -m workshops.resumable_research start \
  --id agent-reliability --max-steps 2 \
  "Jakie są najnowsze wzorce budowy niezawodnych agentów AI?"

uv run --package workshops python -m workshops.resumable_research resume agent-reliability
```

Trwały dokument można obejrzeć bez uruchamiania modelu ani wyszukiwarki:

```bash
uv run --package workshops python -m workshops.resumable_research show agent-reliability
```

Domyślny workspace to `data/resumable-research`. Każda komenda obsługuje opcję
`--workspace`.
