"""Flow PHP Q&A + Reasoning dataset generation via DeepFabric.

Reads Flow PHP documentation, groups files by topic cluster, and generates
DeepFabric YAML configs for basic Q&A and chain-of-thought reasoning datasets.

Commands:
    write-configs  Generate 10 YAML configs (5 clusters × 2 types)
    merge          Concatenate generated JSONL outputs into a single file
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR_NAME: Final[str] = "flow_php_qa"

CLUSTERS: Final[dict[str, dict[str, object]]] = {
    "core-operations": {
        "doc_paths": [
            "quick-start.md",
            "introduction.md",
            "components/core/transformations.md",
            "components/core/filter.md",
            "components/core/map.md",
            "components/core/select-drop.md",
            "components/core/rename.md",
            "components/core/sort.md",
            "components/core/limit.md",
            "components/core/offset.md",
            "components/core/pivot.md",
            "components/core/execution-mode.md",
            "components/core/save-mode.md",
            "components/core/data-retrieval.md",
            "components/core/data-manipulation.md",
            "components/core/building-blocks.md",
            "components/core/display.md",
            "components/core/until.md",
        ],
        "basic_samples": 2571,
        "reasoning_samples": 1243,
        "topic_prompt": (
            "Flow PHP core DataFrame operations: creating data frames, "
            "reading data, transformations (filter, map, select, drop, rename), "
            "sorting, limiting rows, offset/pagination, pivoting data, "
            "execution modes (strict/lenient), save modes (append/overwrite), "
            "data retrieval methods (get, getEach, fetch, count, forEach), "
            "data manipulation (casting, withEntry, dropDuplicates), "
            "building blocks (entries, rows), displaying output as ASCII tables, "
            "and the until/stop signal pattern."
        ),
    },
    "aggregations-joins": {
        "doc_paths": [
            "components/core/group-by.md",
            "components/core/aggregations.md",
            "components/core/join.md",
            "components/core/window-functions.md",
            "components/core/batch-processing.md",
            "components/core/partitioning.md",
        ],
        "basic_samples": 1993,
        "reasoning_samples": 1007,
        "topic_prompt": (
            "Flow PHP aggregation and join operations: groupBy with aggregation "
            "functions (COUNT, SUM, AVERAGE, MIN, MAX, COLLECT, COLLECT_UNIQUE, "
            "FIRST, LAST, STRING_AGG), hash joins (left, inner, right, left_anti), "
            "window functions (ROW_NUMBER, RANK, DENSE_RANK, windowed SUM/AVG/COUNT), "
            "batch processing (batchSize, batchBy, collect), "
            "and partitioning (partitionBy, nested partitions, partition pruning)."
        ),
    },
    "adapters": {
        "doc_paths": [
            "components/adapters/csv.md",
            "components/adapters/json.md",
            "components/adapters/parquet.md",
            "components/adapters/excel.md",
            "components/adapters/postgresql.md",
            "components/adapters/elasticsearch.md",
            "components/adapters/http.md",
            "components/adapters/doctrine.md",
            "components/adapters/google-sheet.md",
            "components/adapters/avro.md",
            "components/adapters/xml.md",
            "components/adapters/text.md",
            "components/adapters/logger.md",
            "components/adapters/chartjs.md",
        ],
        "basic_samples": 2571,
        "reasoning_samples": 1243,
        "topic_prompt": (
            "Flow PHP data adapters for reading and writing: CSV (extractor/loader), "
            "JSON (JSONMachine extractor, JSON lines), Parquet (columnar storage), "
            "Excel (XLSX/ODS with sheet selection and styling), "
            "PostgreSQL (server-side cursors, keyset pagination, INSERT/UPDATE/UPSERT), "
            "Elasticsearch (search extractor, bulk index loader), "
            "HTTP (PSR-18 dynamic/static extractors), "
            "Doctrine (DbalQueryExtractor, bulk operations, schema conversion), "
            "Google Sheets (extractor with auth), Avro, XML (xpath extraction), "
            "Text (line-by-line), Logger (PSR logger loader), "
            "and ChartJS (bar/line/pie chart visualization)."
        ),
    },
    "infrastructure": {
        "doc_paths": [
            "components/core/schema.md",
            "components/core/constraints.md",
            "components/core/error-handling.md",
            "components/core/retry.md",
            "components/core/caching.md",
            "components/core/telemetry.md",
            "components/libs/filesystem.md",
            "installation.md",
        ],
        "basic_samples": 1500,
        "reasoning_samples": 750,
        "topic_prompt": (
            "Flow PHP infrastructure: schema definition and validation "
            "(StrictValidator, SelectiveValidator), constraints (unique, multi-column, custom), "
            "error handling (ErrorHandler, throw/skipRows), "
            "retry strategies (AnyThrowable, Fixed/Linear/Exponential/Jitter delays), "
            "caching (InMemory, LocalFilesystem, PSRSimpleCache), "
            "OpenTelemetry integration (console/OTLP exporters, spans, metrics, logs), "
            "unified filesystem (local/remote, block storage, Azure/AWS), "
            "and installation via Composer with monorepo package structure."
        ),
    },
    "ecosystem": {
        "doc_paths": [
            "components/bridges/filesystem-async-aws-bridge.md",
            "components/bridges/filesystem-azure-bridge.md",
            "components/bridges/monolog-http-bridge.md",
            "components/bridges/monolog-telemetry-bridge.md",
            "components/bridges/openapi-specification-bridge.md",
            "components/bridges/psr18-telemetry-bridge.md",
            "components/bridges/psr7-telemetry-bridge.md",
            "components/bridges/symfony-http-foundation-bridge.md",
            "components/bridges/symfony-http-foundation-telemetry-bridge.md",
            "components/bridges/symfony-telemetry-bundle.md",
            "components/bridges/telemetry-otlp-bridge.md",
            "components/libs/array-dot.md",
            "components/libs/doctrine-dbal-bulk.md",
            "components/libs/postgresql.md",
            "components/libs/types.md",
        ],
        "basic_samples": 1372,
        "reasoning_samples": 750,
        "topic_prompt": (
            "Flow PHP ecosystem bridges and libraries: "
            "AWS S3 filesystem bridge, Azure Blob Storage bridge, "
            "Monolog bridges (HTTP normalization, telemetry forwarding), "
            "OpenAPI specification bridge (schema-to-OpenAPI conversion), "
            "PSR-7/PSR-18 telemetry bridges (context propagation, HTTP tracing), "
            "Symfony bridges (FlowStreamedResponse, HttpFoundation telemetry, bundle), "
            "OTLP transport bridge (Curl/HTTP/gRPC, JSON/Protobuf serialization), "
            "array-dot library (dot notation with ?, *, ?*, {} wildcards), "
            "Doctrine DBAL Bulk (bulk insert/update/delete), "
            "PostgreSQL library (SQL parser, query builder, client with cursors/transactions), "
            "and Types library (type narrowing, casting, lists, maps, structures)."
        ),
    },
}

SYSTEM_PROMPT_BASIC: Final[str] = """\
You are a Flow PHP expert. Generate practical Q&A pairs about Flow PHP.
Answer with: 1) A concise explanation (2-3 sentences max)
2) A working PHP code example with proper imports
3) How to run it (composer require, php command)

ALWAYS use:
- declare(strict_types=1);
- use function Flow\\ETL\\DSL\\{...};
- require __DIR__ . '/vendor/autoload.php';
- data_frame()->read()->...->run() pattern

NEVER use Python concepts (pandas, pydantic, dataframe as Python).\
"""

SYSTEM_PROMPT_OUTPUT: Final[str] = """\
You are a Flow PHP expert. Provide concise, practical answers with \
working code examples. Use proper PHP 8.1+ syntax with Flow PHP DSL.\
"""

LLM_TOPIC: Final[dict[str, str]] = {
    "provider": "openrouter",
    "model": "openrouter/hunter-alpha",
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": "0.2",
    "max_tokens": "1400",
}

LLM_GENERATION: Final[dict[str, str]] = {
    "provider": "openrouter",
    "model": "openrouter/hunter-alpha",
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": "0.3",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    doc_paths: list[str]
    basic_samples: int
    reasoning_samples: int
    topic_prompt: str


def _parse_clusters() -> list[ClusterConfig]:
    configs: list[ClusterConfig] = []
    for name, raw in CLUSTERS.items():
        configs.append(
            ClusterConfig(
                name=name,
                doc_paths=list(raw["doc_paths"]),  # type: ignore[arg-type]
                basic_samples=int(raw["basic_samples"]),  # type: ignore[arg-type]
                reasoning_samples=int(raw["reasoning_samples"]),  # type: ignore[arg-type]
                topic_prompt=str(raw["topic_prompt"]),
            )
        )
    return configs


def _read_doc_snippets(doc_root: Path, paths: list[str]) -> str:
    """Read documentation files and return concatenated content."""
    snippets: list[str] = []
    for rel in paths:
        full = doc_root / rel
        if not full.exists():
            print(f"  [warn] doc not found: {full}", file=sys.stderr)
            continue
        content = full.read_text(encoding="utf-8")
        # Truncate very large docs to keep YAML manageable
        if len(content) > 6000:
            content = content[:6000] + "\n... (truncated)"
        snippets.append(f"--- {rel} ---\n{content}")
    return "\n\n".join(snippets)


def _build_config(
    cluster: ClusterConfig,
    config_type: str,
    doc_context: str,
    output_dir: Path,
) -> dict[str, object]:
    """Build a single DeepFabric YAML config dict."""
    is_reasoning = config_type == "reasoning"
    num_samples = cluster.reasoning_samples if is_reasoning else cluster.basic_samples
    suffix = "reasoning" if is_reasoning else "basic"

    system_prompt = SYSTEM_PROMPT_BASIC + f"\n\n<doc-context>\n{doc_context}\n</doc-context>"

    if is_reasoning:
        instructions = (
            f"Generate a multi-step problem about {cluster.name.replace('-', ' ')}. "
            "The answer should show step-by-step reasoning about how to build "
            "an ETL pipeline, choose the right approach, or debug an issue. "
            "Include working code in the final answer."
        )
        conversation = {"type": "cot", "reasoning_style": "freetext"}
    else:
        instructions = (
            f"Generate a question about {cluster.name.replace('-', ' ')} and provide "
            "a concise answer with a working code example. Focus on practical "
            '"how do I..." questions. Keep answers short - explanation + code + run instructions.'
        )
        conversation = {"type": "basic"}

    return {
        "llm": {
            "provider": "openrouter",
            "model": "openrouter/hunter-alpha",
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.2,
            "max_tokens": 1400,
        },
        "topics": {
            "prompt": cluster.topic_prompt,
            "mode": "graph",
            "depth": 3,
            "degree": 3,
            "save_as": str(output_dir / f"{cluster.name}-{suffix}-topics.jsonl"),
            "llm": {
                "provider": "openrouter",
                "model": "openrouter/hunter-alpha",
                "base_url": "https://openrouter.ai/api/v1",
                "temperature": 0.2,
                "max_tokens": 1400,
            },
        },
        "generation": {
            "system_prompt": system_prompt,
            "instructions": instructions,
            "conversation": conversation,
            "llm": {
                "provider": "openrouter",
                "model": "openrouter/hunter-alpha",
                "base_url": "https://openrouter.ai/api/v1",
                "temperature": 0.3,
            },
        },
        "output": {
            "system_prompt": SYSTEM_PROMPT_OUTPUT,
            "include_system_message": True,
            "num_samples": num_samples,
            "batch_size": 4,
            "save_as": str(output_dir / f"{cluster.name}-{suffix}.jsonl"),
        },
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _default_repo_root() -> Path:
    env_val = os.environ.get("FLOW_REPO_ROOT")
    if env_val:
        return Path(env_val)
    return Path.home() / "Projects" / "flow"


def cmd_write_configs(args: argparse.Namespace) -> None:
    """Generate 10 DeepFabric YAML configs (5 clusters × basic + reasoning)."""
    repo_root = Path(args.repo_root)
    doc_root = repo_root / "documentation"

    if not doc_root.is_dir():
        print(f"Error: documentation directory not found at {doc_root}", file=sys.stderr)
        raise SystemExit(1)

    pkg_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = pkg_dir / "data" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = _parse_clusters()

    for cluster in clusters:
        print(f"Processing cluster: {cluster.name}")
        doc_context = _read_doc_snippets(doc_root, cluster.doc_paths)

        for config_type in ("basic", "reasoning"):
            cfg = _build_config(cluster, config_type, doc_context, output_dir)
            filename = f"{cluster.name}-{config_type}.yaml"
            out_path = output_dir / filename

            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

            print(f"  wrote {out_path}")

    print(f"\nDone — generated {len(clusters) * 2} configs in {output_dir}")


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge all generated JSONL files into a single training file."""
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = pkg_dir / "data" / OUTPUT_DIR_NAME

    if not data_dir.is_dir():
        print(f"Error: data directory not found at {data_dir}", file=sys.stderr)
        raise SystemExit(1)

    output_path = Path(args.output) if args.output else data_dir / "flow-php-qa-all.jsonl"

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    # Exclude topic files and the output file itself
    jsonl_files = [
        f for f in jsonl_files
        if "-topics.jsonl" not in f.name and f.name != output_path.name
    ]

    if not jsonl_files:
        print("No JSONL data files found to merge.", file=sys.stderr)
        raise SystemExit(1)

    seen: set[str] = set()
    rows: list[str] = []

    for jf in jsonl_files:
        print(f"  reading {jf.name}")
        with open(jf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if args.dedup:
                    if line in seen:
                        continue
                    seen.add(line)
                rows.append(line)

    if args.shuffle:
        random.shuffle(rows)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")

    print(f"Merged {len(rows)} samples from {len(jsonl_files)} files → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="flow-php-qa-dataset",
        description="Generate Flow PHP Q&A + reasoning datasets via DeepFabric.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # write-configs
    wc = sub.add_parser("write-configs", help="Generate DeepFabric YAML configs")
    wc.add_argument(
        "--repo-root",
        default=str(_default_repo_root()),
        help="Path to Flow PHP repo root (default: FLOW_REPO_ROOT env or ~/Projects/flow)",
    )

    # merge
    mg = sub.add_parser("merge", help="Merge generated JSONL files")
    mg.add_argument("--output", "-o", default=None, help="Output file path")
    mg.add_argument("--dedup", action="store_true", help="Remove duplicate rows")
    mg.add_argument("--shuffle", action="store_true", help="Shuffle output rows")

    args = parser.parse_args()

    if args.command == "write-configs":
        cmd_write_configs(args)
    elif args.command == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
