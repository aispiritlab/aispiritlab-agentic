PROJECT_NAME = ai-spirit-agent

.DEFAULT_GOAL := help

#################################################################################
# Development                                                                   #
#################################################################################

.PHONY: install
install: ## Install all dependencies (uv sync)
	uv sync

.PHONY: clean
clean: ## Delete all compiled Python files and caches
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

.PHONY: lint
lint: ## Check code style with ruff
	uv run ruff format --check
	uv run ruff check

.PHONY: format
format: ## Format and fix code with ruff
	uv run ruff check --fix
	uv run ruff format

.PHONY: mypy
mypy: ## Run mypy type checking
	uv run mypy .

#################################################################################
# Testing                                                                       #
#################################################################################

.PHONY: test
test: ## Run all unit tests
	uv run pytest

.PHONY: test-pa
test-pa: ## Run personal_assistant tests only
	uv run pytest packages/personal_assistant/tests/ -v

.PHONY: test-runtime
test-runtime: ## Run agentic_runtime framework tests only
	uv run pytest packages/agentic_runtime/tests/ -v

.PHONY: test-agentic
test-agentic: ## Run agentic SDK tests only
	uv run pytest packages/agentic/tests/ -v

.PHONY: test-e2e
test-e2e: ## Run workflow smoke tests (requires live model)
	RUN_WORKFLOW_SMOKE_TESTS=1 uv run pytest packages/personal_assistant/tests/e2e/ -m workflow_smoke -v

.PHONY: test-e2e-live
test-e2e-live: ## Run runtime-first live end-to-end tests
	RUN_AGENT_E2E_LIVE=1 uv run pytest packages/personal_assistant/tests/e2e_live/ -m agent_e2e_live -v

#################################################################################
# Applications                                                                  #
#################################################################################

.PHONY: chat
chat: ## Launch Personal Assistant Gradio chat interface
	uv run personal-assistant

.PHONY: cli
cli: ## Launch CLI in interactive mode
	uv run ai-spirit-cli

.PHONY: registry
registry: ## Launch prompt registry
	uv run registry

.PHONY: generate-kb
generate-kb: ## Generate knowledge base
	uv run knowledge-base-generate

.PHONY: pull-model
pull-model: ## Pull default LLM model
	ollama pull lfm2.5-thinking

#################################################################################
# Infrastructure                                                                #
#################################################################################

.PHONY: mlflow
mlflow: ## Start MLflow with docker-compose
	docker compose -f containers/docker-compose.yml up

.PHONY: mlflow-ui
mlflow-ui: ## Start MLflow UI without Docker
	MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="http://0.0.0.0:5001,http://localhost:5001,http://127.0.0.1:5001" \
		uv run mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///data/mlflow.db

.PHONY: lab6-up
lab6-up: ## Start distributed Lab 6 stack
	docker compose -f containers/docker-compose.lab6.yml up --build

.PHONY: lab6-down
lab6-down: ## Stop distributed Lab 6 stack
	docker compose -f containers/docker-compose.lab6.yml down

#################################################################################
# Data                                                                          #
#################################################################################

.PHONY: flow-php-qa-dataset
flow-php-qa-dataset: ## Generate Flow PHP Q&A + reasoning dataset
	uv run --project packages/dataloader flow-php-qa-dataset write-configs
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/core-operations-basic.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/core-operations-reasoning.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/aggregations-joins-basic.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/aggregations-joins-reasoning.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/adapters-basic.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/adapters-reasoning.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/infrastructure-basic.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/infrastructure-reasoning.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/ecosystem-basic.yaml
	uv run deepfabric generate packages/dataloader/data/flow_php_qa/ecosystem-reasoning.yaml
	uv run --project packages/dataloader flow-php-qa-dataset merge --dedup --shuffle

#################################################################################
# Help                                                                          #
#################################################################################

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
