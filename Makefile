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

.PHONY: typecheck
typecheck: ## Run ty type checking (reports; not yet a gate — see CI)
	-uv run ty check

.PHONY: mypy
mypy: typecheck ## Deprecated alias for `make typecheck`

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

.PHONY: test-resilience
test-resilience: ## Run distributed resilience tests (requires LLM server, Iggy optional)
	RUN_DISTRIBUTED_RESILIENCE=1 uv run pytest packages/agentic_runtime/tests/e2e_resilience/ -m distributed_resilience -v

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

.PHONY: env
env: ## Copy .env.example to .env (if missing)
	@test -f .env && echo ".env already exists — skipping." || (cp .env.example .env && echo ".env created from .env.example")

.PHONY: docker-build
docker-build: ## Build the root Docker image
	docker build -t $(PROJECT_NAME) .

.PHONY: mlflow
mlflow: ## Start MLflow stack (Postgres + MinIO + MLflow server)
	docker compose --project-directory . -f containers/docker-compose.yml up

.PHONY: mlflow-down
mlflow-down: ## Stop MLflow stack
	docker compose --project-directory . -f containers/docker-compose.yml down

.PHONY: mlflow-ui
mlflow-ui: ## Start MLflow UI without Docker
	MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="http://0.0.0.0:5001,http://localhost:5001,http://127.0.0.1:5001" \
		uv run mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///data/mlflow.db

.PHONY: mlflow-logs
mlflow-logs: ## Tail MLflow stack logs
	docker compose --project-directory . -f containers/docker-compose.yml logs -f

.PHONY: iggy
iggy: ## Start standalone Apache Iggy for local distributed dev
	# Every flag is load-bearing. Iggy's runtime is io_uring and the default
	# seccomp profile blocks it; "numa:auto" sharding fails inside a container
	# VM; and without the root credentials the server invents a password, logs
	# it once, and refuses every login.
	docker run --rm --name ai-spirit-iggy \
	  --security-opt seccomp=unconfined \
	  -e IGGY_TCP_ADDRESS=0.0.0.0:8090 \
	  -e IGGY_SYSTEM_SHARDING_CPU_ALLOCATION=2 \
	  -e IGGY_SYSTEM_SHARDING_PIN_CORES=false \
	  -e IGGY_ROOT_USERNAME=iggy \
	  -e IGGY_ROOT_PASSWORD=iggy \
	  -p 8090:8090 apache/iggy:0.9.0-edge.5

.PHONY: iggy-stop
iggy-stop: ## Stop standalone Iggy container
	docker stop ai-spirit-iggy

.PHONY: lab6-up
lab6-up: ## Start distributed Lab 6 stack
	docker compose --project-directory . -f containers/docker-compose.lab6.yml up --build

.PHONY: lab6-down
lab6-down: ## Stop distributed Lab 6 stack
	docker compose --project-directory . -f containers/docker-compose.lab6.yml down

.PHONY: lab6-logs
lab6-logs: ## Tail distributed Lab 6 stack logs
	docker compose --project-directory . -f containers/docker-compose.lab6.yml logs -f

.PHONY: lab6-restart
lab6-restart: ## Restart distributed Lab 6 stack (rebuild)
	docker compose --project-directory . -f containers/docker-compose.lab6.yml down
	docker compose --project-directory . -f containers/docker-compose.lab6.yml up --build

.PHONY: infra-status
infra-status: ## Show status of all infrastructure containers
	@echo "=== MLflow stack ===" && docker compose --project-directory . -f containers/docker-compose.yml ps 2>/dev/null || true
	@echo "\n=== Lab 6 stack ===" && docker compose --project-directory . -f containers/docker-compose.lab6.yml ps 2>/dev/null || true

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
