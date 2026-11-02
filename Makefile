.PHONY: venv install test lint train clean argilla-install argilla-up argilla-down argilla-check

venv:
	python -m venv .venv

install: venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\pip install -r requirements-dev.txt
	.venv\Scripts\pip install -e .

test:
	.venv\Scripts\pytest tests/ -v

lint:
	.venv\Scripts\ruff check src tests scripts argilla/scripts

train:
	.venv\Scripts\python scripts/train_mbert.py --release v4 --output-dir versiones/v4/model

clean:
	rmdir /s /q .pytest_cache 2>nul || exit 0

argilla-install:
	.venv\Scripts\pip install -e ".[argilla]"

argilla-up:
	docker compose --env-file .env.argilla -f infra/argilla/docker-compose.yml up -d

argilla-down:
	docker compose --env-file .env.argilla -f infra/argilla/docker-compose.yml down

argilla-check:
	.venv\Scripts\python argilla/scripts/check_argilla_env.py --ping-server
