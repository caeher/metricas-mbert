.PHONY: venv install test lint train benchmark notebooks clean

venv:
	python -m venv .venv

install: venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\pip install -r requirements-dev.txt
	.venv\Scripts\pip install -e .

test:
	.venv\Scripts\pytest tests/ -v

lint:
	.venv\Scripts\ruff check src tests scripts

train:
	.venv\Scripts\python scripts/train_mbert.py

benchmark:
	.venv\Scripts\python scripts/run_benchmarks.py

notebooks:
	.venv\Scripts\jupyter notebook notebooks/

clean:
	rmdir /s /q .pytest_cache 2>nul || exit 0
