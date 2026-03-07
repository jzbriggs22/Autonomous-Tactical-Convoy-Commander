.PHONY: help install test typecheck demo sweep evaluate clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install package with dev dependencies
	pip install -e ".[dev]"

test:  ## Run all tests
	python -m pytest -q --tb=short

typecheck:  ## Run mypy type checker
	python -m mypy convoy_commander/ --ignore-missing-imports

demo:  ## Run baseline scenario (quick 60s demo)
	python -m convoy_commander run --scenario baseline --seed 42 --duration 60

sweep:  ## Run all 10 scenarios sequentially (seed=42, 60s each)
	@for s in baseline gps_denied comms_degraded leader_failure obstacle_pop \
		gps_spoofed silent_running comms_blackout sensor_drift_spike platooning; do \
		echo "--- $$s ---"; \
		python -m convoy_commander run --scenario $$s --seed 42 --duration 60; \
	done

evaluate:  ## Run evaluation harness (6 scenarios x 3 seeds)
	python -m convoy_commander evaluate

clean:  ## Remove caches (does not delete runs/)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache
