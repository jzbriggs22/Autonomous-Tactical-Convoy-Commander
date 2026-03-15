FROM python:3.11-slim

WORKDIR /app

# Install only what's needed for build
COPY pyproject.toml .
COPY convoy_commander/ convoy_commander/
COPY tests/ tests/
COPY Makefile .

# Install package with dev dependencies
RUN pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["python", "-m"]
CMD ["pytest", "-q"]
