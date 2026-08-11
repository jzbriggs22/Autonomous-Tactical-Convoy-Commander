FROM python:3.11-slim

WORKDIR /app

# System libraries for GDAL/rasterio (needed for geo features)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install only what's needed for build
COPY pyproject.toml .
COPY convoy_commander/ convoy_commander/
COPY tests/ tests/
COPY Makefile .

# Install package with dev + geo dependencies
RUN pip install --no-cache-dir -e ".[dev,geo,weather]"

# The governance stack (ai_governance/ + the "governance" extra, which pulls
# in torch via outlines) is intentionally not installed in this image, so the
# default command must skip tests/governance or collection fails on import.
ENTRYPOINT ["python", "-m"]
CMD ["pytest", "tests/", "-q", "--ignore=tests/governance"]
