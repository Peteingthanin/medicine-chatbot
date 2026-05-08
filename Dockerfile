FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for llama.cpp and other libs
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI code (src/ai/)
COPY src/ai/ ./src/ai/

# Copy frontend static files
COPY src/frontend/ ./src/frontend/

# Set Python path so imports from 'ai' package work correctly
ENV PYTHONPATH=/app/src

# Expose ports
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]