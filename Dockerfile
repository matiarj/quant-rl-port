FROM python:3.9-slim

LABEL maintainer="yourgithubhandle"
LABEL description="Bregma RL Portfolio - Reinforcement Learning for Portfolio Optimization"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set default command
CMD ["python", "main.py", "--config", "config.yaml"]