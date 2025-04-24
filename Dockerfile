FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend ./backend

EXPOSE 8000

# Ensure matplotlib doesn't try to use X11
ENV MPLBACKEND=Agg

# Command to run the application
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:8000", "backend.app:app"]
