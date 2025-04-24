FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend ./backend

EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:8000", "backend.app:app"]
