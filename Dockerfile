FROM python:3.9-slim

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt

RUN pip install --no-cache-dir \
    -r backend/requirements.txt

COPY backend/ ./backend/
COPY ai/sign_recognition/models/ \
     ./ai/sign_recognition/models/

WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]