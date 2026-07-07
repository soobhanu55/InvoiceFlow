FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app/ ./streamlit_app/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
