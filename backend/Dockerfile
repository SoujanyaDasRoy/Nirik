FROM python:3.11-slim

# Install system libraries required by OpenCV and image modules
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV PORT=7860
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Hugging Face runs under user 1000. Set permissions so SQLite can write to patients.db
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

EXPOSE 7860

CMD ["python", "app.py"]
