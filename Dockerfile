FROM python:3.12-slim

WORKDIR /app

# Install deps (upgrade pip first to avoid notices)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy code
COPY app.py .env .

# Expose for potential API (not used here)
EXPOSE 8080

# Run interactively
CMD ["python", "app.py"]