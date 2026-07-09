FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# These match the env var names agent.py actually reads.
# Pass real values at `docker run -e ...` time — do not bake secrets into the image.
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
ENV FIREWORKS_MODEL_NAME=""

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]