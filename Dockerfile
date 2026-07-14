# Hugging Face Spaces (Docker SDK) container for the Credit Default API.
# Free CPU basic tier; Spaces route traffic to app_port declared in README.md.
FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what serving needs: the app and the trained artifacts.
COPY app.py .
COPY model/ model/

# Spaces run containers as a non-root user with a numeric UID.
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
