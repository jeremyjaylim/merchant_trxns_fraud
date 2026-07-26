# 1. Base Image: Use an official lightweight Python image
FROM python:3.12-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy dependency definition and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project source files into the container
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .

# 6. Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# 7. Default command: Start Streamlit UI
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]