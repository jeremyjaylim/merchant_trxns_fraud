# 1. Base Image: Use an official, lightweight Python 3.12 runtime
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Environment flags: Prevent Python from buffering outputs or writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy dependency file and install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project source code and trained model artifacts into the container
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .

# 6. Expose the port Streamlit runs on
EXPOSE 8501

# 7. Default command: Launch the Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]