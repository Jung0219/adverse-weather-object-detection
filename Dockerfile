# Base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download YOLOv8 model weight by default (you can add others)
RUN mkdir -p weights && \
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O weights/yolov8n.pt

# Set default command (change depending on your task)
# CMD ["python", "src/preprocessing/exdark_to_coco.py", \"--exdark_root", "data/ExDark", \"--output", "data/ExDark/ground_truth.json"]
