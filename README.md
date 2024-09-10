# AI-Powered Image Recognition System

This project is an AI-powered image recognition system built using Django and PyTorch. The system allows users to upload images for classification into different categories (e.g., cats, dogs). The project includes a machine learning model (CNN), a REST API for image classification, and a simple frontend interface for uploading images.

## Features
- Image Classification: A Convolutional Neural Network (CNN) is used to classify images.
- API Integration: Django REST framework is used to expose an API for real-time image classification.
- Frontend: A basic HTML interface for uploading images and viewing classification results.
- Dockerized: Fully containerized using Docker, with separate containers for the web app and PostgreSQL database.

## Prerequisites
- Docker: Ensure you have Docker installed.
- Python 3.9+: If running locally without Docker, you'll need Python 3.9 or higher installed

## Getting Started
1. Clone repository
    `git clone https://github.com/your-repository/image-recognition-system.git`
    `cd image-recognition-system`
2. Set up environment variables in `.env` file
    `POSTGRES_DB=django_db`
    `POSTGRES_USER=django_user`
    `POSTGRES_PASSWORD=password`
    `SECRET_KEY=your-secret-key`
    `DEBUG=1`
    `DJANGO_SUPERUSER_USERNAME=admin`
    `DJANGO_SUPERUSER_PASSWORD=admin`
    `DJANGO_SUPERUSER_EMAIL=admin@example.com`
3. Running with Docker
    `docker-compose up --build`
4. Running locally (no Docker)
    `python -m venv venv`
    `source venv/bin/activate`
    `pip install -r requirements.txt`
    `python manage.py migrate`
    `python manage.py createsuperuser`
    `python manage.py runserver`

## Project Workflow
- Training Model
    `docker-compose run web python cnn_model/scripts/train_model.py --data_dir /path/to/training/data --epochs 10`
- Testing Model
    `docker-compose run web python cnn_model/scripts/test_model.py --data_dir /path/to/test/data --checkpoint /path/to/model_checkpoint.pth`
- Preprocessing Images
    `docker-compose run web python cnn_model/scripts/preprocess_images.py --data_dir /path/to/raw/images --output_dir /path/to/preprocessed/images`

## Key Technologies
- Django: A high-level Python web framework for building the backend.
- Django REST Framework: For building API endpoints.
- PyTorch: A machine learning library for building the image classification model.
- PostgreSQL: A relational database for storing data.
- Docker: For containerizing the application and its dependencies.
- Gunicorn: A Python WSGI HTTP server for serving the Django app in production.
