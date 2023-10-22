# Use an official Python image as the base image
FROM python:3.6.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port that your Django app will run on (e.g., 8000)
EXPOSE 8000

# Replace "your_project_name" with your Django project's name
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]