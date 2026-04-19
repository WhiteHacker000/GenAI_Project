# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for some libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Expose the ports for the API (8000) and Streamlit (10000)
# (Though only Streamlit will be publicly accessible via Render)
EXPOSE 8000
EXPOSE 10000

# Specify the command to run on startup
CMD ["./start.sh"]
