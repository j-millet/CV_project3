FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

#copy code to container
COPY ./*.* /app/CV_project3/

WORKDIR /app/CV_project3

# Install any dependencies
RUN pip install -r requirements.txt

RUN pip install dvc[gs]