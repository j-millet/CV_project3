FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Install dvc with gcp support
RUN pip install dvc[gs]

# Copy the content of the local src directory to the working directory
COPY . .

# Go to data
WORKDIR /app/data
# Pull dvc data
RUN dvc remote modify --local gcp_bucket credentialpath /app/data/cv3_read.json

RUN dvc pull

WORKDIR /app

#Forward port for streamlit
EXPOSE 8501

CMD ['echo','Welcome to the container!']
