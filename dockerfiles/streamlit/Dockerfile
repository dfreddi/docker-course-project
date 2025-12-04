FROM ubuntu:22.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /workdir

# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
    git \
    bash \
    nano \
    wget \
    python3.11 \
    python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY ./dockerfiles/streamlit/requirements.txt /workdir/requirements.txt
RUN pip install -r requirements.txt

# Copy source code and .env file
COPY ./src/web_app.py /workdir/web_app.py

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
