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
RUN pip install wrapt --upgrade --ignore-installed

# Copy requirements and install dependencies
COPY ./dockerfiles/chatbot/requirements.txt /workdir/requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY ./src/chatbot_backend.py /workdir/chatbot_backend.py
COPY ./prompts /workdir/prompts

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
