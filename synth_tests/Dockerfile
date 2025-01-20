FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG CONTAINER_TIMEZONE=UTC
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    python3.9 \
    python3.9-distutils \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 

WORKDIR /synth_tests

COPY requirements.txt /synth_tests/requirements.txt

RUN python3.9 -m pip install --no-cache-dir -r /synth_tests/requirements.txt

CMD ["bash"]