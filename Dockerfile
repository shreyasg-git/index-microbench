FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-5 g++-5 \
    make cmake \
    git wget curl \
    libnuma-dev \
    linux-tools-common \
    linux-tools-generic \
    && rm -rf /var/lib/apt/lists/*

# Build and install PAPI 5.4.3 (old API compatible)
WORKDIR /opt

RUN wget http://icl.utk.edu/projects/papi/downloads/papi-5.4.3.tar.gz && \
    tar -xzf papi-5.4.3.tar.gz && \
    cd papi-5.4.3/src && \
    ./configure && \
    make -j$(nproc) && \
    make install

# Environment so compiler finds PAPI
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV CPLUS_INCLUDE_PATH=/usr/local/include
ENV C_INCLUDE_PATH=/usr/local/include

# Default compiler = gcc-5
ENV CC=gcc-5
ENV CXX=g++-5

WORKDIR /workspace

