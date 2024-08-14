# Use an official Python runtime with broad compatibility
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies for pyCGNS, scientific computing, and git
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    zlib1g-dev \
    cmake \
    gfortran \
    libgfortran5 \
    libopenblas-dev \
    liblapack-dev \
    libpng-dev \
    libjpeg-dev \
    wget \
    git \
    && apt-get clean

# Use a script to detect architecture and install the appropriate Miniforge version
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh; \
    elif [ "$arch" = "aarch64" ]; then \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh; \
    elif [ "$arch" = "ppc64le" ]; then \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh -O miniforge.sh; \
    elif [ "$arch" = "s390x" ]; then \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-s390x.sh -O miniforge.sh; \
    else \
        echo "Unsupported architecture: $arch"; exit 1; \
    fi && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh && \
    /opt/conda/bin/conda install -y python=3.8 numpy scipy vtk && \
    /opt/conda/bin/conda clean -afy

# Set path to conda
ENV PATH=/opt/conda/bin:$PATH

# Install pyCGNS from source
RUN git clone https://github.com/pyCGNS/pyCGNS.git /tmp/pyCGNS && \
    cd /tmp/pyCGNS && \
    python setup.py install && \
    rm -rf /tmp/pyCGNS

# Default command
CMD ["python", "main.py"]

