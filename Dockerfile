# Use NVIDIA CUDA 11.8 as the base
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up Conda environment paths
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install basic system tools
RUN apt-get update && apt-get install -y \
    wget \
    libxrender1 \
    libxext6 \
    libsm6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Set working directory
WORKDIR /app

# Copy the updated environment file
COPY environment.yml .

# Create the environment. We use 'mamba' for faster solving if available
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -f environment.yml && \
    conda clean -afy

# Set the active environment in the PATH
ENV PATH /opt/conda/envs/reaction_env/bin:$PATH

# Copy all your source files (evaluate_rl.py, etc.)
COPY . .

# Launch command
CMD ["python", "evaluate_rl.py"]