FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install wget and then Miniforge
RUN apt-get update && apt-get install -y wget && \
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

COPY environment.yaml .

RUN conda env create -f environment.yaml && conda clean -afy

COPY . .

CMD ["bash", "-c", "sleep infinity"]