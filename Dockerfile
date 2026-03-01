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

# Ensure container uses CUDA-enabled PyTorch wheels for cloud GPU training.
RUN conda run -n asidiels python -m pip install --upgrade pip && \
    conda run -n asidiels python -m pip install \
      torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
      --index-url https://download.pytorch.org/whl/cu121

COPY . .

CMD ["bash", "-c", "sleep infinity"]
