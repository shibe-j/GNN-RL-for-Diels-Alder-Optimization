FROM continuumio/miniconda3:23.11.0-0

WORKDIR /app

COPY environment.yaml .

RUN conda env create -f environment.yaml && conda clean -afy

SHELL ["bash", "-lc"]

RUN echo "conda activate asidiels" >> ~/.bashrc

COPY . .

CMD ["bash", "-lc", "conda activate asidiels && python deepchem/reaction_optimizer.py --data-root deepchem"]

