# Linux x86_64 helper image; model environments and checkpoints are separate.
# Build from the repository root:
#   docker build --platform linux/amd64 -f Environment/Dockerfile -t zinc_hydro:local .
# Run with the repository mounted:
#   docker run --rm -it -v "$PWD":/work -w /work zinc_hydro:local bash
# Add --gpus all for CUDA helpers, with NVIDIA Container Toolkit configured.
FROM condaforge/miniforge3:24.9.2-0

LABEL org.opencontainers.image.title="zinc_hydro" \
      org.opencontainers.image.description="Linux CUDA helpers for enzyme design; model stacks installed separately" \
      org.opencontainers.image.source="https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="" \
    PATH=/opt/conda/envs/zinc_hydro/bin:/opt/conda/bin:$PATH

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
        build-essential git wget ca-certificates aria2 \
 && rm -rf /var/lib/apt/lists/*

COPY Environment/zinc_hydro.yml /opt/zinc_hydro.yml

# Preserve the package manager's base runtime; install helpers in their own env.
RUN conda env create -y -f /opt/zinc_hydro.yml \
 && conda clean -afy

WORKDIR /work

RUN python -c "import numpy, pandas, scipy, openpyxl, torch, pyrosetta; print('helper imports OK')" \
 && obabel -V

CMD ["bash"]
