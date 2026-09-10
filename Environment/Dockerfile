# =============================================================================
#  zinc_hydro -- Docker image for the Metallohydrolase enzyme design repository
# =============================================================================
#
#  For users who prefer Docker to Apptainer. Functionally equivalent to
#  Environment/zinc_hydro.def.
#
#  BUILD (run from the repository root, not from Environment/):
#
#      docker build -f Environment/Dockerfile -t zinc_hydro:latest .
#
#  RUN -- mount the repo and work inside it:
#
#      docker run --rm -it --gpus all \
#          -v "$PWD":/work -w /work \
#          zinc_hydro:latest bash
#
#  JUPYTER:
#
#      docker run --rm -it --gpus all -p 8888:8888 \
#          -v "$PWD":/work -w /work zinc_hydro:latest \
#          jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
#
#  CONVERT TO APPTAINER (handy on clusters that forbid Docker but allow .sif):
#
#      apptainer build zinc_hydro.sif docker-daemon://zinc_hydro:latest
#      export ZINC_HYDRO_SIF=$PWD/zinc_hydro.sif
#
#  Only running the wet lab analysis? This image is overkill -- see
#  Environment/requirements-analysis.txt.
# =============================================================================

FROM condaforge/miniforge3:24.9.2-0

LABEL org.opencontainers.image.title="zinc_hydro" \
      org.opencontainers.image.description="Metallohydrolase enzyme design environment (RFdiffusion2/3, PyRosetta, AF2)" \
      org.opencontainers.image.source="https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONNOUSERSITE=1 \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
        build-essential git wget ca-certificates aria2 \
 && rm -rf /var/lib/apt/lists/*

COPY Environment/zinc_hydro.yml /opt/zinc_hydro.yml

RUN conda config --set always_yes yes --set changeps1 no \
 && mamba env update -n base -f /opt/zinc_hydro.yml \
 && mamba clean -afy \
 && find /opt/conda -follow -type f -name '*.a'   -delete \
 && find /opt/conda -follow -type f -name '*.pyc' -delete \
 && rm -rf /opt/conda/pkgs

WORKDIR /work

# Smoke test at build time so a broken solve fails the build, not the user.
RUN python -c "import numpy, pandas, scipy, openpyxl; print('scientific stack OK')" \
 && python -c "import pyrosetta; print('pyrosetta OK')"

CMD ["bash"]
