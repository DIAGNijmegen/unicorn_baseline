ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

# create input/output directory
RUN mkdir -p /input /output /opt/app/workdir/language /opt/tiktoken_cache && \
    chown -R user:user /input /output /opt/app/workdir/language /opt/tiktoken_cache

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Amsterdam
USER root

# Set /home/user as working directory
WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff-dev \
    zlib1g-dev \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python
RUN apt-get update && apt-get install -y python3-pip python3-dev python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ASAP
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN apt-get update && curl -L ${ASAP_URL} -o /tmp/ASAP.deb && apt-get install --assume-yes /tmp/ASAP.deb && \
    SITE_PACKAGES=`python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && apt-get clean && \
    echo "/opt/ASAP/bin" > /usr/lib/python3/dist-packages/asap.pth && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh
ENV OLLAMA_MODELS=/opt/ml/model/phi4

# Switch to user
USER user
WORKDIR /opt/app/

# You can add any Python dependencies to requirements.in/requirements.txt
RUN python -m pip install --upgrade pip setuptools pip-tools \
    && rm -rf /home/user/.cache/pip

COPY --chown=user:user requirements.in /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.in \
    && rm -rf /home/user/.cache/pip

# Cache tiktoken
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN pip install --upgrade tiktoken
ARG TIKTOKEN_URL="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
RUN wget -O /opt/tiktoken_cache/$(echo -n $TIKTOKEN_URL | sha1sum | head -c 40) $TIKTOKEN_URL

# install UNICORN baseline
COPY --chown=user:user src /opt/app/unicorn_baseline/src
COPY --chown=user:user LICENSE /opt/app/unicorn_baseline/
COPY --chown=user:user setup.py /opt/app/unicorn_baseline/
COPY --chown=user:user setup.cfg /opt/app/unicorn_baseline/
COPY --chown=user:user README.md /opt/app/unicorn_baseline/
COPY --chown=user:user pyproject.toml /opt/app/unicorn_baseline/

RUN python -m pip install  /opt/app/unicorn_baseline

ENTRYPOINT ["unicorn_baseline"]
