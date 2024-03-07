FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        wget \
        curl \
        unzip \
        build-essential \
        gfortran
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN curl --proto '=https' --tlsv1.2 -sSf -y https://sh.rustup.rs | sh
ENV PATH $PATH:/root/.cargo/bin

RUN python3 -m pip install \
    albumentations==0.4.3 \
    imageio-ffmpeg==0.4.2 \
    pudb==2019.2 \
    pytorch-lightning==1.4.2 \
    omegaconf==2.1.1 \
    test-tube>=0.7.5 \
    streamlit>=0.73.1 \
    transformers \
    torch-fidelity \
    einops \
    jupyter \
    notebook \
    diffusers \
    tqdm \
    matplotlib \
    Pillow \
    requests

ARG BASE_DIR=/root/diffusion

RUN mkdir -p $BASE_DIR
RUN wget https://github.com/JeonChangmin/latent-diffusion/archive/refs/heads/main.zip -O $BASE_DIR/main.zip
RUN unzip $BASE_DIR/main.zip -d $BASE_DIR
RUN mv $BASE_DIR/latent-diffusion-main $BASE_DIR/latent-diffusion
RUN rm $BASE_DIR/main.zip

RUN apt-get update && apt-get install -y --no-install-recommends git

WORKDIR $BASE_DIR/latent-diffusion
RUN python3 -m pip install \
    -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers \
    -e git+https://github.com/openai/CLIP.git@main#egg=clip

WORKDIR $BASE_DIR
