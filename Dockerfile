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

RUN apt-get update && apt-get install -y --no-install-recommends git

RUN python3 -m pip install \
    -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers \
    -e git+https://github.com/openai/CLIP.git@main#egg=clip

RUN python3 -m pip install ftfy accelerate scipy
RUN python3 -m pip install packaging==21.3
RUN python3 -m pip install 'torchmetrics<0.8'
RUN python3 -m pip install kornia

ARG BASE_DIR=/root/diffusion

RUN mkdir -p $BASE_DIR
RUN git clone --depth 1 https://github.com/JeonChangmin/latent-diffusion.git $BASE_DIR/latent-diffusion

RUN cp $BASE_DIR/latent-diffusion/notebooks/DDPM_DDIM_CFG_tutorial.ipynb $BASE_DIR/
RUN cp $BASE_DIR/latent-diffusion/notebooks/Diffusers_Tasks_Tutorial.ipynb $BASE_DIR/
RUN cp $BASE_DIR/latent-diffusion/notebooks/Diffusers_tutorial_pipeline.ipynb $BASE_DIR/

WORKDIR $BASE_DIR
