FROM python:3.9-slim
USER root

WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    scanpy \
    pandas \
    numpy \
    anndata \
    xgboost \
    shap \
    matplotlib \
    scikit-learn \
    scipy

RUN pip install boto3==1.35.95

RUN pip install pyWGCNA


ENV AWS_ACCESS_KEY_ID="..."
ENV AWS_SECRET_ACCESS_KEY="..." 

ENV CACHE_BUST=854

COPY . .
