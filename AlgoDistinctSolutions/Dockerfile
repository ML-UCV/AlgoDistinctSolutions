FROM  tensorflow/tensorflow:2.17.0-gpu
WORKDIR /app

RUN apt update && apt upgrade -y && apt install wget git cppcheck build-essential -y 

RUN git clone https://github.com/dspinellis/tokenizer.git && make -C /app/tokenizer/src
RUN git clone https://github.com/radareorg/radare2 && radare2/sys/install.sh


RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install --no-cache-dir torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir gdown==5.2.0

COPY requirements.txt AlgoDistinctSolutions/requirements.txt
RUN pip install --no-cache-dir -r AlgoDistinctSolutions/requirements.txt 

RUN apt install git-lfs && git lfs install

#Download unixcoder model
RUN mkdir -p /app/Data/Pretrained_Models && cd /app/Data/Pretrained_Models && git clone https://huggingface.co/microsoft/unixcoder-base

#Download codet5plus model
RUN mkdir -p /app/Data/Pretrained_Models && cd /app/Data/Pretrained_Models && git clone https://huggingface.co/Salesforce/codet5p-110m-embedding

# #Download safe model
RUN mkdir /app/Data/Pretrained_Models/safe && \
    cd /app/Data/Pretrained_Models/safe && \
    gdown https://drive.google.com/uc?id=1Kwl8Jy-g9DXe1AUjUZDhJpjRlDkB4NBs && \
    gdown https://drive.google.com/uc?id=1CqJVGYbLDEuJmJV6KH4Dzzhy-G12GjGP && \
    tar -xvjf i2v.tar.bz2 && \
    rm i2v.tar.bz2

COPY dependencies  AlgoDistinctSolutions/dependencies 
RUN pip install --no-cache-dir AlgoDistinctSolutions/dependencies/safe
RUN pip install --no-cache-dir AlgoDistinctSolutions/dependencies/unixcoder

RUN pip install xgboost openai multiview google-genai


COPY . AlgoDistinctSolutions