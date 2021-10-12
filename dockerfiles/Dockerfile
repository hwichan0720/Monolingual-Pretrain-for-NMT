FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV HOME=/home/app

COPY . $HOME/myapp

RUN apt-get update && apt-get install -y \
    sudo \
    build-essential \
    curl \
    bzip2 \
    libjuman \
    libcdb-dev \
    libboost-all-dev \
    make \
    cmake \
    wget \
    git \
    autoconf \
    unzip \
    automake \
    zlib1g-dev

# install juman++
RUN mkdir $HOME/src && \
    cd $HOME/src && \
    curl -L -o jumanpp-2.0.0-rc2.tar.xz https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc2/jumanpp-2.0.0-rc2.tar.xz && \
    tar Jxfv jumanpp-2.0.0-rc2.tar.xz && \
    cd jumanpp-2.0.0-rc2/ && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ && \
    make && \
    make install

# install mecab
RUN apt-get install -y mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8

# install mecab-ko
RUN wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz &&\
    tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz &&\
    cd mecab-ko-dic-2.1.1-20180720 &&\
    ./autogen.sh &&\
    ./configure  &&\
    make &&\
    make install 

# install sentencepiece
RUN git clone https://github.com/google/sentencepiece.git &&\
    cd sentencepiece &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j $(nproc) &&\
    make install &&\
    ldconfig -v

RUN rm -rf $HOME/src && \
    apt-get purge -y \
    build-essential \
    curl \
    bzip2

# install japanese_bart_pretrained_model branch of fairseq implementation 
RUN pip install -U pip &&\
    pip install pyknp &&\
    git clone -b japanese_bart_pretrained_model https://github.com/utanaka2000/fairseq.git &&\
    cd fairseq &&\
    pip install --editable .


RUN pip install zenhan &&\
    pip install sentencepiece &&\
    pip install pandas &&\
    pip install scipy &&\
    pip install -U numpy &&\
    pip install seaborn &&\
    pip install jupyterlab

RUN jupyter serverextension enable --py jupyterlab 

ARG UID
RUN useradd newuser -u $UID -m  
RUN echo 'newuser:password' | chpasswd
RUN usermod -aG sudo newuser
USER newuser

WORKDIR /home




