FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV HOME=/home/app

COPY . $HOME/myapp

RUN apt-get update && apt-get install -y \
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

RUN pip install -U pip &&\
    pip install zenhan &&\
    pip install sentencepiece 

# install fastalign
RUN git clone https://github.com/clab/fast_align.git &&\
    cd fast_align &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make 

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

# install kytea
RUN wget http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz &&\
    tar zxvf kytea-0.4.7.tar.gz && cd kytea-0.4.7 && \
    wget https://patch-diff.githubusercontent.com/raw/neubig/kytea/pull/24.patch && \
    git apply ./24.patch && ./configure && \
    make && make install && ldconfig -v

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

# install fasttext
RUN wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip &&\
    unzip v0.9.2.zip &&\
    cd fastText-0.9.2 &&\
    make

# clone MUSE
RUN git clone https://github.com/facebookresearch/MUSE.git

RUN rm -rf $HOME/src && \
    apt-get purge -y \
    build-essential \
    curl \
    bzip2

RUN pip install pyknp &&\
    git clone -b japanese_bart_pretrained_model https://github.com/utanaka2000/fairseq.git &&\
    cd fairseq &&\
    pip install --editable .\
    pip install scipy

WORKDIR /home




