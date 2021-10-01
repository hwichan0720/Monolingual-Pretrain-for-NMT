# Can Monolingual Pre-trained Encoder-Decoder Improve NMT for Distant Language Pairs?

Code for Can Monolingual Pre-trained Encoder-Decoder Improve NMT for Distant Language Pairs?

## How to use

Plese set the JPC and Europarl corpora to data directory.

### Build docker image

```
cd dockerfiles
docker build -t image_name t .
```

### Run docker container

```
cd koja_jabart
docker run -v `pwd`:/home --gpus all -it image_name
```

### Train the Ko-Ja model 

#### Download the pretrained BART
```
cd pretrained_bart
bash download.sh
```

#### Data preprocess
```
cd data/jabart/koja
bash preprocess.sh
```

#### Train
```
cd ko-ja/jabart
bash train.sh
```
