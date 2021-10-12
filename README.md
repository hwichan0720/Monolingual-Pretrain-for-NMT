# Can Monolingual Pre-trained Encoder-Decoder Improve NMT for Distant Language Pairs?

Code for 
> Hwichan Kim and Mamoru Komachi. **Can Monolingual Pre-trained Encoder-Decoder Improve NMT for Distant Language Pairs?** In *PACLIC 2021*. [link].

# Main results of our paper

BART realizes consistent improvements regardless of the language pairs and translation directions used. (section 4.1)

<div align="center">
<img width='400' src='https://user-images.githubusercontent.com/49673825/136892091-0e5fe524-9224-4145-810d-1ae6a933dc2c.png'>
</div>
<div align="center">
<img width='700' src='https://user-images.githubusercontent.com/49673825/136892120-a8ea0ba7-fe3b-4d9e-895f-ae90edca0ffc.png'>
</div>

When languages are syntactically similar, BART can yield approximately twice the accuracy of our baseline model in the initial epoch. (section 4.1)
<div align="center">
<img width='700' alt='スクリーンショット 2021-10-12 13 34 16' src='https://user-images.githubusercontent.com/49673825/136892130-a0390378-e111-4eb9-9bea-cb68119edf22.png'>
</div>

The representations of the encoder remain unchanged after fine-tuning when high syntactic similarity prevails between pre-training and finetuning languages; however, the representations of the decoder change regardless of syntactic similarity. (section 4.2)

<div align="center">
 <img width='700' alt='スクリーンショット 2021-10-12 13 34 27' src='https://user-images.githubusercontent.com/49673825/136892137-64643a96-7fa0-4f0f-9dae-951b0fb2c3e8.png'>
</div>

## How to use

Plese set the [JPC](http://lotus.kuee.kyoto-u.ac.jp/WAT/patent/) and [Europarl](https://www.statmt.org/europarl/) corpora to the data directory.

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

#### Preprocess 
```
cd data/jabart/koja
bash preprocess.sh
```

#### Train
```
cd ko-ja/jabart
bash train.sh
```

### Calculate the CKA similarity (the experiment of section 4.2)

We add [examples](https://github.com/hwichan0720/Monolingual-Pretrain-for-NMT/blob/main/notebooks/CKA.ipynb) in the notebook directory.
