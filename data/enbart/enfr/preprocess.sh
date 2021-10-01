#!/usr/bin/env -eu
ROOT=../../..
data_dir=../../europarl/enfr
PRETRAINED_DIR=$ROOT/pretrained_bart
ENBART=$PRETRAINED_DIR/bart.base
tools=$ROOT/tools
MOSES=$tools/mosesdecoder
MOSES_SCRIPT=$MOSES/scripts
TRIM_BART=$PRETRAINED_DIR/trim

echo "random sampling 1M lines"
random_dir=./
mkdir -p $random_dir
python $tools/random_sampling.py -f $data_dir/europarl-v7.fr-en -l1 en -l2 fr -s $random_dir/train.random -n 1000000

echo "train french spm"
random_model=$random_dir/model
mkdir -p $random_model
spm_train --input=$random_dir/train.random.fr --model_prefix=$random_model/fr --vocab_size=30000

echo "apply spm to french"
spm_encode --model=$random_model/fr.model --output_format=piece < $random_dir/train.random.fr > $random_dir/train.sp.fr
spm_encode --model=$random_model/fr.model --output_format=piece < $data_dir/test2007/test2007.fr > $random_dir/dev.sp.fr
spm_encode --model=$random_model/fr.model --output_format=piece < $data_dir/test2008/test2008.fr > $random_dir/test.sp.fr

echo "piece to id using EnBART encoder.json"
for prefix in train dev test;do
    python $tools/token_to_id.py $ENBART/encoder.json $random_dir/$prefix.sp.fr > $random_dir/$prefix.fr
done

echo "preprocess english"
python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json $ENBART/encoder.json \
    --vocab-bpe $ENBART/vocab.bpe \
    --inputs $random_dir/train.random.en  \
    --outputs $random_dir/train.en \
    --workers 60 \
    --keep-empty

python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json $ENBART/encoder.json \
    --vocab-bpe $ENBART/vocab.bpe \
    --inputs $data_dir/test2007/test2007.en \
    --outputs $random_dir/dev.en \
    --workers 60 \
    --keep-empty

python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json $ENBART/encoder.json \
    --vocab-bpe $ENBART/vocab.bpe \
    --inputs $data_dir/test2008/test2008.en  \
    --outputs $random_dir/test.en \
    --workers 60 \
    --keep-empty

echo "binalize data"
fairseq-preprocess \
    --source-lang fr \
    --target-lang en \
    --trainpref $random_dir/train \
    --validpref $random_dir/dev \
    --testpref  $random_dir/test \
    --joined-dictionary \
    --destdir $random_dir/fr-en

fairseq-preprocess \
    --source-lang en \
    --target-lang fr \
    --trainpref $random_dir/train \
    --validpref $random_dir/dev \
    --testpref  $random_dir/test \
    --joined-dictionary \
    --destdir $random_dir/en-fr

echo "trim japanese bart"
mkdir -p $TRIM_BART
mkdir -p $TRIM_BART/enbart_enfr

CUDA_VISIBLE_DEVICES=2 python $tools/trim_bart_en.py \
        --pre-train-dir $ENBART \
        --ft-dict $random_dir/en-fr/dict.en.txt \
        --output $PRETRAINED_DIR/trim/enbart_enfr/enbart_base.pt
cp en-fr/dict.en.txt $TRIM_BART/enbart_enfr/dict.txt 

