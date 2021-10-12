#!/usr/bin/env -eu
ROOT=../../..
JPC=../../JPC4.3
TRAINDEV=$JPC/traindev/en-ja
TEST=$JPC/test/en-ja
PRETRAINED_DIR=$ROOT/pretrained_bart
ENBART=$PRETRAINED_DIR/bart.base
JPBART=$PRETRAINED_DIR/japanese_bart_base_1.1
tools=$ROOT/tools
MOSES=$tools/mosesdecoder
MOSES_SCRIPT=$MOSES/scripts
TRIM_BART=$PRETRAINED_DIR/trim

MODEL=model

echo "preprocess japanese"
for prefix in train dev; do 
    python $tools/jaBART_preprocess_en.py -m $JABART/sp.model -i $TRAINDEV/$prefix -d $JABART/dict.txt  -o $prefix -l1 ja -l2 en
done

for prefix in n n1 n2 n3; do 
    python $tools/jaBART_preprocess_en.py -m $JABART/sp.model -i $TEST/test-$prefix -d $JABART/dict.txt -o test-$prefix -l1 ja -l2 en
done

echo "preprocess english"
python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json $ENBART/encoder.json \
    --vocab-bpe $ENBART/vocab.bpe \
    --inputs $TRAINDEV/train.tmp.en  \
    --outputs ./train.en \
    --workers 60 \
    --keep-empty

python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json $ENBART/encoder.json \
    --vocab-bpe $ENBART/vocab.bpe \
    --inputs $TRAINDEV/dev.tmp.en \
    --outputs ./dev.en \
    --workers 60 \
    --keep-empty

for prefix in n n1 n2 n3; do 
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json $ENBART/encoder.json \
        --vocab-bpe $ENBART/vocab.bpe \
        --inputs $TRAINDEV/test-$prefix.tmp.en  \
        --outputs ./test-$prefix.en \
        --workers 60 \
        --keep-empty
done

echo "binalize data"
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref ./train \
    --validpref ./dev \
    --testpref  ./test \
    --joined-dictionary \
    --destdir ja-en

fairseq-preprocess \
    --source-lang en \
    --target-lang ja \
    --trainpref ./train \
    --validpref ./dev \
    --testpref ./test \
    --joined-dictionary \
    --destdir en-ja

echo "trim japanese bart"
mkdir -p $TRIM_BART
mkdir -p $TRIM_BART/enbart_enja

CUDA_VISIBLE_DEVICES=0 python $tools/trim_bart_en.py \
        --pre-train-dir $ENBART \
        --ft-dict en-ja/dict.en.txt \
        --output $TRIM_BART/enbart_enja/enbart_base.pt
cp en-ja/dict.en.txt $TRIM_BART/enbart_enja//dict.txt 

