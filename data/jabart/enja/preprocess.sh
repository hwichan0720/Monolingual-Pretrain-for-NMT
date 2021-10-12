#!/usr/bin/env -eu
ROOT=../../..
JPC=../../JPC4.3
TRAINDEV=$JPC/traindev/en-ja
TEST=$JPC/test/en-ja
MECABKO=$ROOT/mecab-dic/mecab-ko-dic
PRETRAINED_DIR=$ROOT/pretrained_bart
JABART=$PRETRAINED_DIR/japanese_bart_base_1.1
JABERT=$PRETRAINED_DIR/Japanese_L-12_H-768_A-12_E-30_BPE_WWM
ENBART=$PRETRAINED_DIR/bart.base
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

echo "tokenize english"
for prefix in train dev ; do
  cat $prefix.tmp.en | \
  perl ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l en | \
  perl ${MOSES_SCRIPT}/tokenizer/tokenizer.perl -l en -no-escape \
  > $prefix.tok.en
done

for prefix in n n1 n2 n3; do
  cat test-$prefix.tmp.en | \
  perl ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l en | \
  perl ${MOSES_SCRIPT}/tokenizer/tokenizer.perl -l en -no-escape \
  > test-$prefix.tok.en
done

echo "train SP model for english"
mkdir -p $MODEL
python $tools/apply_sp.py -m $MODEL/sp.en -i train.tok.en -v 32000 -f True

for prefix in train dev; do
    python $tools/apply_sp.py -m $MODEL/sp.en -i $prefix.tok.en > $prefix.en
done

for prefix in n n1 n2 n3; do 
    python $tools/apply_sp.py -m $MODEL/sp.en -i test-$prefix.tok.en > test-$prefix.en
done

echo "binalize en-ja data"
fairseq-preprocess \
    --source-lang en \
    --target-lang ja \
    --trainpref train \
    --validpref dev \
    --testpref  test-n,test-n1,test-n2,test-n3 \
    --joined-dictionary \
    --destdir en-ja

echo "binalize ja-en data"
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref train \
    --validpref dev \
    --testpref  test-n,test-n1,test-n2,test-n3 \
    --joined-dictionary \
    --destdir ja-en

echo "trim bart"
mkdir -p $TRIM_BART
mkdir -p $TRIM_BART/jabart_jaen
python $tools/trim_bart.py --pre-train-dir $JABART --ft-dict en-ja/dict.en.txt --output $TRIM_BART/jabart_jaen/jabart_base.pt
cp ja-en/dict.ja.txt $TRIM_BART/jabart_jaen/dict.txt
