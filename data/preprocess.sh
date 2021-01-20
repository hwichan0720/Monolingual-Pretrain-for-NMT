#!/usr/bin/env -eu
ROOT=..
TRAINDEV=JPC4.3/traindev/ko-ja
TEST=JPC4.3/test/ko-ja
MECABKO=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic
PRETRAINED_DIR=$ROOT/pretrained_bart/
JPBART=$PRETRAINED_DIR/japanese_bart_base_1.1
tools=$ROOT/tools
MODEL=model
TRIM_BART=$PRETRAINED_DIR/trim

echo "preprocess japanese"
for prefix in train dev; do 
    python $tools/jaBART_preprocess.py -m $JPBART/sp.model -i $TRAINDEV/$prefix -d $JPBART/dict.txt  -o $prefix -l1 ja -l2 ko
done

for prefix in n n1 n2 n3; do 
    python $tools/jaBART_preprocess.py -m $JPBART/sp.model -i $TEST/test-$prefix -d $JPBART/dict.txt -o test-$prefix -l1 ja -l2 ko
done

echo "apply mecab-ko"
for prefix in train dev; do
    cat $prefix.tmp.ko | mecab -d $MECABKO -O wakati > $prefix.tok.ko
done

for prefix in n n1 n2 n3; do 
    cat test-$prefix.tmp.ko | mecab -d $MECABKO -O wakati > test-$prefix.tok.ko
done

echo "train SP model for korean"
mkdir -p $MODEL
python $tools/apply_sp.py -m $MODEL/sp.ko -i train.tok.ko -v 32000 -f True

for prefix in train dev; do
    python $tools/apply_sp.py -m $MODEL/sp.ko -i $prefix.tok.ko > $prefix.ko
done

for prefix in n n1 n2 n3; do 
    python $tools/apply_sp.py -m $MODEL/sp.ko -i test-$prefix.tok.ko > test-$prefix.ko
done

echo "binalize ko-ja data"
fairseq-preprocess \
    --source-lang ko \
    --target-lang ja \
    --trainpref train \
    --validpref dev \
    --testpref  test-n,test-n1,test-n2,test-n3 \
    --joined-dictionary　\
    --destdir ko-ja

echo "binalize ja-ko data"
fairseq-preprocess \
    --source-lang ja \
    --target-lang ko \
    --trainpref train \
    --validpref dev \
    --testpref  test-n,test-n1,test-n2,test-n3 \
    --joined-dictionary \
    --destdir ja-ko

echo "binalize ja-ja data"
fairseq-preprocess \
    --source-lang ja \
    --target-lang ja \
    --trainpref train \
    --validpref dev \
    --testpref  test-n,test-n1,test-n2,test-n3 \
    --srcdict ko-ja/dict.ja.txt \
    --joined-dictionary \
    --destdir ja-ja

echo "binalize data"
fairseq-preprocess \
    --source-lang ja \
    --target-lang ja \
    --trainpref train \
    --validpref dev \
    --testpref  test-n,test-n1,test-n2,test-n3 \
    --srcdict $JPBART/dict.txt \
    --joined-dictionary \
    --destdir zeroshot-ja-ja

echo "trim japanese bart"
mkdir -p $TRIM_BART
python $tools/trim_bart.py --pre-train-dir $JPBART --ft-dict bin/dict.ja.txt --output $TRIM_BART/ja_bart_base.pt