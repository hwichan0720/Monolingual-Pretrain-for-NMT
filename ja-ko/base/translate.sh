#!/usr/bin/env -eu

GPU=$1
DIR=$2
ROOT=../..
tools=$ROOT/tools
DATA=$ROOT/data
TRANSLATION=$DIR/translation
RESULTS=$TRANSLATION/translation-results.txt
BIN_DATA=$DATA/ja-ko


echo "mkdir $TRANSLATION"
mkdir -p $TRANSLATION

# Translate
for point in $(ls -tr $DIR/*.pt); do
    MODEL=${point::-3}
    echo "" >> $RESULTS
    echo $MODEL
    echo $MODEL >> $RESULTS
    echo "" >> $RESULTS
for prefix in valid test test1 test2 test3; do
    
    echo $prefix
    echo $prefix >> $RESULTS

    sp_file=$TRANSLATION/${MODEL:${#DIR}}.$prefix.sp.ko
    tok_file=$TRANSLATION/${MODEL:${#DIR}}.$prefix.tok.ko

    if [ ! -f ${sp_file} ]; then
        CUDA_VISIBLE_DEVICES=$GPU PYTHONIOENCODING=utf-8 fairseq-generate $BIN_DATA \
            --path $point \
            --gen-subset $prefix -s ja -t ko \
            --remove-bpe=sentencepiece \
            --scoring sacrebleu > $sp_file
    fi

    # De-tokenize 
    if [ ! -f $tok_file ]; then
       PYTHONIOENCODING=utf-8 python $tools/ordering.py -i $sp_file > $tok_file
    fi

    # Evaluate using sacreBLEU (WMT official scorer)
    if [ $prefix == 'valid' ]; then
        prefix=dev
    elif [ $prefix == 'test' ]; then
        prefix=test-n
    elif [ $prefix == 'test1' ]; then
        prefix=test-n1
    elif [ $prefix == 'test2' ]; then
        prefix=test-n2
    elif [ $prefix == 'test3' ]; then
        prefix=test-n3
    fi
    cat $tok_file | sacrebleu -w 2 $DATA/$prefix.tok.ko
    cat $tok_file | sacrebleu -w 2 $DATA/$prefix.tok.ko >> $RESULTS

done
done