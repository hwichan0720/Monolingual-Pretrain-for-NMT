#!/usr/bin/env -eu

GPU=$1
DIR=$2
ROOT=../..
tools=$ROOT/tools
DATA=$ROOT/data/enja
TRANSLATION=$DIR/translation
RESULTS=$TRANSLATION/translation-results.txt
BIN_DATA=$DATA/enBART/en-ja
ENBART=$ROOT/pretrained_bart/bart.base


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

    sp_file=$TRANSLATION/${MODEL:${#DIR}}.$prefix.sp.ja
    tok_file=$TRANSLATION/${MODEL:${#DIR}}.$prefix.tok.ja
    retok_file=$TRANSLATION/${MODEL:${#DIR}}.$prefix.retok.ja

    if [ ! -f ${sp_file} ]; then
        CUDA_VISIBLE_DEVICES=$GPU fairseq-generate $BIN_DATA \
            --path $point \
            --scoring sacrebleu \
            --gen-subset $prefix -s en -t ja > $sp_file
    fi

    # Ordering
    if [ ! -f $tok_file ]; then
       python $tools/ordering.py -i $sp_file > $tok_file
    fi

    # id-to-words and de-tokenize and re-tokenize
    if [ ! -f $retok_file ]; then
       python $tools/id_to_token_jp.py $ENBART/encoder.json $tok_file > $retok_file
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
    cat $retok_file | sacrebleu -w 2 $DATA/$prefix.tok.ja
    cat $retok_file | sacrebleu -w 2 $DATA/$prefix.tok.ja >> $RESULTS

done
done