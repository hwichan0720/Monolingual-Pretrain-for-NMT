#!/usr/bin/env -eu
GPU=$1
ROOT=../..
fairseq=/fairseq
BIN=$ROOT/data/ko-ja

CUDA_VISIBLE_DEVICES=$GPU fairseq-train $BIN \
    --seed 1 \
    --keep-last-epochs 10 \
    --arch transformer \
    --optimizer adam \
    --encoder-embed-dim 768 \
    --encoder-ffn-embed-dim 3072 \
    --encoder-attention-heads 12 \
    --decoder-embed-dim 768 \
    --decoder-ffn-embed-dim 3072 \
    --decoder-attention-heads 12 \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0005 \
    --min-lr 1e-09 \
    --update-freq 8 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --share-all-embeddings \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4098 \
    --max-update 50000 \
    --eval-bleu \
    --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --patience 10