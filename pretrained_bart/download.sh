#!/usr/bin/env -eu

echo "download japanese bart"
wget http://lotus.kuee.kyoto-u.ac.jp/nl-resource/JapaneseBARTPretrainedModel/japanese_bart_base_1.1.tar.gz
tar -zxvf japanese_bart_base_1.1.tar.gz

echo "download english bart"
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -zxvf bart.base.tar.gz

echo "download mbart"
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
tar -zxvf /mbart.cc25.v2.tar.gz