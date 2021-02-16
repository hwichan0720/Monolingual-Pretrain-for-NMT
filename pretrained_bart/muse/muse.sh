GPU=$1
ROOT=../..
fastText=../../../fastText
fastalign=../../../fast_align/build
muse=../../../MUSE
data=$ROOT/data
tools=$ROOT/tools
jpbart=../japanese_bart_base_1.1
mapping_vector=$dir/muse-koja/debug/ko-ja/vectors-ko.txt
dir=koja
para_file=$dir/para_file.ko-ja
epoch=10
ws=10
ko_embs=$dir/ko_embs_${epoch}_${ws}

mkdir -p $dir

echo "careate parallel file for fastalign"
python3 $tools/create_word_dict.py -s $data/train.ko -t $data/train.ja -f > $para_file

echo "apply fastalign"
$fastalign/fast_align -i $para_file -d -o -v > $para_file.align

echo "create word dictionary using aligned file"
python3 $tools/create_word_dict.py -p $para_file -a $para_file.align -o $dir/dict.ko-ja

echo "train korean emb using fasttext"
$fastText/fasttext skipgram -input $data/train.ko -output $ko_embs -dim 768 -epoch $epoch -ws $ws

echo "extract japanese bart vec for muse"
python3 $tools/extract_japanese_vec.py --pre-train-dir $jpbart --ft-dict $data/bin/dict.ja.txt > $dir/ja_bart.vec.txt

echo "train mapping target to source space"
CUDA_VISIBLE_DEVICES=$GPU python3 $muse/supervised.py --src_lang ko --tgt_lang ja --emb_dim 768 --src_emb $ko_embs.vec --tgt_emb $dir/ja_bart.vec.txt --n_refinement 5 --dico_train $dir/dict.ko-ja.train --dico_eval $dir/dict.ko-ja.val --exp_path $dir/muse-koja

echo "merge mapping korean vectors to japanese bart"
python $tools/trim_bart_muse.py --pre-train-dir $jpbart --ft-dict $data/bin/dict.ja.txt --output muse_bart.pt --muse-model $mapping_vector

echo "get alined word dict (fastalign)"
python $tools/create_word_dict_v2.py -p $para_file -a $para_file.align > $dir/dict.ko-ja.all

echo "trim japanese bart with alined word dict"
python $tools/trim_bart_fastalign.py --pre-train-dir $jpbart --ft-dict $data/bin/dict.ja.txt --output $dir/fastalign_bart.pt --dict $dir/dict.ko-ja.all