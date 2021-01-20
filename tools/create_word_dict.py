import argparse
import sys
import random
from collections import defaultdict

def create_para_file(src_file, tgt_file):
    for src, tgt in zip(open(src_file), open(tgt_file)):
        print(f"{' '.join(src.strip().split())} ||| {' '.join(tgt.strip().split())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create word dictionary')
    parser.add_argument('-p', '--parallel_file')
    parser.add_argument('-a', '--align_file')
    parser.add_argument("-f", "--flag", action='store_true')
    parser.add_argument('-s', '--src')
    parser.add_argument('-t', '--tgt')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    if args.flag:
        create_para_file(src_file=args.src, tgt_file=args.tgt)
    else:
        align_file = args.align_file
        para_file = args.parallel_file

        line_idxs = []
        dic = defaultdict(int)
        for align_line, para_line in zip(open(align_file), open(para_file)):
            idxs = align_line.strip().split()
            idxs = [idx.split("-") for idx in idxs]
            
            lines = para_line.strip().split("|||")
            words1 = lines[0].split()
            words2 = lines[1].split()
            
            for idx in idxs:
                dic[f"{words1[int(idx[0])]} {words2[int(idx[1])]}"] += 1

        pairs = [key for key, value in sorted(dic.items(), key=lambda x:x[1], reverse=True)[:10000]]
        valid_paris = random.sample(list(range(len(pairs))), 1000)
        
        with open(f"{args.output}.train", "w") as f, open(f"{args.output}.val", "w") as val_f:
            for n, key in enumerate(pairs):
                if n in valid_paris:
                    print(key, file=val_f)
                    continue
                print(key, file=f)