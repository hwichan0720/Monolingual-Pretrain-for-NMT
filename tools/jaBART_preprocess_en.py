import argparse
from pyknp import Juman
import sentencepiece as spm
import zenhan


def main(model, sp_dict, input_file, output_file, l2, l1="ja"):
    jumanpp = Juman()
    s = spm.SentencePieceProcessor()
    s.Load(model)
    vocabs = []
    with open(sp_dict) as f:
        for line in f:
            vocabs.append(line.strip().split()[0])
    s.set_vocabulary(vocabs)

    # tokenize japanese 
    with open(f"{output_file}.tok.{l1}", "w", encoding='utf-8') as l1_tok_f, open(f"{output_file}.{l1}", "w", encoding='utf-8') as l1_sub_f,\
         open(f"{output_file}.tmp.{l2}", "w", encoding='utf-8') as l2_f:
        for n, (l1_line, l2_line) in enumerate(zip(open(f"{input_file}.{l1}"), open(f"{input_file}.{l2}"))):
            if (n+1) % 100000 == 0: 
                print(f"done {n+1} lines")
            l1_line = l1_line.strip()
            l2_line = l2_line.strip()
            
            l1_line = zenhan.h2z(l1_line)
            if len(l1_line.encode('utf-8')) > 4096:
                continue
            
            print(l2_line, file=l2_f)
            
            # tokenize japanese
            result =  jumanpp.analysis(l1_line)
            words = [mrph.midasi for mrph in result.mrph_list()]
            tokenized_line = " ".join(words)
            print(tokenized_line, file=l1_tok_f)

            # tokenize japanese to subwords 
            subwords = s.EncodeAsPieces(tokenized_line)
            print(" ".join(subwords), file=l1_sub_f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess japanese')
    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--dict')
    parser.add_argument('-i', '--input_file')
    parser.add_argument('-o', '--output_file')
    parser.add_argument('-l1', '--language1')
    parser.add_argument('-l2', '--language2')

    args = parser.parse_args()

    main(model=args.model, sp_dict=args.dict, input_file=args.input_file, \
        output_file=args.output_file, l1=args.language1, l2=args.language2)