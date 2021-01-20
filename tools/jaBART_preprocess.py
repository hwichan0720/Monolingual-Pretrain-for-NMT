import argparse
from pyknp import Juman
import sentencepiece as spm
import zenhan


def main(model, sp_dict, input_file, output_file, l1, l2):
    jumanpp = Juman()
    s = spm.SentencePieceProcessor()
    s.Load(model)
    vocabs = []
    with open(sp_dict) as f:
        for line in f:
            vocabs.append(line.strip().split()[0])
    s.set_vocabulary(vocabs)

    # tokenize japanese and han2zen korean
    with open(f"{output_file}.tok.ja", "w") as ja_tok_f, open(f"{output_file}.ja", "w") as ja_sub_f,\
         open(f"{output_file}.tmp.ko", "w") as ko_f:
        for n, (ja_line, ko_line) in enumerate(zip(open(f"{input_file}.ja"), open(f"{input_file}.ko"))):

            if (n+1) % 100000 == 0: 
                print(f"done {n+1} lines")
            ja_line = ja_line.strip()
            ko_line = ko_line.strip()
            
            ja_line = zenhan.h2z(ja_line)
            if len(ja_line.encode('utf-8')) > 4096:
                continue
            
            ko_line = " ".join(ko_line.split())
            print(ko_line, file=ko_f)
            
            # tokenize japanese
            result =  jumanpp.analysis(ja_line)
            words = [mrph.midasi for mrph in result.mrph_list()]
            tokenized_line = " ".join(words)
            print(tokenized_line, file=ja_tok_f)

            # tokenize japanese to subwords 
            subwords = s.EncodeAsPieces(tokenized_line)
            print(" ".join(subwords), file=ja_sub_f)



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