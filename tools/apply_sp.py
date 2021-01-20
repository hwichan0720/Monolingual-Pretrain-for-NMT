import sentencepiece as spm
import argparse


def train(model_path, train_file, vocab_size):
    spm.SentencePieceTrainer.Train(
        f'--input={train_file}, --model_prefix={model_path} --character_coverage=0.9995 --vocab_size={vocab_size}'
    )

def apply(model_file, input_file):
    s = spm.SentencePieceProcessor(model_file=model_file)
    for line in open(input_file):
        line = line.strip()
        subwords = s.EncodeAsPieces(line)
        print(" ".join(subwords))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='apply sentencepiece')
    parser.add_argument('-m', '--model_path')
    parser.add_argument('-i', '--input_file')
    parser.add_argument('-v', '--vocab_size', type=int, default=32000)
    parser.add_argument('-f', '--flag', type=bool, default=False)
    args = parser.parse_args()

    # print(args.flag)
    # print(args.vocab_size)
    if args.flag:
        train(model_path=args.model_path, train_file=args.input_file, vocab_size=args.vocab_size)
    else:
        apply(model_file=f"{args.model_path}.model", input_file=args.input_file)

