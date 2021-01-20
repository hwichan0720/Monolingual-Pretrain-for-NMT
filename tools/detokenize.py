from pyknp import Juman
import argparse


def detokenize(line):
    return line[1:].strip().replace('▁', ' ')
    # return line[1:].strip()

def tokenize(line, jumanpp):
    result = jumanpp.analysis(line)
    return ' '.join([mrph.midasi for mrph in result.mrph_list()])

def ordering(input_file):

    num = []
    line_list = []
    jumanpp = Juman()
    for line in open(input_file):
        if line[0] == "H":
            words = line[2:].strip().split()
            # print(words[0])
            idx = int(words[0])
            detok_line = detokenize(''.join(words[2:]))
            line_list.append((idx, detok_line))
    line_list = sorted(line_list)

    for n, line in enumerate(line_list):
        print(line[1].strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file")
    args = parser.parse_args()
    input_file = args.input_file
    ordering(input_file)
