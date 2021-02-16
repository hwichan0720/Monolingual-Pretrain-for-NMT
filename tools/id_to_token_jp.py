import json
from pyknp import Juman
import sys
import re

def load_json(json_file):
    json_open = open(json_file, 'r', encoding = "utf-8")
    token_id_dict = json.load(json_open)
    return token_id_dict

def get_swap_dict(d):
    return {v: k for k, v in d.items()}

def decode(line, d):
    tokens = line.strip().split()
    words = []
    for token in tokens:
        if isdigit(token):
            token_id = int(token)
            if token_id in d:
                words.append(d[token_id])
                continue
        words.append(token)
    return "".join(words)

#半角数字
digitReg = re.compile(r'^[0-9]+$')
def isdigit(s):
    return digitReg.match(s) is not None

def remove_bpe(line):
    return line.strip().replace('▁', '')

def tokenize(line, jumanpp):
    result = jumanpp.analysis(line)
    return ' '.join([mrph.midasi for mrph in result.mrph_list()])

if __name__ == "__main__":
    joson_file = sys.argv[1]
    token_id_dict = load_json(joson_file)
    id_token_dict = get_swap_dict(token_id_dict)

    jumanpp = Juman()
    for line in open(sys.argv[2]):
        decode_line = decode(line.strip(), id_token_dict)
        detok_line = remove_bpe(decode_line[1:])
        print(tokenize(detok_line, jumanpp))

# if __name__ == "__main__":
#     vocab_file = sys.argv[1]
#     text_file = sys.argv[2]
#     token_id_dict = load_json(vocab_file)

#     for line in open(text_file):
#         tokens = line.strip().split()
#         token_ids = []
#         for token in tokens:
#             if token in token_id_dict:
#                 token = str(token_id_dict[token])
#             token_ids.append(token)
#         print(" ".join(token_ids))