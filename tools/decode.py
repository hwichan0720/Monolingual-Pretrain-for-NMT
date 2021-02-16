import sys
import json
import re

def load_json(json_file):
    json_open = open(json_file, 'r', encoding = "utf-8")
    token_id_dict = json.load(json_open)
    return token_id_dict

def get_swap_dict(d):
    return {v: k for k, v in d.items()}


def decode(line, d):
    ids = line.strip().split()
    words = []
    for token_id in ids:
        if not isdigit(token_id):
            words.append(token_id)
            continue
        words.append(d[int(token_id)])
    return "".join(words)

def remove_bpe(line):
    return line.replace("Ġ", " ")

#半角数字
digitReg = re.compile(r'^[0-9]+$')
def isdigit(s):
    return digitReg.match(s) is not None

if __name__ == "__main__":
    joson_file = sys.argv[1]
    token_id_dict = load_json(joson_file)
    id_token_dict = get_swap_dict(token_id_dict)

    for line in open(sys.argv[2]):
        decode_line = decode(line.strip(), id_token_dict)
        print(remove_bpe(decode_line))