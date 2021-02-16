import json
import sys


def load_json(json_file):
    json_open = open(json_file, 'r', encoding = "utf-8")
    token_id_dict = json.load(json_open)
    return token_id_dict


if __name__ == "__main__":
    vocab_file = sys.argv[1]
    text_file = sys.argv[2]
    token_id_dict = load_json(vocab_file)

    for line in open(text_file):
        tokens = line.strip().split()
        token_ids = []
        for token in tokens:
            if token in token_id_dict:
                token = str(token_id_dict[token])
            token_ids.append(token)
        print(" ".join(token_ids))