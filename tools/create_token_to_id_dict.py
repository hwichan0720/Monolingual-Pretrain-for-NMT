import sys
import json


if __name__ == "__main__":
    vocab_file = sys.argv[1]
    out_file = sys.argv[2]

    d = {}
    for n, line in enumerate(open(vocab_file, encoding="utf-8")):
        token = line.split()[0]
        d[token] = n  
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(d, f)