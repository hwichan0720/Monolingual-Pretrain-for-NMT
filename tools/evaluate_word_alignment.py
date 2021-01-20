import numpy as np
import sys


def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (n1 * n2)

def load_dict(dict_f):
    words = []
    for line in open(dic_f):
        line = line.strip().split()
        words.append(line[0])
    
    return words

def load_emb_from_text(emb_f):
    d = {}
    for line in open(emb_f):
        line = line.strip().split()
        if len(line) == 2:
            continue
        word = line[0]
        vec = np.array(line[1:], dtype=float)
        d[word] = vec
    
    return d

def search_most_similar_word(v1, embs):
    best_score = None
    best_word = None

    for word, v2 in embs.items():
        sim = similarity(v1, v2)
        if best_score == None or sim > best_score:
            best_score = sim
            best_word = word
    
    return best_word, best_score

if __name__ == "__main__":
    ko_emb_f = sys.argv[1]
    ja_emb_f = sys.argv[2]
    dic_f = sys.argv[3] 

    ko_embs = load_emb_from_text(ko_emb_f)
    ja_embs = load_emb_from_text(ja_emb_f)
    words = load_dict(dic_f)

    total_sim = 0
    score_pair =[]
    for ko_word in words:
        ko_vec = ko_embs[ko_word]
        ja_word, score = search_most_similar_word(ko_vec, ja_embs)
        print(ko_word, ja_word, score)
        