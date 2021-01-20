import sys
import sacrebleu
import mojimoji


if __name__ == "__main__":
    outs_path = sys.argv[1]
    refs_path = sys.argv[2]

    bleus = []
    for out, ref in zip(open(outs_path), open(refs_path)):
        out = mojimoji.zen_to_han(out.strip())
        ref = mojimoji.zen_to_han(ref.strip())
        bleu = sacrebleu.corpus_bleu(out, ref)
        bleus.append((bleu.score, ref, out))
    
    bleus = sorted(bleus, reverse=True)
    total = 0
    for bleu, ref, out in bleus:
        print(f"{bleu}\nref:{ref}\nout:{out}")
        total += bleu
    
    print(total / len(bleus))