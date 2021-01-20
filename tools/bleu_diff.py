import sacrebleu
import sys
import argparse


def cal_sentence_bleu(translation_file1, translation_file2, reference_file, input_file1, input_file2, output):

    bleu_list =[]

    for n, (tran1, tran2, ref) in enumerate(zip(open(translation_file1, encoding='utf8'), \
                                            open(translation_file2, encoding='utf8'),  \
                                            open(reference_file, encoding='utf8'))):

        tran1_bleu = sacrebleu.corpus_bleu(tran1, ref).score
        tran2_bleu = sacrebleu.corpus_bleu(tran2, ref).score
        bleu_list.append((tran1_bleu - tran2_bleu, tran1_bleu, tran2_bleu, n))
    bleu_list.sort(reverse=True)
    
    with open(translation_file1) as tran1, open(translation_file2) as tran2, open(reference_file) as ref, \
         open(input_file1) as input1, open(input_file2) as input2:
        tran1_lines = tran1.readlines()
        tran2_lines = tran2.readlines()
        ref_lines = ref.readlines()
        input1_lines = input1.readlines()
        input2_lines = input2.readlines()

    line_num = len(ref_lines)
    best_num = int(line_num / 3)
    worst_num = line_num - int(line_num / 3)

    with open(output, 'w', encoding='utf8') as out, \
            open(f'best{best_num}', 'w', encoding='utf8') as best, \
            open(f'worst{worst_num}', 'w', encoding='utf8') as worst:

        for num, (bleu, tran1_bleu, tran2_bleu, n) in enumerate(bleu_list, 1):
            print(f'{bleu}, {tran1_bleu}, {tran2_bleu} \n'
                  + f'Reference : {ref_lines[n]}'
                  + f'{translation_file1}: \n'
                  + input1_lines[n]
                  + tran1_lines[n]
                  + f'{translation_file2}: \n'
                  + input2_lines[n]
                  + tran2_lines[n], file=out)
            if num <= best_num:
                print(f'{bleu}, {tran1_bleu}, {tran2_bleu} \n'
                        + f'Reference : {ref_lines[n]}'
                        + f'{translation_file1}: \n'
                        + input1_lines[n]
                        + tran1_lines[n]
                        + f'{translation_file2}: \n'
                        + input2_lines[n]
                        + tran2_lines[n], file=best)
            elif num >= worst_num:
                print(f'{bleu}, {tran1_bleu}, {tran2_bleu} \n'
                        + f'Reference : {ref_lines[n]}'
                        + f'{translation_file1}: \n'
                        + input1_lines[n]
                        + tran1_lines[n]
                        + f'{translation_file2}: \n'
                        + input2_lines[n]
                        + tran2_lines[n], file=worst)
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", "--trainslation1")
    parser.add_argument("-t2", "--trainslation2")
    parser.add_argument("-ref", "--reference")
    parser.add_argument("-i1", "--input1")
    parser.add_argument("-i2", "--input2")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    translation_file1 = args.trainslation1
    translation_file2 = args.trainslation2
    reference_file = args.reference
    input_file1 = args.input1
    input_file2 = args.input2
    output = args.output
    cal_sentence_bleu(translation_file1, translation_file2, reference_file, input_file1, input_file2, output)
