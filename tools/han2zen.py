import argparse
import zenhan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='han2zen')
    parser.add_argument('-i', '--input_file')
    args = parser.parse_args()

    for line in open(args.input_file):
        line = line.strip().split()
        word = line[0]
        score = line[1]
        zen_word = word[0] + zenhan.h2z(word[1:])

        print(f"{zen_word} {score}")