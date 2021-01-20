import zenhan
import sys


if __name__ == "__main__":
    for line in open(sys.argv[1]):
        line = line.strip()
        print(zenhan.z2h(line))