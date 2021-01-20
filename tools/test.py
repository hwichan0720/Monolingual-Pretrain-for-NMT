import sys
for n, line in enumerate(open(sys.argv[1])):
    if n == 5:
        break
    print(line)