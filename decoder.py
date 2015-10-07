from sys import argv


if len(argv) != 4:
    print "Usage: python decoder.py <input> <output> <dict>"
    exit(1)

input_ = argv[1]
output = argv[2]
dictionary = argv[3]

d_arr = ['0']
with open(dictionary, 'r') as d:
    for line in d:
        line = line.rstrip()
        d_arr.append(line)

with open(input_, 'r') as fin, \
    open(output+".true", 'w') as fout_true, \
    open(output, 'w') as fout:

    for line in fin:
        eos = False
        tokens = line.rstrip().split()
        for token in tokens:
            word = d_arr[int(token)]
            if word == '<eos>':
                eos = True

            fout_true.write(word)
            fout_true.write(' ')
            if not eos:
                fout.write(word)
                fout.write(' ')

        fout_true.write('\n')
        fout.write('\n')
