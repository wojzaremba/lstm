from sys import argv


if len(argv) != 4 and len(argv) != 5:
    print "Usage: python decoder.py <input> <output> <dict> [output_true y|N]"
    exit(1)

input_ = argv[1]
output = argv[2]
dictionary = argv[3]
if len(argv) == 5:
    output_true = argv[4]
    if output_true == 'y':
        output_true = True
    else:
        output_true = False
else:
    output_true = False

d_arr = ['0']
with open(dictionary, 'r') as d:
    for line in d:
        line = line.rstrip()
        d_arr.append(line)

with open(input_, 'r') as fin, \
    open(output, 'w') as fout:

    if output_true:
        fout_true = open(output+".true", 'w')

    for line in fin:
        eos = False
        tokens = line.rstrip().split()
        for token in tokens:
            word = d_arr[int(token)]
            if word == '<eos>':
                eos = True

            if output_true:
                fout_true.write(word)
                fout_true.write(' ')
            if not eos:
                fout.write(word)
                fout.write(' ')

        if output_true:
            fout_true.write('\n')
        fout.write('\n')

    if output_true:
        fout_true.close()
