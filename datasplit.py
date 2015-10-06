from os import mkdir, path
from shutil import rmtree
from sys import argv


def padded_name(name):
    if name < 10:
        return "000"+str(name)
    elif name < 100:
        return "00"+str(name)
    elif name < 1000:
        return "0"+str(name)
    else:
        return str(name)


if len(argv) != 4:
    print "Usage: python datasplit.py <data> <output_size> <output_dir>"
    exit(1)

data = argv[1]
output_size = int(argv[2])
output_dir = argv[3]
if path.exists(output_dir):
    rmtree(output_dir)
mkdir(output_dir)
output_name = 1
output_num = 1

with open(data, 'r') as fin:
    fout = open(path.join(output_dir, padded_name(output_name)), 'w')
    output_num = 1
    for line in fin:
        fout.write(line)
        output_num += 1
        if len(line.rstrip()) == 0:
            if output_num >= output_size:
                output_name += 1
                output_num = 1
                fout.close()
                fout = open(
                    path.join(output_dir, padded_name(output_name)), 'w'
                )
