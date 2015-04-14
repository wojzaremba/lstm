txt = open('ptb.char.train.txt').read().split()
unique = set(txt)
print "Number of unique characters:", len(unique)
print unique
