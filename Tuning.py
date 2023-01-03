from libsvm.tools import grid

rate, param = grid.find_parameters('D:/data/Validate.txt',  '-log2c -1,1,1 -log2g - 1,1,1 -t 1')

print(rate)
print(param)

# uses in the command line
# python grid.py  -t 0  D:/data/Validate.txt
# python grid.py  -t 1  D:/data/Validate.txt
# python grid.py  -t 2  D:/data/Validate.txt