from libsvm.tools import grid

rate, param = grid.find_parameters('D:/data/Validate.txt',  '-log2c -1,1,1 -log2g - 1,1,1 -t 1')

print(rate)
print(param)