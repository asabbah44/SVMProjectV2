from libsvm.tools import grid

rate, param = grid.find_parameters('D:/data/Validate.txt', '-log2c 0,1,1 -log2g 0,1,1 -t 2')

print(rate)
print(param)