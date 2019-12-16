import random

import sys

DIMENSIONS = 2
POINTS = 100000
GRIDSIZE = 10

if len(sys.argv) == 3:
	DIMENSIONS = int(sys.argv[1])
	POINTS = int(sys.argv[2])

print(POINTS, DIMENSIONS)

for p in range(POINTS):
	for d in range(DIMENSIONS):
		print((2 * GRIDSIZE * random.random()) - GRIDSIZE, end = ' ')
	print()