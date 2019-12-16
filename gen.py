import random

DIMENSIONS = 2
POINTS = 100000
GRIDSIZE = 10

print(POINTS, DIMENSIONS)

for p in range(POINTS):
	for d in range(DIMENSIONS):
		print((2 * GRIDSIZE * random.random()) - GRIDSIZE, end = ' ')
	print()