import random

DIMENSIONS = 5
POINTS = 1000000
GRIDSIZE = 1000

print(POINTS, DIMENSIONS)

for p in range(POINTS):
	for d in range(DIMENSIONS):
		print((2 * GRIDSIZE * random.random()) - GRIDSIZE, end = ' ')
	print()