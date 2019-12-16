
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd

import seaborn
from  matplotlib import pyplot

with open('../output.txt') as f:
    lines = f.read().splitlines()
    
for i in range(0,len(lines),3):
    x = lines[i].split(', ')[:-2]
    y = lines[i+1].split(', ')[:-2]
    c = lines[i+2].split(', ')[:-2]
    

    df = pd.DataFrame({
        'x': x,
        'y': y,
        'c': c
    })
    
    df = df.astype(np.float16)
    
    fg = seaborn.FacetGrid(data=df, hue='c', height = 8, aspect=1)
    fg.map(pyplot.scatter, 'x', 'y')
    fg.savefig("pics/output{}.png".format(i))