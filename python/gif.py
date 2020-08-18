from PIL import Image, ImageDraw
import glob, os
import re
import sys

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

images = []

if len(sys.argv) != 2:
    print("You have to specify folder where to find images!")
    exit(1)

os.chdir(str(sys.argv[1]))
for file in sorted(glob.glob("*.png"), key=natural_keys):
    image = Image.open(file)
    images.append(image)

images[0].save('gif.gif', save_all=True, append_images=images[1:], optimize=False, duration=120, loop=0)