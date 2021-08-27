import glob
import os

files = glob.glob("out/source_images/*.png")


for f in files:
    idn = f.split('_')[-1].split('.')[0]
    name_end = f.split('_')[-1]

    path_begin = f.replace(name_end, '')

    if len(idn) == 1:
        idn = "00" + idn
    elif len(idn) == 2:
        idn = "0" + idn
    
    new_path = path_begin + idn + ".png"
    
    os.rename(f, new_path)
