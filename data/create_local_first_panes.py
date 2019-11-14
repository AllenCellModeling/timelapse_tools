#!/usr/bin/env python

"""Create a local cache of the first several frames from each of the movies
listed in `raw_image_list.csv
"""

import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from aicspylibczi import CziFile
from PIL import Image
import imageio
import tqdm

df = pd.read_csv('./raw_image_list.csv')
fns = ['/allen'+fn for fn in list(df['Isilon path']+df['File Name'])]

def fn_to_np(fn, frames=5):
    try:
        out = './first_panes/'+fn.split('/')[-1]+'.npy'
        time_stack = []
        for i in range(frames):
            stack, shape = CziFile(fn).read_image(T=i, B=0, S=0)
            time_stack.append(stack[0,0,0].max(axis=0))
        time_stack = np.array(time_stack)
        np.save(out, time_stack)
    except:
        print(f"Error with file {fn}")
        traceback.print_exc()

for fn in fns:
    fn_to_np(fn)

