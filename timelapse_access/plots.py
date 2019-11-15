#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import io 
import base64

def small_heatmap(data, figsize=(4,1), xylabels=('Time', 'Intensity'), **kwargs): 
    """Heatmap as an rgb"""
    fig, ax = plt.subplots(1,1, figsize=(figsize), **kwargs)
    ax.imshow(data)
    ax.set(xlabel=xylabels[0], ylabel=xylabels[1], xticks=[], yticks=[])
    for k,v in ax.spines.items():
        v.set_visible(False)
    plt.subplots_adjust(0,0,1,1)
    ax.margins(0)
    return fig

def fig_to_array(fig):
    fig.canvas.draw()
    raw, (w,h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(raw, dtype='uint8').reshape((w,h,4))
    return img

def fig_to_base64(fig):	
    img = io.BytesIO()
    fig.savefig(img, transparent=True, bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode("UTF-8")
    return encoded
