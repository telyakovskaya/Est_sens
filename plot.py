from ntpath import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import inv
import string
import json
import cv2                      # pip install opencv-python
from skimage import draw
from pathlib import Path
from raw_prc_pipeline.pipeline import PipelineExecutor, RawProcessingPipelineDemo
from raw_prc_pipeline.pipeline_utils import get_visible_raw_image, get_metadata
import math
import random
import statistics
import seaborn as sns

global channels, alphabet, colors_RGB, illuminants_number, patches_number, choosed_patches_number, wavelengths
wavelengths = list(range(400, 721, 10))

def plot_sens(sens, pattern='-', show=False):
    for i,c in enumerate('rgb'):
        plt.plot(wavelengths, sens[:, i], pattern, c=c)
    if show: plt.show()
    
def plot_spectra(spectras, show=False):
    for i in range(spectras.shape[-1]):
        plt.plot(wavelengths, spectras[:, i], '--')
    if show: plt.show()

def visualization(nslices, tips):
    a = np.array(tips)
    tips1 = a.reshape((nslices, -1))
    value_max = max(tips1, key=lambda item: item[1])[1]
    value_min = min(tips1, key=lambda item: item[1])[1]
    sns.set_theme()
    sns.heatmap(tips1, annot = True, vmin=value_min, vmax=value_max, center= (value_min+value_max)//2, fmt='.3g', cmap= 'coolwarm')