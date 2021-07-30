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

E_df = pd.read_excel('illuminances_std.xlsx', sheet_name='Worksheet')
R_df = pd.read_excel('babelcolor.xlsx', sheet_name='Worksheet')