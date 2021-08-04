from pathlib import Path
import numpy as np
from awb.io import imread, imsave
from get_jpg import SimpleDNGProcessing
import matplotlib.pyplot as plt
import json
from tqdm  import tqdm

indir = Path(r'D:\Dev\crs\est_sens\imgs\DNG')
outdir = Path(r'D:\Dev\crs\est_sens\imgs\PNG')
outdir.mkdir(parents=True, exist_ok=True)

imgs_paths = list(indir.glob('*dng'))

prc = SimpleDNGProcessing()

for path in tqdm(imgs_paths):
    img, metadata = prc(path)
    out_metadata = {
        'exp': metadata['exp'],
        'iso': metadata['iso']}
    
    with open(outdir / path.with_suffix('.json').name, 'w') as fh:
        json.dump(out_metadata, fh, indent=2)
    
    imsave(outdir / path.with_suffix('.png').name, img, unchanged=False, out_dtype=np.uint16, gamma_correction=False)
    
    