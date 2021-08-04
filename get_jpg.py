import argparse
from raw_prc_pipeline.pipeline import PipelineExecutor
from pathlib import Path
from raw_prc_pipeline.pipeline_utils import get_visible_raw_image, get_metadata, normalize, simple_demosaic
import numpy as np
from ntpath import join
 
 
raw_ext = '.dng'
 
 
class SimpleRawPipeline:
    # Linearization not handled.
    def linearize_raw(self, raw_img, img_meta):
        return raw_img

    def normalize(self, linearized_raw, img_meta):
        norm = normalize(linearized_raw, img_meta['black_level'], img_meta['white_level'])
        return np.clip(norm, 0, 1)

    def demosaic(self, normalized, img_meta):
        return simple_demosaic(normalized, img_meta['cfa_pattern'])



class SimpleDNGProcessing():
    def __init__(self):
        self.pipeline_demo = SimpleRawPipeline()
 
    def __call__(self, img_path: Path):
        raw_image = get_visible_raw_image(str(img_path))
        
        metadata = get_metadata(str(img_path))
        metadata['cfa_pattern'] = [1,2,0,1]
 
        pipeline_exec = PipelineExecutor(
                raw_image, metadata, self.pipeline_demo)
        
        return pipeline_exec(), metadata
