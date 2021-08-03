import argparse
from raw_prc_pipeline.pipeline import PipelineExecutor, RawProcessingPipelineDemo
from pathlib import Path
from raw_prc_pipeline.pipeline_utils import get_visible_raw_image, get_metadata
import cv2
# from raw_prc_pipeline import expected_landscape_img_height, expected_landscape_img_width, expected_img_ext
# from utils import fraction_from_json, json_read
import numpy as np
# from functools import partial
from ntpath import join
 
 
raw_ext = '.dng'
 
def do_nothing(img, meta): return img
 
class DNGProcessingDemo():
    def __init__(self, tone_mapping, denoising_flg=True):
        self.pipeline_demo = RawProcessingPipelineDemo(
            denoise_flg=denoising_flg, tone_mapping=tone_mapping)
        self.pipeline_demo.tone_mapping = do_nothing
        # self.pipeline_demo.autocontrast = lambda img, meta: autocontrast(img, preserve_tone=True)
        self.pipeline_demo.autocontrast = lambda img, meta: img/np.max(img)

 
    def __call__(self, img_path: Path, out_path: Path):
 
        raw_image = get_visible_raw_image(str(img_path))
 
        # cv2.imwrite("C:/Users/Vasiliy/Desktop/png/3_D50+OC6.png", raw_image)
 
        metadata = get_metadata(str(img_path))
 
        # wp_dict = self.parse_csv(csv_path)
        # wp_dict['dng'] = np.array(ratios2floats(metadata['as_shot_neutral']))
        # executing img pipeline
        # pipeline_exec = PipelineExecutor(
        #         raw_image, metadata, self.pipeline_demo, last_stage='denoise')
        # # process img
        # denoised_img = pipeline_exec()
 
        # # for key, wp in wp_dict.items():
        # #     wp /= wp[1]
        # #     metadata['as_shot_neutral'] = wp
 
        pipeline_exec = PipelineExecutor(raw_image, metadata, self.pipeline_demo)
        output_image = pipeline_exec()
 
        # save results
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
 
# def main(indir, csvdir, outdir, tone_mapping, denoising_flg):
 

exposures_number = 6
for exp in range(exposures_number):
    img_path = join(r"C:\Users\adm\Documents\IITP\dng_D50", "img_" + str(8332 + exp) + ".dng")
    out_path = join(r"C:\Users\adm\Documents\IITP\jpg_D50", "img_" + str(8332 + exp) + ".jpg")

    processor = DNGProcessingDemo(tone_mapping=0, denoising_flg=0)
    processor(img_path, out_path)
