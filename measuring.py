import numpy as np
import json                     # pip install opencv-python
from skimage import draw
from pathlib import Path
from raw_prc_pipeline.pipeline import PipelineExecutor
from raw_prc_pipeline.pipeline_utils import get_visible_raw_image, get_metadata, normalize, simple_demosaic

class SimpleRawProcessing:
    # Linearization not handled.
    def linearize_raw(self, raw_img, img_meta):
        return raw_img

    def normalize(self, linearized_raw, img_meta):
        norm = normalize(linearized_raw, img_meta['black_level'], img_meta['white_level'])
        return np.clip(norm , 0, 1)

    def demosaic(self, normalized, img_meta):
        return simple_demosaic(normalized, img_meta['cfa_pattern'])


class DNGProcessingDemo():
    def __init__(self):
        self.pipeline_demo = SimpleRawProcessing()
 
    def __call__(self, img_path: Path):
        raw_image = get_visible_raw_image(str(img_path))
        
        metadata = get_metadata(str(img_path))
        metadata['cfa_pattern'] = [1,2,0,1]
 
        pipeline_exec = PipelineExecutor(
                raw_image, metadata, self.pipeline_demo)
        
        return pipeline_exec()

    
def calc_variance(img, points):
    '''
    Args:
        img(np.array): img with int values
        points(list): list of regions coords
    '''
    points = np.array(points)
    y, x = draw.polygon(points[:,1], points[:,0], shape=img.shape)
    img1 = img[y, x]
    region_variance = []
    for channel in range(3):
        region_variance.append(np.std(img1[:,channel]))
    
    return region_variance


def calc_mean_color(img, points):
    '''
    Args:
        img(np.array): img with int values
        points(list): list of regions coords
    '''
    points = np.array(points)
    y, x = draw.polygon(points[:,1], points[:,0], shape=img.shape)

    # img_tmp = np.copy(img)
    # img_tmp /= np.quantile(img_tmp, 0.99)
    # img_tmp[y, x] = [1, 0, 0]
    # plt.imshow(img_tmp)
    # plt.show()
    region_color = np.mean(img[y, x], axis=0)

    return region_color


def process_markup(json_path, img):
    with open(json_path, 'r') as file:
        markup_json = json.load(file)
    color_per_region = {}
    variance_per_region = {}
    for object in markup_json['objects']:
        color_per_region[object['tags'][0]] = calc_mean_color(img, object['data'])
        variance_per_region[object['tags'][0]] = calc_variance(img, object['data'])
    return color_per_region,  variance_per_region