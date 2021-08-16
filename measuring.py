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


def measure_stimuli(patches_number=24, illuminants_number=1):
    P = np.zeros(shape=(patches_number * illuminants_number, 3))
    variances = np.zeros(shape=(patches_number * illuminants_number, 3))
    process  = DNGProcessingDemo()
    # illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC'][:illuminants_number]
    illumination_types = ['LUM']

    for illuminant in illumination_types:
        illuminant_index = illumination_types.index(illuminant)  
        img_path = join(r"C:\Users\Пользователь\Desktop\python\iipt\Calculation of sensitivities\dng", str(illuminant_index + 1) + '_' + illuminant + ".dng")
        json_path = join(r'C:\Users\Пользователь\Desktop\python\iipt\Calculation of sensitivities\png_targed', str(illuminant_index + 1) + '_' + illuminant +'.jpg.json')

        img = process(img_path).astype(np.float32)
        # img_max = np.quantile(img, 0.99)
        
        color_per_region, variance_per_region = process_markup(json_path, img)
        # cc_keys = [str(i) for i in range(1, 25)]
        # return np.asarray([color_per_region[key] for key in cc_keys])
        # carray = carray.reshape((6, 4, 3))
        
        # plt.imshow(img / img_max)
        # plt.show()
        # return carray / img_max
        # plt.imshow(carray / img_max)
        # plt.show()

        P[patches_number * illuminant_index:patches_number * illuminant_index + patches_number] = \
                        [color_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
        variances[patches_number * illuminant_index:patches_number * illuminant_index + patches_number] = \
                        [variance_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
    return P, variances

    # exposures_number = 6
    # P = np.zeros(shape=(patches_number * illuminants_number, 3, exposures_number))
    # variances = np.zeros(shape=(patches_number * illuminants_number, 3, exposures_number))

    # process  = DNGProcessingDemo()
    # illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC'][:illuminants_number]

    # for illuminant in illumination_types:
    #     illuminant_index = illumination_types.index(illuminant)
        
    #     for exp in range(exposures_number):
    #         img_path = join(r"C:\Users\adm\Documents\IITP\dng_D50", "img_" + str(8332 + exp) + ".dng")
    #         json_path = join(r"C:\Users\adm\Documents\IITP\D50_targed", "img_" + str(8332 + exp) + ".jpg.json")
    #         img = process(img_path).astype(np.float32)
            
    #         color_per_region, variance_per_region = process_markup(json_path, img)


    #         # img_max = np.quantile(img, 0.99)
    #         # cc_keys = [str(i) for i in range(1, 25)]
    #         # carray = np.asarray([color_per_region[key] for key in cc_keys])
    #         # carray = carray.reshape((6, 4, 3))
    #         # plt.imshow(img / img_max)
    #         # plt.show()
    #         # plt.imshow(carray / img_max)
    #         # plt.show()
            
    #         P[patches_number * illuminant_index:patches_number * illuminant_index + patches_number, : , exp] = \
    #             [color_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
    #         variances[patches_number * illuminant_index:patches_number * illuminant_index + patches_number, : , exp] = \
    #             [variance_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
    # return P, variances


def choose_best_stimuls(P, variances, patches_number = 24, illuminants_number = 1):
    variances_procent = np.zeros(shape=(patches_number * illuminants_number, 3, 6))
    for channel in range(3):
        for stimul in range(len(P)): 
            for exposure in range(6):
                variances_procent[stimul, channel, exposure] = \
                    variances[stimul, channel, exposure] / P[stimul, channel, exposure] * 100
   
    P_best = np.full(shape=(patches_number * illuminants_number, 3), fill_value=1.)
    variances_best = np.full(shape=(patches_number * illuminants_number, 3), fill_value=101.)

    for channel in range(3):
        for stimul in range(len(P)): 
            for exposure in range(6):
                if P[stimul, channel, exposure] <= 0.99 and variances_procent[stimul, channel, exposure] < variances_best[stimul, channel]:
                     P_best[stimul, channel] = P[stimul, channel, exposure]
                     variances_best[stimul, channel] = variances_procent[stimul, channel, exposure]
    return P_best, variances_best