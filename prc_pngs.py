from pathlib import Path
import numpy as np
from img_io import imread, imsave
from get_jpg import SimpleDNGProcessing
import matplotlib.pyplot as plt
import json
from tqdm  import tqdm
from skimage import draw
from main import draw_colorchecker
from plot import visualization


indir = Path(r'D:\Dev\crs\est_sens\imgs\PNG')
outdir = Path(r'D:\Dev\crs\est_sens\imgs')
outdir.mkdir(parents=True, exist_ok=True)

vis_markup_dir = Path(r'D:\Dev\crs\est_sens\imgs\vis_markup_dir')
vis_markup_dir.mkdir(parents=True, exist_ok=True)

vis_cc_dir = Path(r'D:\Dev\crs\est_sens\imgs\vis_cc_dir')
vis_cc_dir.mkdir(parents=True, exist_ok=True)

vis_noise_dir = Path(r'D:\Dev\crs\est_sens\imgs\vis_noise_dir')
vis_noise_dir.mkdir(parents=True, exist_ok=True)

imgs_paths = list(indir.glob('*.png'))[:4]


markup_path = Path(r'D:\Dev\crs\est_sens\imgs\markup\IMG_8348.png.json')

with open(markup_path, 'r') as fh:
    markup = json.load(fh)


def process_markup(markup, img):
    xy_dict = {}
    for object in markup['objects']:
        name =  object['tags'][0]
        points = np.array(object['data'])
        xy_dict[name] = draw.polygon(points[:,1], points[:,0], shape=img.shape)
    return xy_dict


debug = True

def vis_markup(xy_dict, img, out_path=None):
    if out_path is None: return
    
    img_m = np.copy(img)
    for (x, y) in xy_dict.values():
       img_m[x,y] = [1, 0, 0]
    imsave(out_path, img_m, unchanged=False, out_dtype=np.uint8, gamma_correction=True)  


def vis_cc(cc_mean_array, out_path=None):
    if out_path is None: return
    img = draw_colorchecker(cc_mean_array, show=False)
    plt.imshow(img)
    plt.savefig(out_path)
    
    
def vis_noise(cc_mean_array, cc_std_array, out_path, max_val = 10):
    errors = (cc_std_array / cc_mean_array * 100).reshape(6, 4, 3)
    fig, axes = plt.subplots(1, 3)
    
    for i in range(3):
        im = axes[i].imshow(errors[..., i], vmin=0, vmax=max_val)
        # fig.colorbar(im)
    # fig.clim(0, 50)
    # plt.show()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()



def xy_dict_to_color_dict(xy_dict, img):
    res_dict = {}
    for patch_name, (x, y) in xy_dict.items():
        res_dict[patch_name] = img[x,y]
        
    return res_dict


min_exp = 0.125
norm_coeff = None 


cc_all_dict = {}



for path in tqdm(imgs_paths):
    img = imread(path, out_dtype=np.float32, linearize_srgb=False)   
    xy_dict = process_markup(markup, img)
    vis_markup(xy_dict, img, vis_markup_dir / path.with_suffix('.m.jpg').name)
    
    with open(path.with_suffix('.json'), 'r') as fh:
        meta = json.load(fh)
    exp = meta['exp']
    
    if not exp in cc_all_dict:
        cc_all_dict[exp] = {}
        
    cc_dict = xy_dict_to_color_dict(xy_dict, img)
    # adjust explosure time
    cc_dict = {key: val * (min_exp / exp)  for key, val in cc_dict.items()}
    
    for key, val in cc_dict.items():
        if key not in cc_all_dict[exp]:
            cc_all_dict[exp][key] = []
        cc_all_dict[exp][key].append(val)
    
    # #calc mean, std 
    # cc_mean_array = np.asarray([np.mean(cc_dict[str(i)], axis=0) for i in range(1, 25)])
    
    # #calc normalize coeff
    # norm_coeff = np.max(cc_mean_array[-1]) if norm_coeff is None else norm_coeff

    # cc_std_array = np.asarray([np.std(cc_dict[str(i)], axis=0) for i in range(1, 25)])

    # cc_mean_array /= norm_coeff
    # cc_std_array /= norm_coeff

    # vis_noise(cc_mean_array, cc_std_array, vis_noise_dir / path.with_suffix('.n.jpg').name)

    # vis_cc(cc_mean_array, vis_cc_dir / path.with_suffix('.cc.jpg').name)
    # vis_cc(cc_mean_array/np.sum(cc_mean_array, axis=-1)[..., np.newaxis], vis_cc_dir / path.with_suffix('.ch.jpg').name)
    
        
# print(cc_all_dict)


vis_fin_dir = Path(r'D:\Dev\crs\est_sens\imgs\vis_fin_dir')
vis_fin_dir.mkdir(parents=True, exist_ok=True)

for exp, cc_dict in cc_all_dict.items():
    #calc mean
    print(np.asarray(cc_dict['1']).shape)
    cc_mean_array = np.asarray([np.mean(np.asarray(cc_dict[str(i)]), axis=(0,1)) for i in range(1, 25)])
    #calc normalize coeff
    norm_coeff = np.max(cc_mean_array[-1]) if norm_coeff is None else norm_coeff
    # calc std
    
    
    cc_std_array = np.asarray([np.std(np.asarray(cc_dict[str(i)]), axis=(0, 1)) for i in range(1, 25)])
    errors = (cc_std_array / cc_mean_array * 100).reshape(6, 4, 3)
    
    cc_std_array_tmp = np.asarray([np.std(cc_dict[str(i)][0], axis=0) for i in range(1, 25)])
    errors_tmp = (cc_std_array_tmp / cc_mean_array * 100).reshape(6, 4, 3)
    
    print()
    for i in range(6):
        for j in range(4):
            print(f'all={errors[i,j]}, one={errors_tmp[i,j]}')
            
    # exit()

    cc_mean_array /= norm_coeff
    cc_std_array /= norm_coeff
    
    cc_mean_array /= norm_coeff
    
    cc_std_array /= norm_coeff

    imsave(vis_fin_dir / (str(exp) + '.mean.tiff'),  cc_mean_array, unchanged=True)
    imsave(vis_fin_dir / (str(exp) + '.std.tiff'),  cc_std_array, unchanged=True)

    vis_noise(cc_mean_array, cc_std_array, vis_fin_dir / (str(exp) + '.n.jpg'))

    vis_cc(cc_mean_array, vis_fin_dir / (str(exp) + '.cc.jpg'))
    vis_cc(cc_mean_array/np.sum(cc_mean_array, axis=-1)[..., np.newaxis], vis_fin_dir / (str(exp) + '.ch.jpg'))
    
    

    

    
    
    