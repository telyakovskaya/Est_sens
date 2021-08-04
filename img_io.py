from os import PathLike
from typing import Any, Union
import numpy as np
import tifffile
import cv2

raw_suffixes = ['.dng', '.cr2']
tiff_suffixes = ['.tiff', '.tif']


def srgb2lin_rgb(img):
    """
    Reverse gamma-correction
    """
    res = np.zeros_like(img)
    res[img <= 0.04045] = 25 * img[img <= 0.04045] /323
    res[img > 0.04045] = ((200 * img[img > 0.04045] + 11) / 211) ** (12 / 5)
    return res

def lin_rgb2srgb(img):
    """
    Gamma-correction
    """
    if not np.issubdtype(img.dtype, np.floating):
        raise RuntimeError(
            f'Gamma correction can be performed only when img.dtype is floating!')
    if img.max() > 1 or img.min() < 0:
        raise ValueError(
            f'Warning: image pixel values should be from 0 to 1!')
        
    res = np.zeros_like(img)
    res[img <= 0.0031308] = 12.92 * img[img <= 0.0031308]
    res[img > 0.0031308] = 1.055 * img[img > 0.0031308] ** (1 / 2.4) - 0.055
    return res

def uint2real(img: np.dtype, out_dtype=np.float32):
    if not np.issubdtype(out_dtype, np.floating):
        raise TypeError(
            f'Unsupported out_dtype={out_dtype}, dtype should be floating!')
    out_img = img.astype(out_dtype)

    if not np.issubdtype(img.dtype, np.unsignedinteger):
        raise TypeError(
            f'Unsupported img.dtype={img.dtype}, dtype should be unsignedinteger!')

    max_val = np.iinfo(img.dtype).max
    min_val = np.iinfo(img.dtype).min
    out_img = (img - min_val)/(max_val - min_val)
    return out_img


def real2uint(img: np.ndarray, out_dtype=np.uint8):
    if not np.issubdtype(img.dtype, np.floating):
        raise TypeError(
            f'Unsupported img.dtype={img.dtype}, dtype should be floating!')

    if not np.issubdtype(out_dtype, np.unsignedinteger):
        raise TypeError(
            f'Unsupported out_dtype={out_dtype}, dtype should be unsigned integer!')

    # clip image
    if img.max() > 1 or img.min() < 0:
        print(f'Warning: image pixel values will be clipped to [0, 1]!')
    out_img = np.clip(img, 0, 1)
    # normalize
    max_val = np.iinfo(out_dtype).max
    min_val = np.iinfo(out_dtype).min
    out_img = img*(max_val - min_val) + min_val
    out_img = out_img.astype(out_dtype)
    return out_img


def uint2uint(img: np.ndarray, out_dtype=np.uint8):
    if not np.issubdtype(img.dtype, np.unsignedinteger):
        raise TypeError(
            f'Unsupported img.dtype={img.dtype}, dtype should be unsignedinteger!')

    if not np.issubdtype(out_dtype, np.unsignedinteger):
        raise TypeError(
            f'Unsupported out_dtype={out_dtype}, dtype should be unsigned integer!')

    real_img = uint2real(img, out_dtype=np.float32)
    return real2uint(real_img, out_dtype=out_dtype)


def swap_rb(img):
    ch_order = np.arange(img.shape[-1], dtype=np.int32)
    ch_order[0], ch_order[2] = ch_order[2], ch_order[0]
    return img[..., ch_order]


def imread(path: Union[PathLike, str], out_dtype: np.floating = None, linearize_srgb=False) -> np.ndarray:
    """
    Return image data from file as np.ndarray.

    Parameters
    ----------
    path : Union[PathLike, str]
        Path to image file.
    out_dtype : np.floating, optional
        The np.dtype of the returned data.
        It is assumed to be np.floating.
        Returned image dtype will be changed to out_dtype using `uint2real` function.
        If out_dtype is None or image file is tiff file, data returns as is.
        By default None.
    linearize_srgb : bool, optional
        If out_dtype is not None and linearize_srgb is True, the linearization of the pixel value will be performed (see `srgb2lin_rgb` function).
        By default False.

    Returns
    -------
    np.ndarray
        Returned image data.

    Raises
    ------
    RuntimeError
        If image file does not exist.
    NotImplemented
        If image file format is one of the raw formats (see `raw_suffixes`).
    TypeError
        If image file is not tiff or raw file and image dtype is not unsignedinteger.
    TypeError
        If image file is not tiff or raw file and out_dtype is not None.
    """
    if not path.is_file():
        raise RuntimeError(f'File {path} does not exist!')
    
    if path.suffix.lower() in raw_suffixes:
        raise NotImplemented(
            f'Reading raw formats {raw_suffixes} is not implemented!')
    elif path.suffix.lower() in tiff_suffixes:
        return tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if len(img.shape) >= 3:
            img = swap_rb(img)

        if out_dtype is None:
            return img

        if not np.issubdtype(img.dtype, np.unsignedinteger):
            raise TypeError(
                f'Unsupported img.dtype={img.dtype}, dtype should be unsignedinteger!')

        # integer to real image
        if not np.issubdtype(out_dtype, np.floating):
            raise TypeError(
                f'Unsupported out_dtype={out_dtype}, dtype should be floating!')

        img = uint2real(img, out_dtype=out_dtype)
        if linearize_srgb:
            # tranform from sRGB to linear sRGB
            img = srgb2lin_rgb(img)
        return img


def imsave(path: Union[PathLike, str], img: np.ndarray, unchanged=True, out_dtype: np.unsignedinteger = np.uint8, gamma_correction=False):
    """
    Save np.ndarray to image file.

    Parameters
    ----------
    path : Union[PathLike, str]
        Path to file.
    img : np.ndarray
        Input image.
    unchanged : bool, optional
        If image file is tiff or unchanged flag is True, data saves as is into the file.
        By default True.
    out_dtype : np.unsignedinteger, optional
        If unchanged is False, image dtype will be changed to out_dtype using `real2uint` or `uint2uint` functions.
        By default np.uint8.
    gamma_correction : bool, optional
        If True, gamma correction will be performed before saving (see `lin_rgb2srgb` for details).
        By default False.

    Raises
    ------
    NotImplemented
        Saving into raw formats is not implemented (see `raw_suffixes`).
    TypeError
        Image dtype should be floating or unsignedinteger.
    """
    if path.suffix.lower() in raw_suffixes:
        raise NotImplemented(
            f'Saving into raw formats {raw_suffixes} is not implemented!')
    elif path.suffix.lower() in tiff_suffixes:
        tifffile.imsave(str(path), img)
        return
    elif unchanged:
        if len(img.shape) >= 3:
            img = swap_rb(img)
        cv2.imwrite(str(path), img)
    else:
        if gamma_correction:
            img = lin_rgb2srgb(img)

        if np.issubdtype(img.dtype, np.floating):
            img = real2uint(img, out_dtype)
        elif np.issubdtype(out_dtype, np.unsignedinteger):
            img = uint2uint(img, out_dtype)
        else:
            raise TypeError(
                f'Unsupported img.dtype={img.dtype}, dtype should be floating or unsignedinteger!')
        
        imsave(path, img)
