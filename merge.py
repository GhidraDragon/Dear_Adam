#!/usr/bin/env python3

import sys
import os
import uuid
import datetime
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.utils import check_random_state

def _avg_brightness(img):
    return np.array(img.convert("L")).mean()

def _adjust_brightness(img, target=128):
    mean_val = _avg_brightness(img)
    if mean_val < 80 or mean_val > 180:
        factor = target / (mean_val + 1e-9)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
    return img

def _auto_enhance(img):
    avg_val = _avg_brightness(img)
    c_factor = 1.2 if avg_val < 130 else 1.1
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    img = ImageEnhance.Color(img).enhance(c_factor)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    return img

def create_intelligent_output_filename(image_paths, prefix="merged_"):
    base_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    if len(base_names) > 3:
        base_names = base_names[:3] + ["etc"]
    base_str = "_".join(base_names)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = str(uuid.uuid4())[:8]
    return f"{prefix}{base_str}_{now_str}_{random_str}"

def parse_resize(resize_str):
    w, h = resize_str.lower().split('x')
    return int(w), int(h)

def parse_crop(crop_str):
    size_part, x_off, y_off = crop_str.lower().split('+')
    w, h = size_part.split('x')
    return (int(x_off), int(y_off), int(w), int(h))

def merge_images(image_paths, resize_dims, rng, crop_dims):
    accum_array = None
    count = 0
    for path in image_paths:
        if not os.path.isfile(path):
            sys.exit(f"Input file '{path}' not found.")
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as e:
            sys.exit(f"Error opening image '{path}': {e}")
        img = _adjust_brightness(img)
        if resize_dims:
            w, h = resize_dims
        else:
            w, h = (1024, 1024)
        if hasattr(Image, 'Resampling'):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        else:
            img = img.resize((w, h), Image.ANTIALIAS)
        arr = np.array(img, dtype=np.uint8)
        if accum_array is None:
            accum_array = arr.astype(np.int32)
        else:
            accum_array += arr.astype(np.int32)
        count += 1
    avg_array = (accum_array / count).astype(np.uint8)
    noise = rng.normal(loc=0, scale=10, size=avg_array.shape)
    noisy_array = avg_array.astype(float) + noise
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    final_image = Image.fromarray(noisy_array, mode="RGBA")
    final_image = _auto_enhance(final_image)
    if crop_dims:
        x, y, cw, ch = crop_dims
        final_image = final_image.crop((x, y, x + cw, y + ch))
    return final_image

def apply_sepia(img):
    img = img.convert("RGB")
    arr = np.array(img)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    tr = 0.393*r + 0.769*g + 0.189*b
    tg = 0.349*r + 0.686*g + 0.168*b
    tb = 0.272*r + 0.534*g + 0.131*b
    arr[:,:,0] = np.clip(tr, 0, 255)
    arr[:,:,1] = np.clip(tg, 0, 255)
    arr[:,:,2] = np.clip(tb, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), 'RGB')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--resize", help="Resize final images, e.g. 800x600")
    parser.add_argument("--crop", help="Crop final image, e.g. 400x400+10+10")
    parser.add_argument("--rotate", type=int, help="Rotate final image by degrees")
    parser.add_argument("--flip", choices=["horizontal", "vertical"], help="Flip final image")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality")
    parser.add_argument("--format", choices=["PNG","JPEG"], default="PNG", help="Output format")
    parser.add_argument("--brightness", type=float, help="Adjust brightness factor")
    parser.add_argument("--contrast", type=float, help="Adjust contrast factor")
    parser.add_argument("--color", type=float, help="Adjust color factor")
    parser.add_argument("--sharpness", type=float, help="Adjust sharpness factor")
    parser.add_argument("--grayscale", action='store_true', help="Convert final image to grayscale")
    parser.add_argument("--blur", type=float, help="Apply blur radius")
    parser.add_argument("--sepia", action='store_true', help="Apply sepia filter")
    parser.add_argument("--auto_enhance", action='store_true', help="Apply auto enhance")
    parser.add_argument("--ai_edit", action='store_true', help="Apply advanced AI editing (placeholder)")
    args = parser.parse_args()

    if len(args.images) < 2:
        print("Usage: script.py <image1> <image2> [image3 ...] [options]")
        sys.exit(1)

    output_file_base = create_intelligent_output_filename(args.images)
    ext = ".png" if args.format == "PNG" else ".jpg"
    output_file = output_file_base + ext
    rng = check_random_state(None)
    resize_dims = parse_resize(args.resize) if args.resize else None
    crop_dims = parse_crop(args.crop) if args.crop else None

    final_img = merge_images(args.images, resize_dims, rng, crop_dims)

    if args.rotate:
        final_img = final_img.rotate(args.rotate, expand=True)
    if args.flip == "horizontal":
        final_img = final_img.transpose(Image.FLIP_LEFT_RIGHT)
    elif args.flip == "vertical":
        final_img = final_img.transpose(Image.FLIP_TOP_BOTTOM)
    if args.brightness:
        enhancer = ImageEnhance.Brightness(final_img)
        final_img = enhancer.enhance(args.brightness)
    if args.contrast:
        enhancer = ImageEnhance.Contrast(final_img)
        final_img = enhancer.enhance(args.contrast)
    if args.color:
        enhancer = ImageEnhance.Color(final_img)
        final_img = enhancer.enhance(args.color)
    if args.sharpness:
        enhancer = ImageEnhance.Sharpness(final_img)
        final_img = enhancer.enhance(args.sharpness)
    if args.grayscale:
        final_img = final_img.convert("L").convert("RGBA")
    if args.blur:
        final_img = final_img.filter(ImageFilter.GaussianBlur(args.blur))
    if args.sepia:
        tmp_mode = final_img.mode
        final_img = apply_sepia(final_img)
        if "A" in tmp_mode:
            final_img = final_img.convert("RGBA")
    if args.auto_enhance:
        final_img = _auto_enhance(final_img)
    if args.ai_edit:
        final_img = _auto_enhance(final_img)

    save_params = {}
    if args.format == "JPEG":
        save_params["quality"] = args.quality

    final_img.save(output_file, args.format, **save_params)
    print(f"Output image saved as '{output_file}'")

if __name__ == "__main__":
    main()