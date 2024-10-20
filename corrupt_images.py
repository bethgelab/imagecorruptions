import os
import glob
import argparse
import filetype
import matplotlib.pyplot as plt
import numpy as np

from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names
from imagecorruptions import corruption_dict
from tqdm import tqdm
from multiprocessing import Pool
from enum import Enum

# Example call:
# python corrupt_images.py test_images out_images filename -j 10 -c fog snow -su digital -se 1 2 -n 20
# corrupts all images in test_images and puts the results in out_images
# corruption type will be added to the filename
# corruption happens on 10 cores in parallel
# fog, snow and all digital corruptions are applied
# with severity level 1 and 2
# and on a total of 20 images


class OutputType(Enum):
    """How should the generated files be arranged
    """
    SUBDIRS = 'subdirs'
    FILENAME = 'filename'

    def __str__(self) -> str:
        return self.value


# https://github.com/scikit-image/scikit-image/issues/4294
def corrupt_image(image_path: str, image_path_base: str,
                  output_directory: str, output_type: OutputType,
                  corruptions: list, severity_levels: list) -> bool:
    """Apply image corruption to all images in a given folder

    Args:
        image_path (str): Path to an image
        input_path_base (str): Base path of input folder, needed to keep directory structure
        output_directory (str): Output folder
        output_type (OutputType): How should the files be arranged, in
            subfolders for each corruption and severity level or should 
            the corruption type be added to the filename
        corruptions (list): which corruptions should be applied
        severity_levels (list): List of severity levels

    Returns:
        bool: If succeeded or failed
    """
    kind = filetype.guess(image_path)
    if not kind.mime.startswith('image'):
        # Skip inputs that are not images...
        return False

    if kind.extension == 'png':
        # matplotlib reads png in float format -> convert to uint8
        img_array = plt.imread(image_path) * 255
        img_array = img_array.astype(dtype=np.uint8)
    else:
        # other image formats are already read as uint8
        img_array = plt.imread(image_path)

    output_path_stub = os.path.relpath(os.path.dirname(image_path), image_path_base)

    for corruption in corruptions:  # get_corruption_names(subset=subset):
        for severity in severity_levels:
            if output_type == OutputType.SUBDIRS:
                # Build output_path with subdirectories for each corruption type
                # and severity, e.g., $OUT_DIR/$ORIGINAL_STRUCTURE/snow/1/image.jpg
                output_path = os.path.join(output_directory, output_path_stub, corruption,
                                           str(severity), os.path.basename(image_path))

            elif output_type == OutputType.FILENAME:
                # Put corruption type and severity into filename, e.g., $OUT_DIR/$ORIGINAL_STRUCTURE/image_snow_1.jpg
                fname, ext = os.path.splitext(os.path.basename(image_path))
                fn = "{}_{}_{}{}".format(fname, corruption, str(severity), ext)
                output_path = os.path.join(output_directory, output_path_stub, fn)

            else:
                raise ValueError("output_type unsupported")

            out_dir = os.path.dirname(output_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Apply corruptions
            corrupted = corrupt(img_array, corruption_name=corruption, severity=severity)

            plt.imsave(output_path, corrupted)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("in_path", help="Directory which has to be processed")
    parser.add_argument("out_path", help="Output folder")
    parser.add_argument("output_type", choices=list(OutputType), type=OutputType,
                        help="How should the output be organized")
    parser.add_argument("-su", "--subset", choices=['common', 'validation', 'all', 'noise', 'blur',
                        'weather', 'digital'], help="Which subsets of corruptions should be applied")
    parser.add_argument("-c", "--corruptions", type=str, choices=corruption_dict.keys(), nargs="+",
                        help="Kind of corruptions to be applied, can be mixed with subset")
    parser.add_argument("-se", "--severity", type=int, choices=range(1, 5), nargs="*",
                        help="Severity level of corruption, if not provided all 5 levels will be applied")
    parser.add_argument("-j", type=int, default=1,
                        help="Multiprocessing, default is 1 core")
    parser.add_argument("-n", type=int, help="Limit the number of input images to be corrupted")

    opt = parser.parse_args()

    # make severity a list
    severity_levels = list(range(1, 6)) if opt.severity is None else opt.severity

    # Get the total number of images to be corrupted, mainly for progress bar
    total = opt.n if opt.n is not None else sum([len(files) for r, d, files in os.walk(opt.in_path)])

    corruptions = opt.corruptions if opt.corruptions else []
    if opt.subset:
        corruptions.extend(get_corruption_names(opt.subset))
    corruptions = list(set(corruptions))  # remove duplicates
    assert len(corruptions) > 1, ValueError("No corruption types were provided")

    # Spawn multiprocessing pool
    pool = Pool(opt.j)
    progress_bar = tqdm(total=total, ascii=True)

    def update_bar(*args):
        progress_bar.update()

    i = 0
    for filename in glob.glob(os.path.join(opt.in_path, "**"), recursive=True):
        i += 1
        # skip directories
        if os.path.isdir(filename):
            continue

        pool.apply_async(corrupt_image,
                         args=[filename, opt.in_path, opt.out_path, opt.output_type, opt.corruptions, severity_levels],
                         callback=update_bar)

        # break when n is reached
        if opt.n and i > opt.n:
            break

    pool.close()
    pool.join()
