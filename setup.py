import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="corruptions",
    version="0.0.1",
    author="Benjamin Mitzkus",
    author_email="benjamin.mitzkus@bethgelab.org",
    description="This package provides a set of image corruptions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bethgelab/add-common-image-corruptions",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy',
          'Pillow',
          'scikit-image',
          'wand',
          'opencv-python',
          'scipy',
      ],
      include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import os
from pkg_resources import resource_filename
