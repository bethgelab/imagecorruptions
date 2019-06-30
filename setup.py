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
