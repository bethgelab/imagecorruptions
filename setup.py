import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="imagecorruptions",
    version="1.1.2",
    author="Evgenia Rusak, Benjamin Mitzkus",
    author_email="evgenia.rusak@bethgelab.org, benjamin.mitzkus@bethgelab.org",
    description="This package provides a set of image corruptions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bethgelab/imagecorruptions",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy >= 1.16',
          'Pillow >= 5.4.1',
          'scikit-image >= 0.15',
          'opencv-python >= 3.4.5',
          'scipy >= 1.2.1',
      ],
      include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
