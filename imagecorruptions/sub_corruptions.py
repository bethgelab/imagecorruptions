from imagecorruptions import corrupt 
from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 


def corrupt_scale(directory,targetdir,corruption_name,severity_size):
    """
    The purpose of this function is to create a subplot by applying corruptions at the selected severity levels to the pictures in the folder specified in the directory. 
    Then this subplot will be saved as ".svg" to the location we specified in targetdir. 
    In this way, it creates an opportunity to examine the effects of severity values in corruptions on our pictures. 
    In addition, these subplots will make it easier to observe the effects of changes made in corruptions.py and to reach the c values you need.
    
    Args:
        directory (str):  directory of the images to be corrupted

        targetdir (str): directory to save .svg files

        corruption_name (str): name of the corruption to be applied to the images in the directory.
                               specifies which corruption function to call, must be one of
                                           (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                                            glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                                            brightness, contrast, elastic_transform, pixelate,
                                            jpeg_compression, speckle_noise, gaussian_blur, spatter,
                                            saturate, blackout, postcontrast)
        

        severity_size (int): number of severity, number of columns to be displayed in subplot, integer in [1,10]
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    if not os.path.exists(directory):
        raise ValueError("Directory does not exist")
    if not os.path.isdir(directory):
        raise ValueError("Directory is not a directory")
    if not os.path.isdir(targetdir):
        raise ValueError("Target directory is not a directory")
    if not (severity_size >= 0):
        raise ValueError("Invalid severity size")
    if not (corruption_name == "gaussian_noise" or corruption_name == "shot_noise" or 
            corruption_name == "impulse_noise" or corruption_name == "defocus_blur" or 
            corruption_name == "glass_blur" or corruption_name == "motion_blur" or 
            corruption_name == "zoom_blur" or corruption_name == "snow" or 
            corruption_name == "frost" or corruption_name == "fog" or 
            corruption_name == "brightness" or corruption_name == "contrast" or 
            corruption_name == "elastic_transform" or corruption_name == "pixelate" or 
            corruption_name == "jpeg_compression" or corruption_name == "speckle_noise" or 
            corruption_name == "gaussian_blur" or corruption_name == "spatter" or 
            corruption_name == "saturate" or corruption_name == "blackout" or 
            corruption_name == "postcontrast"):
        raise ValueError("Invalid corruption name") 
        
    def display_multiple_img(images, mrows = 1, mcols=1):
        figure, ax = plt.subplots(mrows, mcols) 
    
        for ind, rows in enumerate(images):
            for col,img in enumerate(rows):
                if ind == 0:
                    ax[ind,col].set_title(col+1)
                imag = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ax[ind,col].imshow(imag ,interpolation= "none")
                ax[ind,col].set_axis_off()

        plt.suptitle("%s"%corruption_name, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        figure.savefig(("%s/%s.svg"%(targetdir,corruption_name)),dpi=1200)


    os.chdir(directory)

    images = []

    image_number = 0

    for filename in os.listdir(directory):
        img = cv2.imread("%s/%s" % (directory,filename))
        image_number = image_number + 1
        rows = []

        for severity in range(severity_size):       
            corrupted = corrupt(img, corruption_name=corruption_name, severity= severity+1)
            rows.append(corrupted)
        
        images.append(rows)

    display_multiple_img(images, image_number, severity_size )
