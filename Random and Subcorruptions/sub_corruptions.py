
from imagecorruptions import corrupt 
from imagecorruptions import get_corruption_names
from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 



def display_multiple_img(images, mrows = 1, mcols=1):
    figure, ax = plt.subplots(mrows, mcols) 
  
    for ind, rows in enumerate(images):
        for col,img in enumerate(rows):
            if ind == 0:
                ax[ind,col].set_title(col+1)
            imag = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            ax[ind,col].imshow(imag ,interpolation= "none")
            ax[ind,col].set_axis_off()
    plt.tight_layout()
    plt.show()
    figure.savefig("/home/hue/Desktop/plotcorr/pixelate.svg",dpi=1200)



directory = "/home/hue/Desktop/deneme"
os.chdir(directory)

images = []

for filename in os.listdir(directory):
    img = cv2.imread("%s/%s" % (directory,filename))
    
    rows = []

    for severity in range(10):       
        corrupted = corrupt(img, corruption_name="pixelate", severity= severity+1)
        outfile = "%d"%(severity+1)
        rows.append(corrupted)
    images.append(rows)           
total_images = len(images)
display_multiple_img(images, 10, 10 )
