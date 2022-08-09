import cv2
from imagecorruptions import corrupt 
from imagecorruptions import get_corruption_names
import random as random
import os

def random_corruptions(directory,targetdir,corrcopy,last_corruption_name = "original"):
    """
    The purpose of this module :
            is to generate copies of the labeled dataset in the given directory, consisting of randomly corrupted elements, in the targetdir.
            Corruptions and severity values are randomly selected for each image.
   
    Args:
     directory (str): directory of the labeled dataset to be corrupted.
     
     targetdir (str): directory to save corrupted copies
     
     corrcopy: the number of corrupted datasets to be created.

     last_corruption_name: Option to save corrupted images with which name. 
                            If "corrupted" is selected, it will be saved as "(corruptionName_severity)originalName". 
                            If "original is selected, it will be saved with the original name.

    Returns:
        corrupted copy of the labeled dataset.
    """     
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    if not os.path.exists(directory):
        raise ValueError("Directory does not exist")
    if not os.path.isdir(directory):
        raise ValueError("Directory is not a directory")
    if not os.path.isdir(targetdir):
        raise ValueError("Target directory is not a directory")
    if not (corrcopy >= 0):
        raise ValueError("Invalid number of corruptions to be copied")
    if not (last_corruption_name == "original" or last_corruption_name == "corrupted"):
        raise ValueError("Invalid last corruption name")
    
    os.chdir(directory)

    for filename in os.listdir(directory):
        for labels in range(1,corrcopy+1):

            try:
                os.mkdir("%s/corrupted_%d" % (targetdir,labels))
                os.mkdir("%s/corrupted_%d/%s"%(targetdir,labels,filename))
            except:
                os.listdir("%s/corrupted_%d" % (targetdir,labels))
                os.mkdir("%s/corrupted_%d/%s"%(targetdir,labels,filename))

    for filename in os.listdir(directory):
        for  imgname in os.listdir(filename):
            img = cv2.imread("%s/%s/%s" % (directory, filename, imgname))

            for x in range(1,corrcopy+1):
                os.chdir("%s/corrupted_%d/%s"%(targetdir,x,filename))

                for corruption in get_corruption_names(subset ="random_robust"):
                    severity = random.randint(1,10)
                    
                    if last_corruption_name ==  "original":
                        outfile = "%s"%(imgname)
                    elif last_corruption_name == "corrupted":
                        outfile = "(%s_%d)%s"%(corruption,severity,imgname)

                    try:
                        corrupted = corrupt(img, corruption_name=corruption, severity=severity)
                        cv2.imwrite(outfile,corrupted)
                        print("%s saved as %s"%(corruption,outfile))
                    except:
                        imgResized = cv2.resize(img, (330,230))
                        corrupted = corrupt(imgResized, corruption_name=corruption, severity=severity)
                        cv2.imwrite(outfile,corrupted)
                        print("%s saved as %s"%(corruption,outfile))
                        
        os.chdir(directory) 

                