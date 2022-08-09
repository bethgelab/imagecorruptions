import cv2
from imagecorruptions import corrupt 
from imagecorruptions import get_corruption_names
import random as random
import os

def all_corruptions(directory,targetdir,corruption_mode = "all_robust",save_type ="direct", severity = -1):
   
    """"
    The purpose of this module :
         is to produce corrupted copies of each image in a labeled data set, 
         according to the selected corruption mode, at the selected severity value.

    Args: 
        directory: directory of the images to be corrupted
        
        targetdir: directory to save corrupted images
        
        corruption_mode: "all_corr" for all corruptions, all_robust for corruptions selected for robustness, 
                    "noise" for noise corruptions, "blur" for blur corruptions, 
                    "weather" for weather corruptions,"digital" for digital corruptions,
                    "random_corr" for random corruptions,
                    "random_robust" for random corruptions selected for robustness.
        
        save_type:  select how to save corrupted images.
                For example, the cat inside the folder labeled cats.  
                if save_type = "direct" is selected, 
                    the corrupted image  is saved in the same place as the corruptions of the other images in the  cat folder. 
                if save_type = "dir" is selected, 
                    it creates a folder for each cat image in my cat folder and saves different corrupted versions of the cat image in this folder.

    severity: -1 for random severity, 1,2,3,4,5,6,7,8,9,10 for specific severity
    """     

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    if not os.path.exists(directory):
        raise ValueError("Directory does not exist")
    
    if not os.path.isdir(directory):
        raise ValueError("Directory is not a directory")
    
    if not os.path.isdir(targetdir):
        raise ValueError("Target directory is not a directory")
    
    if not corruption_mode in ["all_corr","all_robust","noise","blur","weather","digital","random_corr","random_robust"]:
        raise ValueError("Invalid corruption mode")
    
    if not (save_type == "direct" or save_type == "dir"):
        raise ValueError("Invalid save type")
    
    if not severity in [-1,1,2,3,4,5,6,7,8,9,10]:
        raise ValueError("Invalid severity")
        
  
    
    os.chdir(directory)
    sevrnd = random.randint(1,10)
    
    for filename in os.listdir(directory):
        os.mkdir("%s/%s"%(targetdir,filename))  

    for filename in os.listdir(directory):
        for  img_name in os.listdir(filename):

            if save_type == "dir":
                try:
                    os.mkdir("%s/%s/%s"%(targetdir,filename,img_name))
                except:
                    os.chdir("%s/%s"%(targetdir,filename))
                    os.mkdir("%s/%s/%s"%(targetdir,filename,img_name))
                img = cv2.imread("%s/%s/%s" % (directory, filename, img_name))
                os.chdir("%s/%s/%s"%(targetdir,filename,img_name))
            
            else:
                img = cv2.imread("%s/%s/%s" % (directory, filename, img_name))
                os.chdir("%s/%s"%(targetdir,filename))
            
            for corruption in get_corruption_names(subset ="%s"%corruption_mode):
                    
                    if severity == -1 or severity == sevrnd : 
                        sevrnd = random.randint(1,10)
                        severity = sevrnd
                    
                    try:
                        corrupted = corrupt(img, corruption_name=corruption, severity=severity)
                        outfile = "(%s_%s)%s"%(corruption, severity,img_name)
                        cv2.imwrite(outfile,corrupted)
                        print("%s saved as %s"%(corruption,outfile))
                    except:
                        imgResized = cv2.resize(img, (330,230))
                        corrupted = corrupt(imgResized, corruption_name=corruption, severity=severity)
                        outfile = "(%s_%s)%s"%(corruption, severity,img_name)
                        cv2.imwrite(outfile,corrupted)
                        print("%s saved as %s"%(corruption,outfile))
            
            os.chdir(directory)
                    
    print("All corruptions saved in %s"%targetdir)