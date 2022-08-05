import cv2
from imagecorruptions import corrupt 
from imagecorruptions import get_corruption_names
import random as random
import os


directory = "/home/hue/Desktop/test"
os.chdir(directory)

for filename in os.listdir(directory):
    for labels in range(1,6):
        try:
            os.mkdir("/home/hue/Desktop/corruptions_test/test_corruption%d"%labels)
            os.mkdir("/home/hue/Desktop/corruptions_test/test_corruption%d/%s"%(labels,filename))
        except:
            os.listdir("/home/hue/Desktop/corruptions_test/test_corruption%d"%labels)
            os.mkdir("/home/hue/Desktop/corruptions_test/test_corruption%d/%s"%(labels,filename))

for filename in os.listdir(directory):
    for  i in os.listdir(filename):
        img = cv2.imread("%s/%s/%s" % (directory, filename, i))
        for x in range(1,6):
             
            os.chdir("/home/hue/Desktop/corruptions_test/test_corruption%d/%s"%(x,filename))
            for corruption in get_corruption_names(subset ="random"):
                sev_aralik = random.randint(1,10)
                try:
                    corrupted = corrupt(img, corruption_name=corruption, severity=sev_aralik)
                    outfile = "%s"%(i)
                    cv2.imwrite(outfile,corrupted)
                    print("%s saved as %s"%(corruption,outfile))
                except:
                    imgResized = cv2.resize(img, (330,230))
                    corrupted = corrupt(imgResized, corruption_name=corruption, severity=sev_aralik)
                    outfile = "%s"%(i)
                    cv2.imwrite(outfile,corrupted)
                    print("%s saved as %s"%(corruption,outfile))
                    
    os.chdir(directory) 
            