from imagecorruptions.all_corruptions import all_corruptions

all_corruptions(directory= "/home/hue/Desktop/animals/",
                targetdir= "/home/hue/Desktop/all_corruptions",
                corruption_mode="all_robust",
                save_type= "dir",
                severity= -1)
