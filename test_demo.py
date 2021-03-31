from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
image = np.asarray(Image.open('test_image.jpg'))
#image = np.ones((427, 640, 3), dtype=np.uint8)

# corrupted_image = corrupt(img, corruption_name='gaussian_blur', severity=1)

for corruption in get_corruption_names('blur'):
    tic = time.time()
    for severity in range(5):
        corrupted = corrupt(image, corruption_name=corruption, severity=severity+1)
        plt.imshow(corrupted)
        plt.show()
    print(corruption, time.time() - tic)