'''Trying to load the image and classify pixels'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#put our tiff into a variable
img = Image.open('/home/williamlavoy/model_traits/root_root_interaction_traits/2D_model_traits_work/images/test_classes.tiff')

#check image
#img.show()

#put tiff into np array
img_array = np.array(img)

#plot array in plt
data = plt.imshow(img_array)

img_array.shape
plt.show()