'''Trying to load the image and classify pixels'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#put our tiff into a variable

#load at school
#img = Image.open('/home/williamlavoy/model_traits/root_root_interaction_traits/2D_model_traits_work/images/test_classes.tiff')

#load at home
img = Image.open('/home/wlavoy/workDir/root_root_interaction_traits/2D_model_traits_work/images/test_classes.tiff')

#check image
#img.show()

#put tiff into np array
img_array = np.array(img)

#plot array in plt
data = plt.imshow(img_array)


#draw the plot
plt.show()

#determine how large the array is
#print(img_array.shape)
# 3456 by 5184

#empty variables for walking
i = 0
root = 0
background = 0

#start position for walker
startPos = np.array([0,i])


#want to walk from top to bottom one column at a time
for i in range(0,5184):
    print("column number: " + str(i))
    #steps and variable for walker
    j = 0
    step = [j,0]
    #while loop to iterate through columns
    while j < 3457:
        #print("row number: " + str(j))
        if img_array[startPos[0], startPos[1]] == 1:
            root += 1
            print("root count: " + str(root))
            j += 1
            startPos = startPos + step
        elif img_array[startPos[0], startPos[1]] == 2:
            background += 1
            #print("background count: " + str(background))
            j += 1
            startPos = startPos + step
    


