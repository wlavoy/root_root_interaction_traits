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
#plt.show()

#determine how large the array is
#print(img_array.shape)
# 3456 by 5184

#empty variables for walking
j = 0
i = 0
root = 0
background = 0

#start position variable
startPos = []



#want to walk from top to bottom one column at a time
for i in range(0,5183):
    #start position updates in array
    startPos = np.array([j,i])
    i += 1
    #print("column number: " + str(i))
   
    #while loop to iterate through rows
    for j in range(0,3456):
        #start position updates in array
        startPos = np.array([j,i])
        
        #check to see if position contains a root and update root count
        if img_array[startPos[0], startPos[1]] == 1:
            root += 1
            #print("root count: " + str(root))
            j += 1
        
        #if position is not a root it must be background
        elif img_array[startPos[0], startPos[1]] != 1:
            background += 1
            #print("background count: " + str(background))
            j += 1
           
    #reset j value after while loop so we start from the top position and work our way down
    j = 0

print("Total Root Count: " + str(root))
print("Total Background Count: " +str(background))

density = (root/background)
print("Density of Roots/Background: ")
print(density)





