'''
A root density measurement based on pixel classification from ilastik
The image is divided into thirds
Each zone can be described as interactive or non interactive zones
Roots in between the two plants "zone 2" are considering interactive
Roots growing away from the other plant "zones 1 and 3" are considered non interactive
'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#put our tiff into a variable

#load at school
#img = Image.open('/home/williamlavoy/model_traits/root_root_interaction_traits/2D_model_traits_work/images/test_classes.tiff')

#load at home
img = Image.open('/home/wlavoy/workDir/root_root_interaction_traits/2D_model_traits_work/images/test_classes.tiff')

#put tiff into np array
img_array = np.array(img)

#plot array in plt
data = plt.imshow(img_array)


#empty variables for walking
j = 0
i = 0
root = 0
root1 = 0
root2 = 0
root3 = 0
background = 0
background1 = 0
background2 = 0
background3 = 0

#start position variable
startPos = []

#want to walk from top to bottom in zone1
for i in range(0,1728):
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
            root1 += 1
            #print("root count: " + str(root))
            j += 1
        
        #if position is not a root it must be background
        elif img_array[startPos[0], startPos[1]] == 2:
            background += 1
            background1 += 1
            #print("background count: " + str(background))
            j += 1
           
    #reset j value after while loop so we start from the top position and work our way down
    j = 0

#want to walk from top to bottom in zone2
for i in range(1728,3456):
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
            root2 += 1
            #print("root count: " + str(root))
            j += 1
        
        #if position is not a root it must be background
        elif img_array[startPos[0], startPos[1]] == 2:
            background += 1
            background2 += 1
            #print("background count: " + str(background))
            j += 1
           
    #reset j value after while loop so we start from the top position and work our way down
    j = 0

#want to walk from top to bottom in zone1
for i in range(3456,5183):
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
            root3 += 1
            #print("root count: " + str(root))
            j += 1
        
        #if position is not a root it must be background
        elif img_array[startPos[0], startPos[1]] == 2:
            background += 1
            background3 += 1
            #print("background count: " + str(background))
            j += 1
           
    #reset j value after while loop so we start from the top position and work our way down
    j = 0

#check total counts
print("Total Root Count: " + str(root))
print("Total Background Count: " +str(background))

#density calculation of root points/total points
density = (root/(background+root))
print("Density of All Roots/Background: ")
print(density)

#zone 1 calculations
print("Zone 1 Root Count: " + str(root1))
print("Zone 1 Background Count: " +str(background1))

density1 = (root1/(background1+root1))
print("Density Zone 1 Roots/Background: ")
print(density1)

#zone 2 calculations
print("Zone 2 Root Count: " + str(root2))
print("Zone 2 Background Count: " +str(background2))

density2 = (root2/(background2+root2))
print("Density Zone 2 Roots/Background: ")
print(density2)

#zone 3 calculations
print("Zone 3 Root Count: " + str(root3))
print("Zone 3 Background Count: " +str(background3))

density3 = (root3/(background3+root3))
print("Density Zone 3 Roots/Background: ")
print(density3)

#non interactive calculations (zone1+3)
rootn = root1 + root3
print("Non-Interactive Root Count: " + str(rootn))

densityn = (rootn/(background+root))
print("Density of Non-Interactive Roots/Background: ")
print(densityn)

#interactive calculations (zone2)
print("Interactive Root Count: " + str(root2))

densityi = (root2/(background+root))
print("Density of Interactive Roots/Background: ")
print(densityi)

