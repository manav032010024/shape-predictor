"""
In this code we take the image and convert it to grayscale = following which facial detection is performed and 
then facial landmarks are obtained using facial landmark predictor dlib.shape_predictor from the dlib library

We are using the pre-trained facial landmark detector included in the dlib library. 
It estimates the 68 (x,y) coordinates that map to facial structures on the face.
(https://www.researchgate.net/profile/Pavel-Korshunov/publication/329392737/figure/fig2/AS:731613625335813@1551441689799/The-68-landmarks-detected-by-dlib-library-This-image-was-created-by-Brandon-Amos-of-CMU.ppm)

Face detection can be performed in multiple ways but most methods try to localize and label the following facial regions:
Nose | Jaws | Eyes| Eyebrows | Mouth


"""










# Necessary imports
import cv2
import dlib
import numpy as np
import os
import imutils


## set directories
os.chdir(r"C:\Users\91905\Desktop\DaDlib")
path = r"C:\Users\91905\Desktop\DaDlib\put"

#Initialize color [color_type] = (Blue, Green, Red). Colour space for OpenCV is BGR instead of RGB.
color_blue = (254,207,110)
###color_cyan = (255,200,0)
color_black = (0, 0, 0)

# Use input () function to capture from user requirements for mask type and mask colour
choice1 = input("Please select your choice of mask color\nEnter 1 for blue\nEnter 2 for black:\n")
choice1 = int(choice1)

if choice1 == 1:
    choice1 = color_blue
    print('You have selected a BLUE mask')
elif choice1 == 2:
    choice1 = color_black
    print('You selected a BLACK mask')
else:
    print("Invalid selection, please select again.")
    input("Please select your choice of mask color\nEnter 1 for blue\nEnter 2 for black :\n")


choice2 = input("Please enter your choice of mask coverage \nEnter 1 for high \nEnter 2 for medium \nEnter 3 for low :\n")
choice2 = int(choice2)

if choice2 == 1:
    # choice2 = fmask_a
    print(f'You have chosen an overlay of the wide, high coverage mask')
elif choice2 == 2:
    # choice2 = fmask_c
    print(f'You have chosen an overlay of the wide, medium coverage mask')
elif choice2 == 3:
    # choice2 = fmask_e
    print(f'You have chosen an over lay of the wide, low coverage mask')
else:
    print("Invalid selection, please select again.")
    input("Please enter your choice of mask coverage \nEnter 1 for high \nEnter 2 for medium \nEnter 3 for low :\n")

# print(choice2)



# Loading the image and resizing, converting it to grayscale
img= cv2.imread("C:\\Users\\91905\\Desktop\\DaDlib\\put.jpg")
img = imutils.resize(img, width = 500) #imutils maintains the aspect ratio and provides the keyword arguments width and height  
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts image from RGB hues to gray. Faces are detected in grayscale image. 

# Initialize dlib's face detector
"""
The frontal face detector in DLIB is based on Histogram of Oriented Gradients or HOG and linear SVM.
HOG is a feature descriptor used in CV and image processing for object detection.
The technique counts the occurences of gradient orientation in localized positions of an image. Gradient Orientation tells us rge direction of greatest intensity change in th eneighborhood of pixel (x,y)
SVM works by mapping data to a high dimensional feature space so that data points can be cateorized (in this case, think of it as face and no face)
"""
detector = dlib.get_frontal_face_detector() 

"""
Detecting faces in the grayscale image and creating an object - faces to store the list of bounding rectangles coordinates
The "1" in the second argument indicates that we should upscaled the image 1 time.  
This will make everything bigger and allow us to detect more faces
"""

faces = detector(gray, 1)

# printing the coordinates of the bounding rectangles
print(faces)
print("Number of faces detected: ", len(faces))

"""
# Using a for loop in order to extract the specific coordinates (x1,x2,y1,y2)
for face in faces:
  x1 = face.left()
  y1 = face.top()
  x2 = face.right()
  y2 = face.bottom()
  # Drawing a rectangle around the face detected
  cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0),3)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
Detecting facial landmarks using facial landmark predictor dlib.shape_predictor from dlib library
This shape prediction method requires the file called "shape_predictor_68_face_landmarks.dat" to be downloaded
Source of file: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
# Path of file
p = r"C:\Users\91905\Desktop\DaDlib\Shape predictor/shape_predictor_68_face_landmarks.dat"
# Initialize dlib's shape predictor
predictor = dlib.shape_predictor(p) #used to localize individual face structures including the eyes, eyebrows, nose, jawline, lips/mouth

"""
This method of localizing individual features has other uses. 
$ Computer vision based dox scanner
$ Detect structural joints in the human body
$ Face alignment, head pose estimation, drowsiness detector etc 
"""

# Get the shape using the predictor

for face in faces:
    landmarks = predictor(gray, face)

    """
    Two steps to detecting face landmarks in an image:
    $ Face Detection: returns value in x, y, w , h (a rectangle)
    $ Face Landmarks (go through points inside the rectangle)

    After the shape predictor is downloaded (shape_predictor_68_landmarks.dat). we can initialize the predictor for subsequent 
    use to detect facial landmarks on EVERY face detection in the input image.

    Once the face landmarks are detected we will be able to start "drawing"/overlaying the face masks on the faces by joining 
    the required points using the drawing functions of OpenCV.

    """

    # for n in range(0,68):
    #     x = landmarks.part(n).x
    #     y = landmarks.part(n).y
    #     img_landmark = cv2.circle(img, (x, y), 4, (0, 0, 255), -1)


    points = []
    for i in range(1, 16):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)
    # print(points)

    # Coordinates for the additional 3 points for wide, high coverage mask - in sequence
    mask_a = [((landmarks.part(42).x), (landmarks.part(15).y)),
              ((landmarks.part(27).x), (landmarks.part(27).y)),
              ((landmarks.part(39).x), (landmarks.part(1).y))]

    # Coordinates for the additional point for wide, medium coverage mask - in sequence
    mask_c = [((landmarks.part(29).x), (landmarks.part(29).y))]

    # Coordinates for the additional 5 points for wide, low coverage mask (lower nose points) - in sequence
    mask_e = [((landmarks.part(35).x), (landmarks.part(35).y)),
              ((landmarks.part(34).x), (landmarks.part(34).y)),
              ((landmarks.part(33).x), (landmarks.part(33).y)),
              ((landmarks.part(32).x), (landmarks.part(32).y)),
              ((landmarks.part(31).x), (landmarks.part(31).y))]

    fmask_a = points + mask_a
    fmask_c = points + mask_c
    fmask_e = points + mask_e

    # mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}
    # mask_type[choice2]


    # Using Python OpenCV – cv2.polylines() method to draw mask outline for [mask_type]:
    # fmask_a = wide, high coverage mask,
    # fmask_c = wide, medium coverage mask,
    # fmask_e  = wide, low coverage mask

    fmask_a = np.array(fmask_a, dtype=np.int32)
    fmask_c = np.array(fmask_c, dtype=np.int32)
    fmask_e = np.array(fmask_e, dtype=np.int32)

    mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}
    mask_type[choice2]


    # change parameter [mask_type] and color_type for various combination
    img2 = cv2.polylines(img, [mask_type[choice2]], True, choice1, thickness=2, lineType=cv2.LINE_8)

    """
    cv2.polylines() is used to draw a polygon.
    Syntax - cv.polylines (img, [], True, coordinates)
    If the third argument is false, you will get a polygon joining all the points and not a closed space.

    cv2.polylines() can be used to draw multiple lines. Just create a list of all lines you want to draw and pass it to the function 
    This is better than using cv.line() for each line.
    """

    # Using Python OpenCV – cv2.fillPoly() method to fill mask
    # change parameter [mask_type] and color_type for various combination
    img3 = cv2.fillPoly(img2, [mask_type[choice2]], choice1, lineType=cv2.LINE_AA)

# cv2.imshow("image with mask outline", img2)
cv2.imshow("image with mask", img3)

#Save the output file for testing
outputNameofImage = "output/imagetest.jpg"
print("Saving output image to", outputNameofImage)
cv2.imwrite(outputNameofImage, img3)


cv2.waitKey(0)
cv2.destroyAllWindows()