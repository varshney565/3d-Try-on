import cv2
import dlib
import numpy as np
import math

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Load the facial landmark predictor model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the image of glasses
glasses_img = cv2.imread('images/glasses1.png', -1)

def calculate_angle(pt1, pt2):
    # Calculate the angle between two points
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    return math.degrees(math.atan2(y_diff, x_diff))

def overlay_glasses(face_img, glasses_img, x, y, w, h, angle):
    # Calculate the size of the glasses based on the width and height of the detected face
    glasses_width = int(w * 0.8)
    glasses_height = int(h * 0.5)
    
    # Resize the glasses to fit the face
    resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))

    # Rotate the glasses image according to the face angle
    center = (glasses_width // 2, glasses_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_glasses = cv2.warpAffine(resized_glasses, rotation_matrix, (glasses_width, glasses_height))

    # Extract the alpha channel from the glasses image
    alpha_mask = rotated_glasses[:, :, 3]

    # Create an inverted mask for the glasses
    inverted_mask = cv2.bitwise_not(alpha_mask)

    # Calculate the position to overlay the glasses
    x_offset = int(x - (glasses_width - w) / 2)
    y_offset = int(y + h * 0.19)

    # Overlay the glasses on the face region
    roi = face_img[y_offset:y_offset + glasses_height, x_offset:x_offset + glasses_width]
    background = cv2.bitwise_and(roi, roi, mask=inverted_mask)
    overlay = cv2.bitwise_and(rotated_glasses[:, :, :3], rotated_glasses[:, :, :3], mask=alpha_mask)
    face_img[y_offset:y_offset + glasses_height, x_offset:x_offset + glasses_width] = cv2.add(background, overlay)

# Load the input image
image = cv2.imread('images/temp.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Overlay glasses on each detected face
print(faces)
for (x, y, w, h) in faces:
    # Estimate the angle of rotation for the face
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = predictor(gray, rect)
    
    # Calculate the angle of rotation using eye landmarks
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    angle = -calculate_angle(left_eye, right_eye)
    # Overlay glasses on the face
    overlay_glasses(image, glasses_img, x, y, w, h, angle)

# Display the result
cv2.imshow('Face with Glasses', image)
output_path = 'images/face_with_glasses.jpg'
cv2.imwrite(output_path, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
