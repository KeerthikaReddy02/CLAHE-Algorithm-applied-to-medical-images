import cv2
import numpy as np
from brisque import BRISQUE

# Load two images
img1 = cv2.imread('clahe1.jpg')
img2 = cv2.imread('clahe2.jpg')

# Calculate BRISQUE scores
brisque = BRISQUE()
brisque_score1 = brisque.score(img1)
brisque_score2 = brisque.score(img2)

# Calculate the artifact level metric for artifacts
blur1 = cv2.GaussianBlur(img1, (5, 5), 0)
blur2 = cv2.GaussianBlur(img2, (5, 5), 0)
diff1 = np.abs(img1 - blur1)
diff2 = np.abs(img2 - blur2)
al1 = np.sum(diff1) / np.sum(img1)
al2 = np.sum(diff2) / np.sum(img2)

# Print the scores
print('BRISQUE score of image 1:', brisque_score1)
print('BRISQUE score of image 2:', brisque_score2)
print('Artifact Level of image 1:', al1)
print('Artifact Level of image 2:', al2)

    
if brisque_score1 < brisque_score2:
    print('Image 1 has better quality than image 2 based on BRISQUE.')
else:
    print('Image 2 has better quality than image 1 based on BRISQUE.')

if al1 < al2:
    print('Image 1 has better quality than image 2 based on artifact level.')
else:
    print('Image 2 has better quality than image 1 based on artifact level.')


