from cv2 import cv2
import numpy as np
import random

display_size = (640,480)
number_of_points = 200 # Number of random points to generate

# Creating numpy array of random points
points = np.zeros((number_of_points,2), dtype=np.float32)
for i in range(0,number_of_points):
    if i < number_of_points/5:
        x = random.gauss((i+1)*16, 12)
        y = random.gauss((i+10)*2, 12)
    else:
        x = random.gauss((i-30)*3.5 - (number_of_points/4), 12)
        y = random.gauss((i+10)*2.5 - (number_of_points/4), 12)

    points[i,:] = [x, y]

# Initializing display with ones
display = np.ones((display_size[1],display_size[0],3), dtype=np.uint8)

# minimum inlier distance and iterations
eeta = 30
iterations = int(number_of_points/2)

# Initializing best params
max_inliers = 0
best_m = 0
best_b = 0

# Iterations begin
for i in range(iterations):
    # Selecting random samples (two)
    r1 = np.random.randint(0,points.shape[0])
    r2 = np.random.randint(0,points.shape[0]-1)
    if r2 == r1:
        r2 += 1

    point_1 = points[r1, :]
    point_2 = points[r2, :]

    # Calculating new values of m and b from random samples
    m = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    b = -m * point_1[0] + point_1[1]

    # Finding difference (perpendicular distance of point to line)
    diff = abs(((-m * points[:,0]) + (points[:,1] - b)) / ((-m)**2 + 1)**0.5)

    # Calculating inliers
    inliers = len(np.where(diff < eeta)[0])

    # Updating best params if better inliers found
    if inliers > max_inliers:
        max_inliers = inliers
        best_m = m
        best_b = b

    # Making canvas white
    display[:,:,:] = 255

    # Drawing random points
    for point in points:
        cv2.circle(display, (int(point[0]),int(point[1])), 3, (255, 0, 0), -1)
    
    # Drawing line
    p1 = (0, int(b))
    p2 = (display_size[0], int(m*display_size[0]+b))
    cv2.line(display, p1, p2, (100,255,0), 2)

    # Drawing best line
    p1 = (0, int(best_b))
    p2 = (display_size[0], int(best_m*display_size[0]+best_b))
    cv2.line(display, p1, p2, (0,0,255), 2)

    # Displaying
    cv2.imshow('Display', cv2.flip(display, 0))
    cv2.waitKey(20)


# Displaying final result

# Making canvas white
display[:,:,:] = 255

# Drawing random points
for point in points:
    cv2.circle(display, (int(point[0]),int(point[1])), 3, (255, 0, 0), -1)

# Drawing best line
p1 = (0, int(best_b))
p2 = (display_size[0], int(best_m*display_size[0]+best_b))
cv2.line(display, p1, p2, (0,0,255), 2)

# Displaying
cv2.imshow('Display', cv2.flip(display, 0))
cv2.waitKey(0)