import cv2
import numpy as np
import imutils
import sys
from collections import defaultdict
#source - https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines): segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    to_return = (x0, y0)
    return to_return
'''for x0,y0 in [x0, y0]:
        for x1, y1 in [x0, y0]:
            if x0!=x1 and y0!=y1:pass'''
#https://stackoverflow.com/questions/19375675/simple-way-of-fusing-a-few-close-points
def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append([int(point[0]), int(point[1])])
    return ret

def take_avg_x_val(points):
    total = 0
    for point in points:
        total += point[0]
    return int(total / len(points))

def standardize(points):
    categories = []
    categories.append(points[0])
    for i in range(1,len(points)):
        if abs(points[i][0] - take_avg_x_val(categories)) > 8:
            new_x = take_avg_x_val(categories)
            for point in points[i-len(categories):i]:
                point[0] = new_x
            #print(points[i-len(categories):i])
            categories.clear()
        categories.append(points[i])

def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group: intersections.append(intersection(line1, line2))
    points = fuse(intersections, 130)
    return points

file = sys.argv[1]
img = cv2.imread(file)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,15)
edges = cv2.Canny(gray,80, 160,apertureSize = 3, L2gradient = True)#2nd and 3rd params are low and high thresholds

new_edges = imutils.resize(edges,width=500)

new_img = imutils.resize(img,width=500)
cv2.imshow('image',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('edge',new_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLines(edges,1,np.pi/180,240)#last param is hough threshold #160 is base
print(len(lines))
segmented = segment_by_angle_kmeans(lines)
intersections = segmented_intersections(segmented)
intersections = sorted(intersections)
#standardize(intersections)
#intersections = sorted(intersections)
print(intersections)
print('num lines: ', len(lines))
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        #cv2.line(img,(x1,y1),(x2,y2),(0,255,0),10)
print("num of corners: ",len(intersections))
counter = 0
'''for corner in intersections[:]:
    cv2.circle(img,(corner[0],corner[1]),21,(0,0,255),12)
    counter += 1'''
'''print('\n\n\n\n')'''
'''cv2.circle(img,(intersections[0][0],intersections[0][1]),21,(0,0,255),12)
cv2.circle(img,(intersections[80][0],intersections[80][1]),21,(0,0,255),12)
cv2.circle(img,(intersections[69][0],intersections[69][1]),21,(0,0,255),12)
cv2.circle(img,(intersections[10][0],intersections[10][1]),21,(0,0,255),12)
print(intersections[11])
print(intersections[68])'''
'''
print('\n\n\n\n')
mod = 20
intersections[10][0] -= mod # top left
intersections[10][1] -= mod
intersections[68][0] += mod # top right
intersections[68][1] -= mod
intersections[80][0] += mod # bottom right
intersections[80][1] += mod
intersections[0][0] -= mod  # bottom left
intersections[0][1] += mod
print(intersections[10])
print(intersections[68])'''

output = imutils.resize(img,width=500)
cv2.imshow('hough',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

far_corners = [intersections[10], intersections[69], intersections[80], intersections[0]]
far_corners = np.array(far_corners, dtype = "float32")
print(far_corners)
warped = four_point_transform(img, far_corners)
out = imutils.resize(warped,width=1000)
cv2.imshow('perspective',out)
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = cv2.medianBlur(warped,15)
edges = cv2.Canny(warped,80, 160,apertureSize = 3, L2gradient = True)
new_edges = imutils.resize(edges,width=500)
cv2.imshow('edge',new_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLines(edges,1,np.pi/180,240)#last param is hough threshold #160 is base
print(len(lines))
segmented = segment_by_angle_kmeans(lines)
intersections = segmented_intersections(segmented)
intersections = sorted(intersections)
standardize(intersections)
intersections = sorted(intersections)
print("intersections: ",len(intersections))
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        #cv2.line(warped,(x1,y1),(x2,y2),(0,255,0),20)
#for corner in intersections:
   # cv2.circle(warped,(corner[0],corner[1]),21,(0,0,255),12)
output = imutils.resize(warped,width=500)
cv2.imshow('hough',output)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(71):
    print(i,': ' ,intersections[i])
    if (i + 1) % 9 != 0: # x ([0]) is vertical, y ([1]) is horizontal
        if i % 9 == 0:
            crop_img = warped[intersections[i][1]:intersections[10 + i][1], intersections[i][0]:intersections[i + 10][0]]
            cv2.imwrite(f'hough2/houghlines{i}.jpg',crop_img)
        else:
            crop_img = warped[intersections[i][1]-170:intersections[10 + i][1], intersections[i][0]:intersections[i + 10][0]]
            cv2.imwrite(f'hough2/houghlines{i}.jpg',crop_img)
#crop_img = img[0:500,0:500]
#cv2.imwrite('houghlines.jpg',crop_img)
#print(intersections)
#print(intersections[0][0]," " ,intersections[10][0]," " ,intersections[0][1]," " ,intersections[10][1])
crop_img =  img[0:300, 0:300]
#cv2.imshow('hough',crop_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imshow('hough',output)
cv2.imwrite('final_out.jpg',output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''

new_edges = imutils.resize(edges,width=500)
cv2.imshow('edge',new_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
