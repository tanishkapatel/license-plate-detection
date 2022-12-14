# import required libraries
import cv2
import imutils
import numpy as np
import pytesseract

# executing license plate detection on 8 different vehicles
for i in range(1,9):

	name = 'vehicle' + str(i) + '.jpeg'

	# read the image
	img = cv2.imread(name)

	# percent of original size
	scale_percent = 60
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)

	# resize the image
	img = cv2.resize(img,dim)

	# display the image
	cv2.imshow('Input Image',img)
	cv2.waitKey(0)

	# convert colour image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow('Grayscale Image',gray)
	cv2.waitKey(0)

	# remove unwanted noise using gaussian blur
	# here, kernel of 3x5 size is used
	gray = cv2.GaussianBlur(gray, (3, 5), 0)
	cv2.imshow('Grayscale Blurred Image',gray)
	cv2.waitKey(0)

	# perform canny edge detection
	# minimum threshold value = 30
	# maximum threshold value = 200
	edged = cv2.Canny(gray, 30, 220)
	cv2.imshow('Canny Edge detection',edged)
	cv2.waitKey(0)

	# find contours in the image
	# Contours are defined as the line joining
	# all the points along the boundary of an image that are having the same intensity.
	contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)

	# Once the counters have been detected we sort them from big to
	# small and consider only the first 10 results ignoring the others
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

	# To filter the license plate image among the obtained results, we will loop
	# though all the results and check which has a rectangle shape contour
	# with four sides and closed figure.

	screenCnt = None

	for c in contours:

		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.018 * peri, True)

		if len(approx) == 4:
			screenCnt = approx

		break

	# Once we have found the right counter we save it in a variable called
	# screenCnt and then draw a rectangle box around it to make sure we have
	# detected the license plate correctly.

	if screenCnt is None:
		detected = 0
		print ("No contour detected")

	else:
		detected = 1

	if detected == 1:
		cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

	# Now that we know where the number plate is, we can proceed with masking
	# the entire picture except for the place where the number plate is.
	# bitwise and function is used to mask the image

	mask = np.zeros(gray.shape,np.uint8)
	new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
	new_image = cv2.bitwise_and(img,img,mask=mask)

	# display the masked image
	cv2.imshow('Masked Image',new_image)
	cv2.waitKey(0)

	# segment the license plate out of the image by cropping it and saving it as a new image
	(x, y) = np.where(mask == 255)
	(topx, topy) = (np.min(x), np.min(y))
	(bottomx, bottomy) = (np.max(x), np.max(y))
	Cropped = gray[topx:bottomx+1, topy:bottomy+1]

	# read the number plate information from the segmented image
	# config = '---psm 7' ---> Treat the image as a single text line.

	text = pytesseract.image_to_string(Cropped, config='--psm 7')
	print("Vehicle number ",i)
	print("Detected license plate Number is:",text)

	# display final result
	cv2.imshow('car',img)
	cv2.waitKey(0)
	cv2.imshow('Cropped',Cropped)
	cv2.waitKey(0)
	cv2.destroyAllWindows()