import os.path
import cv2
import glob
import imutils
import numpy as np

CAPTCHA_IMAGE_FOLDER = "../Gujarat_Rural_Captchas"
OUTPUT_FOLDER = "../extracted_letter_images"

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # These 3 images are skipped(couldn't detect contours: 979875.png, 135175.png, 115716.png)
    if i == 1330 or i == 1624 or i == 1766:
        print(captcha_correct_text)
        continue

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur and threshold
    blur = cv2.blur(gray, (1, 1))
    ret, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    dil = cv2.dilate(thresh, np.ones((1, 1), np.uint8))
    ero = cv2.erode(dil, np.ones((2, 1), np.uint8))

    contours = cv2.findContours(ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv2() else contours[0]
    contours = [cnt for cnt in contours if (cv2.contourArea(cnt) < 10000) and (cv2.contourArea(cnt) > 0)]
    contours = sorted(contours, key=lambda elem: (cv2.arcLength(elem, True), cv2.contourArea(elem)), reverse=True)[:6]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 6 letters in the captcha, our letter extraction
    # didn't work correctly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 6:
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(4)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1
