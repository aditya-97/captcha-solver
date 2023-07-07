from model.helpers import resize_to_fit
import numpy as np
import imutils
import cv2


def predict(image, model, lb):
    try:
        # Load the image and convert it to grayscale
        np_arr = np.fromstring(image, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.blur(image, (1, 1))
        ret, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

        dil = cv2.dilate(thresh, np.ones((1, 1), np.uint8))
        ero = cv2.erode(dil, np.ones((2, 1), np.uint8))

        contours = cv2.findContours(ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1] if imutils.is_cv2() else contours[0]
        contours = [cnt for cnt in contours if (cv2.contourArea(cnt) < 10000) and (cv2.contourArea(cnt) > 0)]
        contours = sorted(contours, key=lambda elem: (cv2.arcLength(elem, True),
                                                      cv2.contourArea(elem)), reverse=True)[:6]

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
            return None

        # Sort the detected letter images based on the x coordinate to make sure
        # we are processing them from left-to-right so we match the right image
        # with the right letter
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # Create an output image and a list to hold our predicted letters
        predictions = []

        # loop over the letters
        for letter_bounding_box in letter_image_regions:
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = resize_to_fit(letter_image, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

        return ''.join(predictions)
    except Exception as e:
        return None
