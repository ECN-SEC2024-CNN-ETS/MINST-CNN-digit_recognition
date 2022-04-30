# packages
import os, sys
import cv2
from PIL import Image
import numpy as np
from resizeimage import resizeimage

# global coordinates and drawing state
y = 0
x = 0
drawing = False


def draw_number():
    # image directory path--CHANGE FOR YOUR USAGE
    IMG_DIR = 'resources'

    size = (28, 28)

    # init a canvas
    canvas = np.zeros((244, 244, 1), np.uint8)

    # make canvas black
    canvas.fill(0)

    # mouse callback function
    def draw(event, current_x, current_y, flags, params):
        # hook up global variables
        global x
        global y
        global drawing

        # handle mouse down event
        if event == cv2.EVENT_LBUTTONDOWN:
            # update coordinates
            x = current_x
            y = current_y

            # enable drawing flag
            drawing = True

        # handle mouse move event
        elif event == cv2.EVENT_MOUSEMOVE:
            # draw only if mouse is down
            if drawing:
                # draw the line
                cv2.line(canvas, (current_x, current_y), (x, y), 255, thickness=3)

                # update coordinates
                x, y = current_x, current_y

        # handle mouse up event
        elif event == cv2.EVENT_LBUTTONUP:
            # disable drawing flag
            drawing = False

    # display the canvas in a window
    cv2.imshow('Draw', canvas)

    # bind mouse events
    cv2.setMouseCallback('Draw', draw)

    # infinite drawing loop
    while True:
        # update canvas
        cv2.imshow('Draw', canvas)

        # break out of a program on 'Esc' button hit
        if cv2.waitKey(1) & 0xFF == 27:
            # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
            # filename = "my_drawing.png"
            # canvas = cv2.resize(crop, dsize=size, interpolation=cv2.INTER_CUBIC)

            cv2.imwrite('utility/resources/handwritten_numbers.png', canvas)
            img = Image.open('utility/resources/handwritten_numbers.png')
            img = resizeimage.resize_contain(img, size)
            img.save('utility/resources/handwritten_numbers_resize.png', img.format)
            break
    # clean up windows
    cv2.destroyAllWindows()
