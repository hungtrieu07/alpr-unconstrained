import sys
import traceback
import cv2
import numpy as np

if __name__ == '__main__':
    try:

        image_path = sys.argv[1]
        annotation_path = sys.argv[2]

        image = cv2.imread(image_path)
        height = image.shape[0]
        width = image.shape[1]

        f = open(annotation_path, 'r')
        line = f.readline()
        line = line.split()

        x = float(line[1])
        y = float(line[2])
        w = float(line[3])
        h = float(line[4])

        x1 = int((x - w/2) * width)
        y1 = int((y - h/2) * height)
        x2 = int((x + w/2) * width)
        y2 = int((y + h/2) * height)

        # start_point = (x1, y1)
        # end_point = (x2, y2)
        # color = (255,0,0)
        # thickness = 2

        crop_img = image[y1:y2, x1:x2]
        # cv2.imwrite("image1.jpg", crop_img)
        cv2.imshow('Image', crop_img)
        
    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)





