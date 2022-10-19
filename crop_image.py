import os
import cv2
import traceback
import sys
import blob

IMG_PATH = "/home/hungtrieu07/Downloads/plate_trinam/img_plate_trinam/"
TXT_PATH = "/home/hungtrieu07/Downloads/plate_trinam/txt_plate_trinam/"

if __name__ == '__main__':
    try:

        for imgfile in blob.blob(TXT_PATH + ".jpg"):

            image = cv2.imread(imgfile)
            height = image.shape[0]
            width = image.shape[1]

            with open(TXT_PATH + os.path.splitext(imgfile)[0] + ".txt") as f:
                line = f.readline()
                line = line.split()

            x1 = int(float(line[1]))
            y1 = int(float(line[2]))
            x2 = int(float(line[3]))
            y2 = int(float(line[4]))

            crop_img = image[y1:y2, x1:x2]
            cv2.imwrite("image7.jpg", crop_img)

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)