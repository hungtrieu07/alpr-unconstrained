import os
import cv2
import traceback
import sys
import glob

if __name__ == '__main__':
    try:

        for file in glob.glob("/home/hungtrieu07/Downloads/LP_detection/images/train/" + "*.txt"):
            os.remove(file)

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)