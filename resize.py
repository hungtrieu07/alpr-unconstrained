from PIL import Image
import glob
import os
path = "/home/hungtrieu07/Downloads/plate_trinam/img_plate_trinam"
os.chdir(path)

for filename in glob.glob('*.jpg'):
    image = Image.open(filename)
    new_image = image.resize((640, 640))
    new_image.save("/home/hungtrieu07/Downloads/plate_trinam/resize_image/" + "new_" + filename)