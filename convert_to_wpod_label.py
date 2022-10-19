import glob
import os
import cv2
path = "/home/hungtrieu07/Downloads/LP_detection/images/val/"
os.chdir(path)

for filename in glob.glob('*.txt'):
    image = cv2.imread(os.path.splitext(filename)[0] + '.jpg')
    height = image.shape[0]
    width = image.shape[1]

    f = open(filename, 'r')
    line = f.readline()
    line = line.split()

    x = float(line[1])
    y = float(line[2])
    w = float(line[3])
    h = float(line[4])

    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    tl_x = f'{x1:6f}'
    tl_y = f'{y1:6f}'
    tr_x = f'{x2:6f}'
    tr_y = f'{y1:6f}'
    br_x = f'{x2:6f}'
    br_y = f'{y2:6f}'
    bl_x = f'{x1:6f}'
    bl_y = f'{y2:6f}'

    output = '4,' + str(tl_x) + "," + str(tr_x) + "," + str(br_x) + "," + str(bl_x) + "," + str(tl_y) + "," + str(tr_y) + "," + str(br_y) + "," + str(bl_y) + ',,'

    fo = open(filename, 'w')
    fo.write(output)
    print('Writing to %s' % filename)
    fo.close()

