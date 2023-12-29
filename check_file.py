import glob
import os
path = "/home/hungtrieu07/Downloads/plate/server/output/plate"
os.chdir(path)

for filename in glob.glob('*.txt'):
    # print(filename)
    if (os.stat(filename).st_size == 0):
        os.remove(f'{os.path.splitext(filename)[0]}.jpg')
        os.remove(filename)

    # print(filename)
    # count = 0
    # for line in lines:
    #     count+=1
    #     print(line)
    # time.sleep(0.01)
    # if(count > 1):
    #     os.remove(filename)
    #     os.remove(path + os.path.splitext(filename)[0] + '.jpg')
    #     print('Removed %s and reference image' %(filename))