import cv2
import numpy as np
import os
from matplotlib import pyplot
from PIL import Image


# Dir where video files to be split are located
unlabeled_directory = '/Users/august/Documents/EITN35_AIQ/video_files/test_set/unlabeled_images/'

for frame_file in os.listdir(unlabeled_directory):
    if(frame_file.endswith("jpg")):
        data = pyplot.imread(unlabeled_directory+frame_file)
        # plot the image
        pyplot.imshow(data)
        pyplot.show()
        # input
        input1 = input()
        print("First input was: " + input1)

        # output
        if input1 == str(1):
            print("Input was: " + input1)
            os.rename(
                unlabeled_directory + frame_file,
                unlabeled_directory + 'done' + frame_file)

    pyplot.close()