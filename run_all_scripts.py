import os
directory_1 = '/Users/august/PycharmProjects/EITN35'

os.chdir(directory_1)
os.system("python mp4_conversion.py")
os.system("python video_splitting.py")
os.system("python test_set_creator.py")
os.system("python object_detection_v5.py")