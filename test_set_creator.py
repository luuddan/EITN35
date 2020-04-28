import os
import random as rand

#Dir where the test_set folder should be placed
directory_1 = '/Users/august/Documents/EITN35_AIQ/video_files/'

#Dir where the frames folder is located
directory_2 = '/Users/august/Documents/EITN35_AIQ/video_files/frames/'
os.chdir(directory_1)

# Create "frames" folder if it does not exist
try:
    if not os.path.exists('test_set'):
        os.makedirs('test_set')
except OSError:
    print('Error: Creating directory of data')

os.chdir(directory_2)

count = 0
wantedSplit = [0.8, 0.1, 0.1] #training, validation, test
noExtract = len(os.listdir(directory_2))*wantedSplit[2]
PRINT_DEBUG = True

while count < noExtract:
    index = rand.randint(1, len(os.listdir(directory_2)))
    file_list = os.listdir(directory_2)

    if PRINT_DEBUG : print(str(file_list[index]) + " exported to test_set...")

    os.rename(
        directory_2 + str(file_list[index]),
        directory_1 + '/test_set/' + str(file_list[index])
    )
    count += 1

print(str(noExtract) + " files exported to test_set")

