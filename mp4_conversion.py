import os
import ffmpy

PRINT_DEBUG = True

#Dir where video files to be converted are located
directory = 'C:/Users/eitn35/Documents/EITN35/video_files/'
os.chdir(directory)

#Create "converted" folder if it does not exist
try:
    if not os.path.exists('converted'):
        os.makedirs('converted')
except OSError:
    print ('Error: Creating directory of data')

#Iterate over video files and convert, then move to converted folder
for filename in os.listdir(directory):
    if (filename.endswith(".asf") | filename.endswith(".avi")):
        if PRINT_DEBUG: print("Starting conversion of " + filename + "...")
        ff = ffmpy.FFmpeg(
            inputs={filename: None},
            outputs={"converted_" + filename.split('.')[0]+'.mp4': None}
        )
        ff.run()
        if PRINT_DEBUG: print("Finished conversion of " + filename + ".")
        os.rename(
            directory + "/"+ "converted_" + filename.split(".")[0]+".mp4",
            directory + "/converted/" + "converted_" +filename.split(".")[0]+".mp4")
        if PRINT_DEBUG: print("File " + filename + " moved to converted folder.")