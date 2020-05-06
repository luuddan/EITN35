import os
#project file directory
project_directory = '/Users/august/PycharmProjects/EITN35/'
os.chdir(project_directory)

#results directory, will be created if it non-existent
results = project_directory + '/training_results/'

# Create "training_results" folder if it does not exist
try:
    if not os.path.exists('training_results'):
        os.makedirs('training_results')
except OSError:
    print('Error: Creating directory of data')

#retrieves number of previous runs
runNo = len(os.listdir(results))+1
file_name = 'object_detection_run_' + str(runNo) + '.txt'

#creates new results file
os.chdir(results)
file = open(file_name, "w")
file.write("Object Detection Run, No:" + str(runNo))
file.close()

#trains and tests the models with different layer configurations, writes acc, acc_loss, loss and val_loss to file
os.chdir(project_directory)
os.system("python CNN_from_scratch_cats_dogs.py")
# os.system("python model2.py")
# os.system("python model3.py")
# os.system("python model4.py")

#Open the file back and print the contents
os.chdir(results)
print("Trying to open " + 'object_detection_run_' + str(runNo) + '.txt')
f = open('object_detection_run_' + str(runNo) + '.txt', "r")
if f.mode == 'r':
    contents =f.read()
    print(contents)