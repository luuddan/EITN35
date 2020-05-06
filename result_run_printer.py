import os

results = os.getcwd() + '/training_results/'

def print_to_file(acc, val_acc, loss, val_loss, epochs, layers, model):
    os.chdir(results)
    current_results = os.listdir(results)
    current_results.sort()
    working_file = current_results[len(os.listdir(results)) - 1]
    print("Latest file is: " + str(working_file))

    file = open(working_file, "a+")
    file.write("\n \n -----------------------------------")
    file.write("\n Model " + str(model) + " Run Result")
    file.write("\n -----------------------------------")
    file.write("\n Hidden Layers:           " + str(layers))
    file.write("\n Epochs:                  " + str(epochs))
    file.write("\n Training Set Accuracy:   " + str(acc))
    file.write("\n Validation Set Accuracy: " + str(val_acc))
    file.write("\n Training Loss:           " + str(loss))
    file.write("\n Validation Loss:         " + str(val_loss))
    file.close()