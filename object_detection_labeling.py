#!/usr/bin/env python
# coding: utf-8

# Object Detection With YOLOv3 in Keras

# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import pandas as pd
import os
import math
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

#directory_1 = "/Users/august/Documents/EITN35_AIQ"
directory_1 = "../EITN35_Resources/"
os.chdir(directory_1)


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white', lw=2)
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    #pyplot.show()


# load yolov3 model
model = load_model('model.h5', compile=False)
# define the expected input shape for the model
input_w, input_h = 416, 416

# In[77]:


# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# define the probability threshold for detected objects
class_threshold = 0.6
dist_threshold = 150  # gör denna beroende på framerate som valts


# In[78]:


class Mem_Object:
    def __init__(self, label, xmin, xmax, ymin, ymax):
        self.label = label
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.mid = [(xmin + xmax) / 2, (ymin + ymax) / 2]


# In[79]:


def dist(p1, p2):
    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return distance


# In[80]:


def check_new_object(v_labels, v_boxes, v_scores, frames_memory):
    print("Inside check_new_object")

    # create temp counter for this frames
    temp_counter = pd.DataFrame(0, columns=['Number'], index=labels)
    new_objects = pd.DataFrame(data=0, columns=range(10), index=['T1'])

    # add all new objects from v_labels with attributes
    for i in range(len(v_labels)):
        new_objects[i]['T1'] = Mem_Object(v_labels[i], v_boxes[i].xmin, v_boxes[i].xmax, v_boxes[i].ymin,
                                          v_boxes[i].ymax)
        print("Label i objects" + v_labels[i])

    print("Length of new_objects: " + str(len(new_objects)))
    print("Content of new_objects" + str(new_objects) + "... End of contents")
    # print("Content of new_objects" + new_objects)

    # loop through new objects and compare to last frame
    for i in new_objects:
        # loop until zero -> no more objects
        if new_objects[i][0] == 0: break  # eller continue

        temp_counter.loc[str(v_labels[i])] += 1
        # print("New object up counted!")
        # print("Temp counter inside function: ")
        # print(temp_counter)

        # placehold_count += 1 #hundar eller persons
        uniqueObj = True

        # current frame obj
        cf_obj = new_objects[i][0]
        print("Mid new object" + str(cf_obj.mid))

        # Check memory for objects in last frame, if same type of object but too close, count down!
        for j in frames_memory:
            lf_obj = frames_memory[j][0]  # index 0 is 'F1'

            if (lf_obj != 0):

                if (lf_obj.label == cf_obj.label):
                    print("Mid old object" + str(lf_obj.mid))
                    distance = dist(lf_obj.mid, cf_obj.mid)
                    print("Distance between objects is: " + str(distance))

                    # if same type of object is too close, conclude same object, count down.
                    if (distance < dist_threshold):
                        temp_counter.loc[str(v_labels[i])] -= 1
                        print("Same object!")
                        uniqueObj = False

        if uniqueObj:
            print("New object!")  # syns ej på första

    # Update frames_memory
    for i in range(4):
        frames_memory.loc['F' + str(5 - i)] = frames_memory.loc['F' + str(4 - i)]

    # Overwrite first row with new objetcts
    frames_memory.loc['F1'] = new_objects.loc['T1']

    # frames_memory[rand.randint(5,9)]['F1'] = Mem_Object("person", 1, 1, 1, 1)
    # print("Temp counter")
    # print(temp_counter)
    # print("Frames memory")
    # print(frames_memory)
    return temp_counter

frames_memory = pd.DataFrame(data=0, columns=range(10), index=['F1', 'F2', 'F3', 'F4', 'F5'])
# Ladda in ett objekt
frames_memory

big_counter = pd.DataFrame(0, columns=['Number'], index=labels)

#Dir to be iterated
directory_2 = 'C:/Users/eitn35/Documents/EITN35/video_files/frames/'
#directory_2 = "/Users/august/Documents/EITN35_AIQ/video_files/frames/"
directory_3 = "C:/Users/eitn35/Documents/EITN35/video_files/test_set/"
os.chdir(directory_2)

# Create "unlabeled_images" folder if it does not exist
try:
    if not os.path.exists('unlabeled_images'):
        os.makedirs('unlabeled_images')
    if not os.path.exists('autolabeled_images'):
        os.makedirs('autolabeled_images')
except OSError:
    print('Error: Creating directory of data')

#MÅSTE GÖRAS LÄNGRE OM VI HITTAR MER ÄN ETT OBJEKT I VARJE BILD, GÖR HELLRE FÖR LÅNG OCH DROPPA
annotation_df = pd.DataFrame(data=0,index=np.arange(len(os.listdir(directory_2))),columns="image xmin ymin xmax ymax label".split())
index = 0

# Store found objects and their positions from previous frame in vector.
# When new object found compare type and position, then decide to count or not
# lf_memory = pd.DataFrame(0,columns=['Label','Middle'],index=range(10))


for photo_filename in os.listdir(directory_2):
    if not photo_filename.endswith('jpg'):continue
    #photo_filename = 'frame_' + str(i + 8) + '.jpg'
    # define our new photoc
    # photo_filename = 'man_on_scooter.jpg'
    # load and prepare image
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
    # make prediction
    yhat = model.predict(image)

    # summarize the shape of the list of arrays
    #print([a.shape for a in yhat])

    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    contains_person = False
    #load dataframe for csv export

    persons = 0
    bikes = 0
    dogs = 0

    for i in range(len(v_labels)):
       if v_labels[i] == 'person':
           persons += 1

       if v_labels[i] == 'bicycle':
           bikes += 1

       if v_labels[i] == 'dog':
           dogs += 1



    # summarize what we found
    # temp_2_counter = check_new_object(v_labels, v_boxes, v_scores, frames_memory)

    # print(temp_2_counter)
    # big_counter.loc["person"] = big_counter.loc["person"] + temp_2_counter.loc["person"]
    # big_counter.loc["dog"] = big_counter.loc["dog"] + temp_2_counter.loc["dog"]

    # draw what we found
    # draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
    # print(str(big_counter))

    #move files that couldn't be labeled with YOLO
    counter = persons + bikes + dogs
    if (counter == 0):

        os.rename(
            directory_2 + photo_filename,
            directory_2 + 'unlabeled_images/' + 'persons_0_dogs_0_bikes_0_' + photo_filename
        )
    else:
        os.rename(
            directory_2 + photo_filename,
            directory_2 + 'autolabeled_images/' + 'persons_' + str(persons) + '_dogs_' + str(dogs) + '_bikes_' + str(bikes) + '_' + photo_filename
        )



    #print("THIS IS ANNOTATIONS DF")
    #print(str(annotation_df))
    print('labeling file '+ photo_filename)



#Make CSV-file from auto-annotations
#annotation_df.to_csv('/Users/august/Documents/EITN35_AIQ/Annotations-export_YOLO_auto.csv', index=False, header=True)

#for labeled in os.listdir(directory_3):
 #   if not labeled.endswith('jpg'): continue
  #  os.rename(
   #     directory_3 + labeled,
    #    directory_3 + 'autolabeled_images/' + labeled
    #)

# In[90]:


#big_counter.loc['dog']

# In[ ]:




