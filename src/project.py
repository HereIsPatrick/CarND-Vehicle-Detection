# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import time
import csv
from skimage.feature import hog
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

#---------------------------------------------------------------------------
#Exploring data
car_img_files = glob.glob('../../data/vehicles/**/*.png')
notcar_img_files = glob.glob('../../data/non-vehicles/**/*.png')

print("Car : ",len(car_img_files), "No Car : ", len(notcar_img_files))

fig, axs = plt.subplots(4,4, figsize=(16, 16))
fig.subplots_adjust(hspace = .4, wspace=.002)
axs = axs.ravel()

# Show car and no car.
for i in np.arange(8):
    img = cv2.imread(car_img_files[np.random.randint(0,len(car_img_files))])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    axs[i].set_title('car', fontsize=14)
    axs[i].imshow(img)
for i in np.arange(8,16):
    img = cv2.imread(notcar_img_files[np.random.randint(0,len(notcar_img_files))])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    axs[i].set_title('Not-Car', fontsize=14)
    axs[i].imshow(img)

#---------------------------------------------------------------------------
# Show Hog in car and not-car
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis: # return with image
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:  # return without image    
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features 
    
car_img = mpimg.imread(car_img_files[0])
car_features, car_hog_img = get_hog_features(car_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
notcar_img = mpimg.imread(notcar_img_files[0])
notcar_features, notcar_hog_img = get_hog_features(notcar_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)


f, ((axs0, axs1), (axs2, axs3)) = plt.subplots(2, 2, figsize=(7,7))
f.subplots_adjust(hspace = .4, wspace=.002)
axs0.imshow(car_img)
axs0.set_title('Car Image', fontsize=14)
axs1.imshow(car_hog_img, cmap='gray')
axs1.set_title('Car HOG', fontsize=14)
axs2.imshow(notcar_img)
axs2.set_title('Not-Car Image', fontsize=14)
axs3.imshow(notcar_hog_img, cmap='gray')
axs3.set_title('Not-Car HOG', fontsize=14)    

#---------------------------------------------------------------------------
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):

    features = []
    # process each image
    for file in imgs:

        image = mpimg.imread(file)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if hog_channel == 'ALL': # All Channel
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else: # one of Channel.
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        features.append(hog_features)

    return features
#---------------------------------------------------------------------------
# Feature extraction parameters
    
from collections import namedtuple
 
Config=namedtuple("Config",['color_space','orientations','pixels_per_cell', 'cells_per_block', 'hog_channel'])

# Step. set configure parameter
Configs=[Config('RGB',9,8,2,'ALL'),
         Config('HSV',9,8,2,1),
         Config('HSV',9,8,2,'ALL'),
         Config('HLS',9,8,2,0),
         Config('HLS',9,8,2,1),
         Config('HSV',9,8,2,2),
         Config('YUV',9,8,2,0),
         Config('YUV',9,8,2,'ALL')    
          ]

Extract_times = []
Accuracys = []
Train_times = []

# Step. initialize 
for i in range(len(Configs)):
    Extract_times.append(0.0)
    Accuracys.append(0.0)
    Train_times.append(0.0)


for config in Configs:

    print("Run config ",Configs.index(config))
    t = time.time()
    
    # Step. extract car and not-car feature
    car_features = extract_features(car_img_files, cspace=config.color_space, orient=config.orientations, 
                            pix_per_cell=config.pixels_per_cell, cell_per_block=config.cells_per_block, 
                            hog_channel=config.hog_channel)
    notcar_features = extract_features(notcar_img_files, cspace=config.color_space, orient=config.orientations, 
                            pix_per_cell=config.pixels_per_cell, cell_per_block=config.cells_per_block, 
                            hog_channel=config.hog_channel)
    t2 = time.time()

    # Step. Time to extract hog features
    Extract_times[Configs.index(config)] = round(t2-t, 2)

    # Step. Stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)  
    
    
    # Step. Define the label
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Step. Split data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
    
    
    # Step . Use linear SVC 
    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    
    # Step. store training time
    Train_times[Configs.index(config)] = round(t2-t, 2)

    # Step . store accuracy.
    Accuracys[Configs.index(config)] = round(svc.score(X_test, y_test), 4)

    
# Step. show hog parameter    
for config, extract_time, accuracy, train_time in zip(Configs, Extract_times, Accuracys, Train_times):
    print(Configs.index(config), config, extract_time, accuracy, train_time)    

# Step. write hog parameter to csv.
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)    
    for config, extract_time, accuracy, train_time in zip(Configs, Extract_times, Accuracys, Train_times):
        csvdata = [Configs.index(config), config.color_space, config.orientations, 
                   config.pixels_per_cell, config.cells_per_block, config.hog_channel, extract_time, train_time, accuracy
                   ]
        writer.writerow(csvdata)
    #print(Configs.index(config), config, extract_time, accuracy, train_time)    
    
#---------------------------------------------------------------------------
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_rectangles=False):
    
    # when car detect
    car_rects = []
    
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]

    # Step . color conversion
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: 
        ctrans_tosearch = np.copy(img_tosearch)   
    
    # Step. scale image
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # Step. select color space.
    if hog_channel == 'ALL':
        channel0 = ctrans_tosearch[:,:,0]
        channel1 = ctrans_tosearch[:,:,1]
        channel2 = ctrans_tosearch[:,:,2]
    else: 
        channel0 = ctrans_tosearch[:,:,hog_channel]

    # Step. Define blocks
    nxblocks = (channel0.shape[1] // pix_per_cell)+1
    nyblocks = (channel0.shape[0] // pix_per_cell)+1
    
    # Step. calculate next step
    window = 8*8
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Step . get feature.
    hog1 = get_hog_features(channel0, orient, pix_per_cell, cell_per_block, feature_vec=False)   
    if hog_channel == 'ALL':
        hog2 = get_hog_features(channel1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(channel2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
                        
            test_prediction = svc.predict(hog_features.reshape(1,-1))
            
            if test_prediction == 1 or show_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                car_rects.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return car_rects
#---------------------------------------------------------------------------
test_img = mpimg.imread('../test_images/test1.jpg')

ystart = 400
ystop = 660
scale = 1.5
colorspace = 'YUV'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'

rectangles = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, None, orient, pix_per_cell, cell_per_block, None, None)

print(len(rectangles), 'rectangles found')    

#---------------------------------------------------------------------------

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):

    img_copy = np.copy(img)
    is_random = False
    for bbox in bboxes:
        if color == 'random' or is_random:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            is_random = True
        cv2.rectangle(img_copy, bbox[0], bbox[1], color, thick)

    return img_copy
#---------------------------------------------------------------------------
    
test_img_rects = draw_boxes(test_img, rectangles)
plt.figure(figsize=(10,10))
plt.imshow(test_img_rects)

#---------------------------------------------------------------------------

def window_search(img_file, ystart_1, ystop_1, ystart_2, ystop_2, scale):

    test_img = mpimg.imread(img_file)
    
    rects = []
    
    rects.append(find_cars(test_img, ystart_1, ystop_1, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None, show_rectangles=True))

    rects.append(find_cars(test_img, ystart_2, ystop_2, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None, show_rectangles=True))
    
    rectangles = [item for sublist in rects for item in sublist] 
    test_img_rects = draw_boxes(test_img, rectangles, color='random', thick=2)
    plt.figure(figsize=(10,10))
    plt.imshow(test_img_rects)
    print('Number of boxes: ', len(rectangles))

ystart_1_0_top = 400
ystop_1_0_top = 464
ystart_1_0_bottom = 416
ystop_10_bottom = 480

ystart_1_5_top = 400
ystop_1_5_top = 496
ystart_1_5_bottom = 432
ystop_1_5_bottom = 528

ystart_2_0_top = 400
ystop_2_0_top = 528
ystart_2_0_bottom = 432
ystop_2_0_bottom = 560

ystart_3_5_top = 400
ystop_3_5_top = 596
ystart_3_5_bottom = 464
ystop_3_5_bottom = 660

window_search('../test_images/test1.jpg', ystart_1_0_top,ystop_1_0_top, ystart_1_0_bottom, ystop_10_bottom, 1.0)
window_search('../test_images/test1.jpg', ystart_1_5_top,ystop_1_5_top, ystart_1_5_bottom, ystop_1_5_bottom, 1.5)
window_search('../test_images/test1.jpg', ystart_2_0_top,ystop_2_0_top, ystart_2_0_bottom, ystop_2_0_bottom, 2.0)
window_search('../test_images/test5.jpg', ystart_3_5_top,ystop_3_5_top, ystart_3_5_bottom, ystop_3_5_bottom, 3.0)


#---------------------------------------------------------------------------

def creat_heatmap(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


#---------------------------------------------------------------------------
# Test  heatmap
heatmap_img = np.zeros_like(test_img[:,:,0])
heatmap_img = creat_heatmap(heatmap_img, rectangles)
plt.figure(figsize=(10,10))
plt.imshow(heatmap_img, cmap='hot')

#---------------------------------------------------------------------------

def apply_threshold(heatmap, threshold):
    # set pixel zero below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap

heatmap_img = apply_threshold(heatmap_img, 1)
plt.figure(figsize=(10,10))
plt.imshow(heatmap_img, cmap='hot')

#---------------------------------------------------------------------------
labels = label(heatmap_img)
plt.figure(figsize=(10,10))
plt.imshow(labels[0], cmap='gray')
print(labels[1], ' cars found')


#---------------------------------------------------------------------------
def draw_labeled_bboxes(img, labels):

    rects = []
    for car_count in range(1, labels[1]+1):
        nonzero = (labels[0] == car_count).nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Step. define bounding fo boxes.
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        
        # Step. Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    return img, rects

# Step. draw label car bounding.
draw_img, rect = draw_labeled_bboxes(np.copy(test_img), labels)

# Step. display
plt.figure(figsize=(10,10))
plt.imshow(draw_img)

#---------------------------------------------------------------------------

test_img = mpimg.imread('../test_images/test1.jpg')

rectangles = []

colorspace = 'YUV'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'


scale = 1.0
rectangles.append(find_cars(test_img, ystart_1_0_top, ystop_1_0_top, scale, colorspace, hog_channel, svc, None, 
                       orient, pix_per_cell, cell_per_block, None, None))

scale = 1.0
rectangles.append(find_cars(test_img, ystart_1_0_bottom, ystop_10_bottom, scale, colorspace, hog_channel, svc, None, 
                       orient, pix_per_cell, cell_per_block, None, None))

scale = 1.5
rectangles.append(find_cars(test_img, ystart_1_5_top, ystop_1_5_top, scale, colorspace, hog_channel, svc, None, 
                       orient, pix_per_cell, cell_per_block, None, None))

scale = 1.5
rectangles.append(find_cars(test_img, ystart_1_5_bottom, ystop_1_5_bottom, scale, colorspace, hog_channel, svc, None, 
                       orient, pix_per_cell, cell_per_block, None, None))

scale = 2.0
rectangles.append(find_cars(test_img, ystart_2_0_top, ystop_2_0_top, scale, colorspace, hog_channel, svc, None, 
                       orient, pix_per_cell, cell_per_block, None, None))

scale = 2.0
rectangles.append(find_cars(test_img, ystart_2_0_bottom, ystop_2_0_bottom, scale, colorspace, hog_channel, svc, None, 
                       orient, pix_per_cell, cell_per_block, None, None))

scale = 3.5
rectangles.append(find_cars(test_img, ystart_3_5_top, ystop_3_5_top, scale, colorspace, hog_channel, svc, None, 
                       orient, pix_per_cell, cell_per_block, None, None))

scale = 3.5
rectangles.append(find_cars(test_img, ystart_3_5_bottom, ystop_3_5_bottom, scale, colorspace, hog_channel, svc, None, 
                       orient, pix_per_cell, cell_per_block, None, None))

# apparently this is the best way to flatten a list of lists
rectangles = [item for sublist in rectangles for item in sublist] 

test_img_rects = draw_boxes(test_img, rectangles, color='random', thick=2)

heatmap_img = np.zeros_like(test_img[:,:,0])
heatmap_img = creat_heatmap(heatmap_img, rectangles)
heatmap_img = apply_threshold(heatmap_img, 1)
labels = label(heatmap_img)
draw_img, rects = draw_labeled_bboxes(np.copy(test_img), labels)

plt.figure(figsize=(10,10))
plt.imshow(test_img_rects)

plt.figure(figsize=(10,10))
plt.imshow(draw_img)


    




#---------------------------------------------------------------------------
#Video pipeline

# class for collect data
class Vehicle_Detect():
    def __init__(self):
        # History of frames.
        self.prev_rects = [] 
        
    def add_rects(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > 15:
            self.prev_rects = self.prev_rects[len(self.prev_rects)-15:]

def process_frame_for_video(img):

    rectangles = []

    colorspace = 'YUV' 
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    
    scale = 1.0
    rectangles.append(find_cars(img, ystart_1_0_top, ystop_1_0_top, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    scale = 1.0
    rectangles.append(find_cars(img, ystart_1_0_bottom, ystop_10_bottom, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    scale = 1.5
    rectangles.append(find_cars(img, ystart_1_5_top, ystop_1_5_top, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    scale = 1.5
    rectangles.append(find_cars(img, ystart_1_5_bottom, ystop_1_5_bottom, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    scale = 2.0
    rectangles.append(find_cars(img, ystart_2_0_top, ystop_2_0_top, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    scale = 2.0
    rectangles.append(find_cars(img, ystart_2_0_bottom, ystop_2_0_bottom, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))

    scale = 3.5
    rectangles.append(find_cars(img, ystart_3_5_top, ystop_3_5_top, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))

    scale = 3.5
    rectangles.append(find_cars(img, ystart_3_5_bottom, ystop_3_5_bottom, scale, colorspace, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
 
    rectangles = [item for sublist in rectangles for item in sublist] 
    
    if len(rectangles) > 0:
        vehicle_detect.add_rects(rectangles)
    
    heatmap_img = np.zeros_like(img[:,:,0])
    for rect_set in vehicle_detect.prev_rects:
        heatmap_img = creat_heatmap(heatmap_img, rect_set)
    heatmap_img = apply_threshold(heatmap_img, 1 + len(vehicle_detect.prev_rects)//2)
     
    labels = label(heatmap_img)
    draw_img, rect = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img


vehicle_detect = Vehicle_Detect()

project_video_out_file = '../project_video_out.mp4'
clip = VideoFileClip('../project_video.mp4')
#project_video_out_file = '../test_video_out.mp4'
#clip = VideoFileClip('../test_video.mp4')
clip_out = clip.fl_image(process_frame_for_video)
clip_out.write_videofile(project_video_out_file, audio=False)

