from ctypes import *                                               # Import libraries
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from sklearn.cluster import MeanShift

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    #================================================================
    # 3.1 Purpose : Filter out Persons class from detections and get 
    #           bounding box centroid for each person detection.
    #================================================================
    # Focal length
    pro_peo = 12 # The appropriate number of people - how to get this by argue????
    F = 615
    red = (255, 0, 0)
    green =(0, 255, 0)
    blue = (0, 0, 255)

    if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
        person_detection = 0
        mask_detection = 0
        no_mask_detection = 0
        incorrect_detection = 0

        pos_dict = dict()
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0								# We inialize a variable called ObjectId and set it to 0

        for detection in detections:				# In this if statement, we filter all the detections for persons only
            # Check for the only person name tag 
            name_tag = str(detection[0].decode())   # Coco file has string of all the names
            x, y, w, h = detection[2][0],\
                         detection[2][1],\
                         detection[2][2],\
                         detection[2][3]      	# Store the center points of the detections
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox            
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            
            if name_tag == 'mask_weared':                
                cv2.rectangle(img, pt1, pt2, green, 1)
                cv2.putText(img,
                            detection[0].decode() +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            green, 2)
                mask_detection += 1
            
            if name_tag == 'mask_off':                
                cv2.rectangle(img, pt1, pt2, red, 1)
                cv2.putText(img,
                            detection[0].decode() +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            red, 2)
                no_mask_detection += 1
                
            if name_tag == 'mask_incorrect':
                cv2.rectangle(img, pt1, pt2, red, 1)
                cv2.putText(img,
                            detection[0].decode() +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            blue, 2)
                incorrect_detection += 1
                
            if name_tag == 'mask_unknown':                
                cv2.rectangle(img, pt1, pt2, blue, 1)
                cv2.putText(img,
                            detection[0].decode() +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt2[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            blue, 2)
            
            person_detection += 1
                
            centroid_dict[objectId] = (xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox
            x_mid = round((xmin+xmax)/2, 4)
            y_mid = round((ymin+ymax)/2, 4)
            height = round(ymax-ymin,4)

            distance = (22 * F)/height   # average human head size = 22
            print("Distance(cm):{dist}\n".format(dist = distance))

            x_mid_cm = (x_mid * distance) / F
            y_mid_cm = (y_mid * distance) / F
            pos_dict[objectId] = (x_mid_cm,y_mid_cm,distance)

            objectId += 1 #Increment the index for each detection
                
        # distance
        dist_num = 0
        close_objects = set()
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i < j:
                    dist_num += 1
                    dist = np.sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))
                    #print(dist)
                    if dist < 100:
                        close_objects.add(i)
                        close_objects.add(j)
        
        for i in pos_dict.keys():
            if i in close_objects:
                COLOR = [255,0,0]
            else:
                COLOR = [0,255,50]
            
            (startX, startY, endX, endY) = centroid_dict[i]

            cv2.rectangle(img, (startX, startY), (endX, endY), COLOR, 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
       
            cv2.putText(img, 'Depth: {i} cm'.format(i=pos_dict[i][2]), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

        #for clustering
        if len(pos_dict) != 0:
            pos_arr = np.zeros((len(pos_dict), 3))
            for s in range(len(pos_dict)):
                pos_arr[s] = pos_dict[s]
            #print(pos_arr)
            band = 100        # bandwidth = 100(1m)
            clust = MeanShift(bandwidth=band).fit(pos_arr)   
            labels = clust.labels_
            #print(labels)
            centers = clust.cluster_centers_
            #print(centers)

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            print("cluster number: %s"%str(n_clusters_))
            max_center = np.zeros(n_clusters_)
            cluster_den = np.zeros(n_clusters_)

            for k in range(n_clusters_):
                my_members = labels == k
                #print(my_members)
                member = pos_arr[my_members]
                print(member)
                n_member = len(member)

                if n_member == 1:
                    cluster_den[k] = 1/pow(band, 2)

                else:
                    center = centers[k]
                    #print(center)
                    distance_to_center = np.zeros(len(member))
                    for q in range(n_member):
                        distance_to_center[q] = np.power((member[q][0]-center[0]),2)+ np.power((member[q][1]-center[1]),2) + np.power((member[q][2]-center[2]),2) # x_mid, y_mid, distance..
                    max_center[k] = np.max(distance_to_center)
                    cluster_den[k] = len(member)/(max_center[k]*0.0001)
      
            #cluster_den과 centers간 거리를 이용하여 혼잡도 구하기
            total_den = 0
            """
            for a in range(n_clusters_):
                for b in range(n_clusters_):
                    if a<b:
                        dist_c = np.sqrt(pow(centers[a][0]-centers[b][0],2)+pow(centers[a][1]-centers[b][1],2)+pow(centers[a][2]-centers[b][2],2))
                        total_den += (cluster_den[a] + cluster_den[b])/(dist_c*0.01)
            """
            for a in range(n_clusters_):
                total_den += cluster_den[a]
            total_den = total_den/n_clusters_
            
            print("total_density: %s "%str(total_den))

            #opencv에 나타내기

        # estimate congestion
        congestion = person_detection / pro_peo
        cv2.putText(img,
                    "Total people: %s"%str(person_detection) + "Mask weared people: %s "%str(mask_detection) + "Mask off people: %s"%str(no_mask_detection), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [0, 255, 50], 2)
        
        if congestion < 0.8:
            level = "spare"
            cv2.putText(img,
                    "congestion: %s "% level + "(%s)"%str(round(congestion,4)), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [0, 255, 50], 2)        
        elif congestion < 1.3:
            level = "normal"
            cv2.putText(img,
                    "congestion: %s "% level + "(%s)"%str(round(congestion,4)), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [255, 255, 0], 2)
        else:
            level = "congestion"
            cv2.putText(img,
                    "congestion: %s "% level + "(%s)"%str(round(congestion,4)), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [255, 0, 0], 2)
            
        # estimate risk
        #close_object_num = total_den
        
        if no_mask_detection > 0:
            level = "high risk"
            cv2.putText(img,
                    "risk: %s "% level, (10,125), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [255, 0, 0], 2)   
        else:            
            if dist_num > 0:
                risk = total_den
        
                if risk < 3:
                    level = "low risk"
                    cv2.putText(img, "risk: %s "% level + "(%s)"%str(round(risk, 4)), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 50], 2)
                elif risk < 5:
                    level = "risk"
                    cv2.putText(img, "risk: %s "% level + "(%s)"%str(round(risk, 4)), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 0], 2)
                else:
                    level = "high risk"
                    cv2.putText(img, "risk: %s "% level + "(%s)"%str(round(risk, 4)), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
 
    return img



netMain = None
metaMain = None
altNames = None


def YOLO():
   
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4-mask.cfg"                                 
    weightPath = "./yolov4-mask_final.weights"                                 # 나중에 train한 weight 넣기  -  backup에서 가져오면 됨              
    metaPath = "./cfg/mask.data" 


    if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode( 
            "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)                                      # Uncomment to use Webcam
    cap = cv2.VideoCapture("test.mp4")                             # Local Stored video detection - Set input video
    frame_width = int(cap.get(3))                                   # Returns the width and height of capture video
    frame_height = int(cap.get(4))
    # Set out for video writer
    out = cv2.VideoWriter(                                          # Set the Output path for video writer
        "output_test.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (frame_width, frame_height))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
    while True:                                                      # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()                                 # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:                                                  # Check if frame present otherwise he break the while loop
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.                                                                                   
        image = cvDrawBoxes(detections, frame_resized)               # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        #cv2.imshow('Demo', image)                                    # Display Image window
        cv2.waitKey(3)
        out.write(image)                                             # Write that frame into output video
    cap.release()                                                    # For releasing cap and out. 
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":  
    YOLO()                                                           # Calls the main function YOLO()