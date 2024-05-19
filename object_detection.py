#import libraries
import cv2
import os
import numpy as np

#setup input and output paths
current_directory = os.path.dirname(os.path.realpath(__file__))

input_video_filename = 'luxonis_task_video.mp4'
output_video_filename = 'output_video.mp4'
data_filename = 'data.txt'

input_video_path = os.path.join(current_directory, input_video_filename)
output_video_path = os.path.join(current_directory, output_video_filename)
data_path = os.path.join(current_directory, data_filename)

#check for valid video in input_video_path
input_video = cv2.VideoCapture(input_video_path)
if not input_video.isOpened():
    print("Error: Could not open video.")
    exit()

#setup video output properties (same as input)
fps = input_video.get(cv2.CAP_PROP_FPS)
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = int(input_video.get(cv2.CAP_PROP_FOURCC))

output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#function to detect circles
def detect_circles(frame, i):
    #configuration
    threshold = 10
    kernel_size = (15,15)
    sigma = 5
    dp = 1
    minDist = 50
    param1 = 20
    param2 = 20
    minRadius = 10
    maxRadius = 0

    circles = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) #convert to binary frame
    binary_blurred = cv2.GaussianBlur(binary, kernel_size, sigma) #blur frame for easier circle deteciton
    cir = cv2.HoughCircles(binary_blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius) #find circles with Houghcircles
    
    if cir is not None:
        cir = cir[0, :]
        cir = [[int(round(x)) for x in circle] for circle in cir] #round everything
        for (x, y, r) in cir:
            area = np.pi * (r ** 2)
            colour = frame[y, x].tolist()
            circles.append([None, i, 'Circle', x, y, r, round(area), colour]) #[ID, Frame no., Shape, x-coordinate, y-coordinate, radius, area, colour]
            cv2.circle(frame, (x, y), r, [0,0,0], -1) #delete found object from frame (easier square detection)

    return circles, frame

#function to detect rectangles 
def detect_rectangles(frame, i):
    #configuration
    threshold = 50
    kernel_size = (3,3)
    min_area = 500
    approximation_accuracy  = 0.01

    rectangles = []
    partial_rectangles = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) #convert to binary frame
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones(kernel_size, np.uint8)) #clean frame from edges of circles (not pixel perfect circle detection)
    contours,_ = cv2.findContours(opening, 1, 2) #detect contours

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, approximation_accuracy*cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if area >= 100: #discard small contours (false positives)
            if (len(approx) == 4) or (area >= min_area):
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    x = int(M["m10"] / M["m00"])
                    y = int(M["m01"] / M["m00"])
                colour = frame[y, x].tolist() 
                if len(approx) == 4: #if rectangle
                    _,_, w, h = cv2.boundingRect(cnt)
                    rectangles.append([None, i, 'Rectangle', x, y, [w, h], round(area), colour]) #[ID, Frame no., Shape, x-coordinate, y-coordinate, [width, height], area, colour]
                elif area >= min_area: #if big enough contour (not detected as circle or rectangle)
                    partial_rectangles.append([None, i, 'Partial_Rectangle', x, y, cnt, round(area), colour]) #detect as Partial_Rectangle by process of elimination (not circle or rectangle)
        
    return rectangles, partial_rectangles

#function to draw object outlines   
def draw_outline(frame, circles, rectangles, partial_rectangles):
    #configuration
    colour = [255, 255, 255]
    thickness = 2
    dot_size = 2

    for circle in circles:
        x = circle[3]
        y = circle[4]
        radius = circle[5]

        cv2.circle(frame, (x, y), dot_size, colour, -1) #centre dot
        cv2.circle(frame, (x, y), radius, colour, thickness) #outline

    for rectangle in rectangles:
        x = rectangle[3]
        y = rectangle[4]
        w, h = rectangle[5]

        cv2.circle(frame, (x, y), dot_size, colour, -1) #centre dot
        cv2.rectangle(frame, (x - int(w/2), y - int(h/2)), (x + int(w/2), y + int(h/2)), colour, thickness) #outline

    for partial_rectangle in partial_rectangles:
        x = partial_rectangle[3]
        y = partial_rectangle[4]
        coordinates = partial_rectangle[5]

        cv2.circle(frame, (x, y), dot_size, colour, -1) #centre dot
        cv2.drawContours(frame, [coordinates], -1, colour, thickness) #outline

#function to link the same object across multiple frames with ID
def track(previous_frames, next_id, objects):
    next_id = next_id
    if next_id > 0: #if not first frame
        for new_objects in objects: #go through the previous 2 frames and try to make a link based on criteria
            for previous_objects in previous_frames:
                if (new_objects[2] == 'Circle') and (previous_objects[2] == 'Circle'):
                    if ((abs(new_objects[3] - previous_objects[3]) <= 20) and #x
                        (abs(new_objects[4] - previous_objects[4]) <= 20) and #y
                        (abs(new_objects[5] - previous_objects[5]) <= 4) and #radius
                        (abs(new_objects[6] - previous_objects[6]) <= 1500) and #area
                        (abs(new_objects[7][0] - previous_objects[7][0]) <= 10) and #B channel
                        (abs(new_objects[7][1] - previous_objects[7][1]) <= 10) and #G channel
                        (abs(new_objects[7][2] - previous_objects[7][2]) <= 10)): #R channel
                        new_objects[0] = previous_objects[0] #ID
                        break

                elif (new_objects[2] == 'Rectangle') and (previous_objects[2] == 'Rectangle'):
                    if ((abs(new_objects[3] - previous_objects[3]) <= 20) and #x
                        (abs(new_objects[4] - previous_objects[4]) <= 20) and #y
                        (abs(new_objects[5][0] - previous_objects[5][0]) <= 4) and #width
                        (abs(new_objects[5][1] - previous_objects[5][1]) <= 4) and #height
                        (abs(new_objects[6] - previous_objects[6]) <= 500) and #area
                        (abs(new_objects[7][0] - previous_objects[7][0]) <= 10) and #B channel
                        (abs(new_objects[7][1] - previous_objects[7][1]) <= 10) and #G channel
                        (abs(new_objects[7][2] - previous_objects[7][2]) <= 10)): #R channel
                        new_objects[0] = previous_objects[0] #ID
                        break

                elif (new_objects[2] == 'Rectangle') and (previous_objects[2] == 'Partial_Rectangle'):
                    if ((abs(new_objects[3] - previous_objects[3]) <= 20) and #x
                        (abs(new_objects[4] - previous_objects[4]) <= 20) and #y
                        (new_objects[6] >= previous_objects[6]) and #area
                        (abs(new_objects[7][0] - previous_objects[7][0]) <= 10) and #B channel
                        (abs(new_objects[7][1] - previous_objects[7][1]) <= 10) and #G channel
                        (abs(new_objects[7][2] - previous_objects[7][2]) <= 10)): #R channel
                        new_objects[0] = previous_objects[0] #ID
                        break

                elif (new_objects[2] == 'Partial_Rectangle') and (previous_objects[2] == 'Rectangle'):
                    if ((abs(new_objects[3] - previous_objects[3]) <= 20) and #x
                        (abs(new_objects[4] - previous_objects[4]) <= 20) and #y
                        (new_objects[6] <= previous_objects[6]) and #area
                        (abs(new_objects[7][0] - previous_objects[7][0]) <= 10) and #B channel
                        (abs(new_objects[7][1] - previous_objects[7][1]) <= 10) and #G channel
                        (abs(new_objects[7][2] - previous_objects[7][2]) <= 10)): #R channel
                        new_objects[0] = previous_objects[0] #ID
                        break

                elif (new_objects[2] == 'Partial_Rectangle') and (previous_objects[2] == 'Partial_Rectangle'):
                    if ((abs(new_objects[3] - previous_objects[3]) <= 20) and #x
                        (abs(new_objects[4] - previous_objects[4]) <= 20) and #y
                        (abs(new_objects[7][0] - previous_objects[7][0]) <= 20) and #B channel
                        (abs(new_objects[7][1] - previous_objects[7][1]) <= 10) and #G channel
                        (abs(new_objects[7][2] - previous_objects[7][2]) <= 10)): #R channel
                        new_objects[0] = previous_objects[0] #ID
                        break

            #if no link found
            if new_objects[0] == None:
                new_objects[0] = next_id
                next_id += 1

    #if first frame
    else:
        for new_objects in objects:
            new_objects[0] = next_id
            next_id += 1

    return objects, next_id

#function to draw ID on top of an object
def draw_text(objects):
    #configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    outline_colour = [255, 255, 255]
    text_colour = [0, 0, 255]
    thickness = 2

    for object in objects:
        text = str(object[0])
        x = object[3] + 5
        y = object[4] - 5

        #border
        cv2.putText(frame, text, (x - 1, y - 1), font, font_scale, outline_colour, thickness + 2)
        cv2.putText(frame, text, (x + 1, y - 1), font, font_scale, outline_colour, thickness + 2)
        cv2.putText(frame, text, (x - 1, y + 1), font, font_scale, outline_colour, thickness + 2)
        cv2.putText(frame, text, (x + 1, y + 1), font, font_scale, outline_colour, thickness + 2)

        #main
        cv2.putText(frame, text, (x, y), font, font_scale, text_colour, thickness)

#function to save data into a file
def save_data(objects, i):
    #delete previous file if starting from beginning
    if i == 1:
        if os.path.exists(data_path):
            os.remove(data_path)

    #save data
    with open(data_path,'a') as f:
        for object in objects:
            if object[2] == 'Partial_Rectangle':
                object[5] = [[point[0][0], point[0][1]] for point in object[5]] #cleanup the structure of contour point coordinates (less messy save file)
            f.write(f'{object}\n')

#config
frames_to_keep = 2

#initialize variables
i = 0
next_id = 0
previous_objects = []

#main loop to process all video frames
while True:
    ret, frame = input_video.read() #get next frame
    if not ret: #if out of frames
        break

    i += 1 #keep track of frame no.

    circles, detect_frame = detect_circles(frame.copy(), i)
    rectangles, partial_rectangles = detect_rectangles(detect_frame, i)

    draw_outline(frame, circles, rectangles, partial_rectangles)
    combined, next_id = track(previous_objects, next_id, circles + rectangles + partial_rectangles) #link same objects together with ID (across frames)
    draw_text(combined) #draw ID on top of an object

    save_data(combined, i) #save detected objects
    output_video.write(frame) #add frame to a video output

    previous_objects = [obj for obj in previous_objects if not obj[1] <= i - frames_to_keep] #remove objects that are more than frames_to_keep frames old
    previous_objects.extend(combined) #add current objects

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #q to interrupt
        break

#properly release resources
input_video.release()
output_video.release()
cv2.destroyAllWindows()