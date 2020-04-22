# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:46:52 2020

@author: johan
"""



def hole_frame_reader(working_dir,frame_queue,image_size):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       def hole_frame_reader(working_dir,frame_queue,image_size):
    '''
    hole_images_folder = os.path.join(working_dir,"frames_without_bees")
    all_images = file_utils.get_all_image_paths_in_folder(hole_images_folder)
    
    all_detections = []
    num_holes_detected = []
    for image_path in all_images:
        
        image = Image.open(image_path)
        resized_image = image.resize(image_size)
        image_np = np.asarray(resized_image)         
        image_expand = np.expand_dims(image_np, 0)
        is_done = Event()
        detections_dict = {}
        
        queue_item = PrioritizedItem(1,(image_expand,detections_dict,is_done))
        frame_queue.put(queue_item)
        is_done.wait()
        
        detections = []
        for i,score in enumerate(detections_dict['detection_scores']):
            if score >= min_confidence_score:
                top = detections_dict['detection_boxes'][i][0]
                left = detections_dict['detection_boxes'][i][1]
                bottom = detections_dict['detection_boxes'][i][2]
                right = detections_dict['detection_boxes'][i][3]
                detection_class = detections_dict['detection_classes'][i]
                detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "class": detection_class})
        
        detections = eval_utils.non_max_suppression(detections,0.5)
        all_detections.append(detections)
        num_holes_detected.append(len(detections))
    most_frequent_answer = max(set(num_holes_detected), key = num_holes_detected.count)
    index_of_most_frequent_answer = num_holes_detected.index(most_frequent_answer)
    
    print(num_holes_detected)
    
    holes = all_detections[index_of_most_frequent_answer]
    '''
    holes = []
    
    with open(os.path.join(working_dir,"detected_holes.pkl"), 'rb') as f:
        holes = pickle.load(f)

    enumerate_holes(holes)
    src_image = all_images[index_of_most_frequent_answer]
    save_holes_predictions_image(holes,src_image,os.path.join(hole_images_folder,"detected_holes.jpg"))
    

    detection_map = {}
    with open(os.path.join(working_dir,"detection_map.pkl"), 'rb') as f:
        detection_map = pickle.load(f)

    def is_id_in_frame(bee_id, frame_number):
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if detection["id"] == bee_id:
                    return True
        return False
            

    
    starts = {}
    ends = {}
    
    frame_number = 0
    while frame_number in detection_map:
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                bee_id = detection["id"]
                [top,left,bottom,right] = hole["bounding_box"]
                (center_x,center_y) = ((right+left)/2,(bottom+top)/2)

                if bee_id == -1:
                    continue
                #check if the bee with bee_id was already present in the previous frame
                if not is_id_in_frame(bee_id,frame_number-1):
                    if detection["class"] == 1:
                        #Bee is sitting
                        starts[bee_id] = get_hole_at_position(center_x,center_y,holes)
                    else:
                        starts[bee_id] = None
                if not is_id_in_frame(bee_id,frame_number+1):
                    if detection["class"] == 1:
                        #Bee is sitting
                        ends[bee_id] = get_hole_at_position(center_x,center_y,holes)
                    else:
                        ends[bee_id] = None    
        frame_number += 1
    
    frame_number = 0
    while frame_number in detection_map:
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                bee_id = detection["id"]
                if bee_id == -1:
                    continue
                detection["start"] = starts[bee_id]
                detection["end"] = ends[bee_id]
        frame_number += 1
        
    with open(os.path.join(working_dir,"detection_map.pkl"), 'wb') as f:
        pickle.dump(detection_map,f)

    with open(os.path.join(working_dir,"detected_holes.pkl"), 'wb') as f:
        pickle.dump(holes,f)



    
def get_hole_at_position(x,y,holes):
    for hole in holes:
        [top,left,bottom,right] = hole["bounding_box"]
        if x<right and x>left and y<bottom and y>top:
            return hole["name"]
    return None

    
    
def save_holes_predictions_image(holes,image_path,destination_path):
    
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    
    for hole_id,hole in enumerate(holes):
        [top,left,bottom,right] = hole["bounding_box"]
        top = int(top*height)
        bottom = int(bottom*height)
        left = int(left*width)
        right = int(right*width)
        rectangle_color = (0,0,255)
        
        image = cv2.rectangle(image, (left,top), (right,bottom), rectangle_color, 3)
        show_str = str(hole_id)
        if "name" in hole:
            show_str = hole["name"]
        cv2.putText(image, show_str, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, rectangle_color, 3)
    cv2.imwrite(destination_path,image)


    
def enumerate_holes(detections):
    
    right_neighbours = {}
    
    for hole_id,hole in enumerate(detections):
        [top,left,bottom,right] = hole["bounding_box"]
        (center_x,center_y) = ((right+left)/2,(bottom+top)/2)
        right_neighbours[hole_id] = None
        for hole_id2,hole2 in enumerate(detections):
            if hole_id == hole_id2:
                continue
            [top2,left2,bottom2,right2] = hole2["bounding_box"]
            (center_x2,center_y2) = ((right2+left2)/2,(bottom2+top2)/2)
            if center_y2 < bottom and center_y2 > top and center_x2 > center_x:
                #it should be in the same row and right of the current hole
                squared_distance = pow(center_x-center_x2,2) + pow(center_y-center_y2,2)
                if right_neighbours[hole_id] == None or right_neighbours[hole_id]["distance"] > squared_distance:
                    right_neighbours[hole_id] = {"right_neighbour" : hole_id2, "distance": squared_distance}
    
    right_most_holes = []
    
    for hole_id in right_neighbours.keys():
        if right_neighbours[hole_id] == None:
            [top,left,bottom,right] = detections[hole_id]["bounding_box"]
            (center_x,center_y) = ((right+left)/2,(bottom+top)/2)
            right_most_holes.append((hole_id,center_y))
    
    right_most_holes.sort(key=lambda tup: tup[1])  # sorts in place
    
    print(right_most_holes)
    
    def get_left_most_element_of_row(hole_id):
        left_neighbor = hole_id
        has_found_a_left_neighbor = True
        while has_found_a_left_neighbor:
            has_found_a_left_neighbor = False
            for hole_id2 in right_neighbours.keys():
                if right_neighbours[hole_id2] == None:
                    continue
                #print(str(hole_id2) + "/" + str(right_neighbours[hole_id2]["right_neighbour"]))

                if right_neighbours[hole_id2]["right_neighbour"] == left_neighbor:
                    left_neighbor = hole_id2
                    has_found_a_left_neighbor = True
                    break
                
        return left_neighbor

    


    for index,(hole_id,center_y) in enumerate(right_most_holes):
        letter = string.ascii_uppercase[index]
        current_element = get_left_most_element_of_row(hole_id)

        hole_column = 1

        while True:
            detections[current_element]["name"] = str(str(letter) + str(hole_column))
            hole_column += 1
            if right_neighbours[current_element] == None:
                break
            else:
                current_element = right_neighbours[current_element]["right_neighbour"]
            
        
                
    
    
            
    
    
    

        
        
