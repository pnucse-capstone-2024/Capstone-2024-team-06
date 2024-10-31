import argparse
import cv2
import numpy as np
from sahi.predict import get_prediction
from sahi.models.yolov8 import Yolov8DetectionModel
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker
from shapely.geometry import Polygon, Point
from .CAM_DIRECTION import CAM_DIRECTION

# Define the class IDs we only want to detect (person and vehicle)
allowed_class_ids = [0, 1, 2, 3, 5, 7]  
# 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
vehicle_class_ids = [1, 2, 3, 5, 7]
    
# Initialize YOLOv8 model
model_path = 'YOLO_models/yolov8l.pt'
model = Yolov8DetectionModel(model_path=model_path, device='cuda', confidence_threshold = 0.35)  # 0.3

args = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=30, mot20=False)  # Set this to True if you're using the MOT20 dataset

class ObjectTracking:
    def __init__(self):
        pass
       
    def start(self):
        BaseTrack._count = 0

        # frame_rate은 트래킹 버퍼사이즈랑 관계있기에, carla의 fps(or delta)와 관계없음!!
        self.byte_tracker_reward_NORTH = BYTETracker(args, frame_rate=30)  # carla의 fps(or delta)와 관계없음!!
        self.byte_tracker_reward_EAST = BYTETracker(args, frame_rate=30)
        self.byte_tracker_reward_SOUTH = BYTETracker(args, frame_rate=30)
        self.byte_tracker_reward_WEST = BYTETracker(args, frame_rate=30)

        # self.byte_trackers_state = [self.byte_tracker_state_NORTH, self.byte_tracker_state_EAST, self.byte_tracker_state_SOUTH, self.byte_tracker_state_WEST]
        self.byte_trackers_reward = [self.byte_tracker_reward_NORTH, self.byte_tracker_reward_EAST, self.byte_tracker_reward_SOUTH, self.byte_tracker_reward_WEST]


    def calculate_vehicle_count(self, frame: np.ndarray, cam_direction: CAM_DIRECTION):
        ## 1. return the total number of cars in 4 ROIs
        ## 2. return the total lasting times(frames) of cars in 4 ROIs
                
        roi_polygon_points_straight_and_right = np.array([(258, 28),(314, 29),(451, 799),(4, 801),(3, 369)])
        roi_polygon_points_left = np.array([(307, 27),(332, 26),(598, 513),(598, 799),(391, 798)])
        
        roi_polygon_straight_and_right = Polygon(roi_polygon_points_straight_and_right)
        roi_polygon_left = Polygon(roi_polygon_points_left)

        cv2.polylines(frame, [roi_polygon_points_straight_and_right], isClosed=True, color=(255, 0, 0), thickness = 2)
        cv2.putText(frame, f'ROI:SR', (226, 82 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1) # {score:.2f}

        cv2.polylines(frame, [roi_polygon_points_left], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.putText(frame, f'ROI:L', (312, 80 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 0), 1) # {score:.2f}        

        result = get_prediction(frame, model)

        detections0 = result.to_coco_annotations()

        vehicle_count_straight_and_right = 0
        vehicle_count_left = 0
        total_vehicle_count = 0

        for d in detections0:
            # Extract bounding box coordinates (xmin, ymin, width, height)
            xmin, ymin, width, height = d['bbox']
            xmax, ymax = xmin + width, ymin + height
            category_id = d['category_id']
            score = d['score']

            # Check if the center of the bounding box is inside the ROI polygon
            bbox_center = Point((xmin + xmax) / 2, (ymin + ymax) / 2)

            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))

            if category_id in allowed_class_ids:
                if roi_polygon_straight_and_right.contains(bbox_center):
                    vehicle_count_straight_and_right += 1
                    # yolo detection 결과를 노란색 박스로 그리기
                    cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, f'{category_id}({score:.2f})', (int(xmin), int(ymin - 3)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1) # {score:.2f}
                elif roi_polygon_left.contains(bbox_center):
                    vehicle_count_left += 1 
                    # yolo detection 결과를 노란색 박스로 그리기
                    cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, f'{category_id}({score:.2f})', (int(xmin), int(ymin - 3)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1) # {score:.2f}
                else:
                    # yolo detection 결과를 노란색 박스로 그리기
                    cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(100, 255, 255), thickness=1)
                    cv2.putText(frame, f'{category_id}({score:.2f})', (int(xmin), int(ymin - 3)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 255, 255), 1) # {score:.2f}
                    


        total_vehicle_count = vehicle_count_straight_and_right + vehicle_count_left
        cv2.putText(frame, f"{cam_direction} : {total_vehicle_count} vehicles", (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"sum_cnt_SR: {vehicle_count_straight_and_right}, sum_cnt_L: {vehicle_count_left}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (128, 255, 128), 1)

        
        resized_image = cv2.resize(frame, (522, 696))  # (522, 696)
        return total_vehicle_count, vehicle_count_straight_and_right, vehicle_count_left, resized_image


    def calculate_lasted_frames(self, frame: np.ndarray, cam_direction: CAM_DIRECTION, frame_interval: int, currentAction: int, yellowFlag: bool, prevAction: int):

        roi_polygon_points_whole_lanes = np.array([(229, 21),(358, 21),(595, 595),(597, 797),(2, 797),(2, 462)])        
        roi_polygon_whole_lanes = Polygon(roi_polygon_points_whole_lanes)

        roi_polygon_points_lane1 = np.array([(315, 21),(353, 21),(598, 580),(598, 798),(398, 799)])     ## 상단 21픽셀   
        roi_polygon_lane1 = Polygon(roi_polygon_points_lane1)  

        roi_polygon_points_lane2 = np.array([(281, 21),(315, 21),(399, 799),(171, 798)])     ## 상단 21픽셀   
        roi_polygon_lane2 = Polygon(roi_polygon_points_lane2)

        roi_polygon_points_lane3 = np.array([(234, 21),(282, 21),(172, 799),(1, 798),(2, 437)])     ## 상단 21픽셀   
        roi_polygon_lane3 = Polygon(roi_polygon_points_lane3)



        # Draw the ROI on the frame (optional, for visualization)
        cv2.polylines(frame, [roi_polygon_points_lane1], isClosed=True, color=(255, 0, 0), thickness = 2)
        cv2.polylines(frame, [roi_polygon_points_lane2], isClosed=True, color=(0, 255, 0), thickness = 2)
        cv2.polylines(frame, [roi_polygon_points_lane3], isClosed=True, color=(0, 0, 255), thickness = 2)

        # cv2 uses (B,G,R), while numpy uses (R,G,B)
        R = (0,0,255)
        Y = (0,198,255)
        G = (0,181,82)

        traffic_sig_color_NS_pattern = [(G,G,R),(R,R,G),(R,R,R),(R,R,R),(R,R,R)]
        traffic_sig_color_EW_pattern = [(R,R,R),(R,R,R),(G,G,R),(R,R,G),(R,R,R)]
        

        if (cam_direction == CAM_DIRECTION.NORTH or cam_direction == CAM_DIRECTION.SOUTH):
            cv2.rectangle(frame, (1, 764), (141, 799), traffic_sig_color_NS_pattern[currentAction][0], -1)
            cv2.rectangle(frame, (142, 764), (419, 799), traffic_sig_color_NS_pattern[currentAction][1], -1)
            cv2.rectangle(frame, (420, 764), (598, 799), traffic_sig_color_NS_pattern[currentAction][2], -1)
        elif (cam_direction == CAM_DIRECTION.EAST or cam_direction == CAM_DIRECTION.WEST):
            cv2.rectangle(frame, (1, 764), (141, 799), traffic_sig_color_EW_pattern[currentAction][0], -1)
            cv2.rectangle(frame, (142, 764), (419, 799), traffic_sig_color_EW_pattern[currentAction][1], -1)
            cv2.rectangle(frame, (420, 764), (598, 799), traffic_sig_color_EW_pattern[currentAction][2], -1)


        if yellowFlag==True:
            if (prevAction == 0 and (cam_direction == CAM_DIRECTION.NORTH or cam_direction == CAM_DIRECTION.SOUTH)):
                cv2.rectangle(frame, (1, 764), (141, 799), Y, -1)
                cv2.rectangle(frame, (142, 764), (419, 799), Y, -1)
                # cv2.rectangle(frame, (420, 764), (598, 799), Y, -1)
            elif (prevAction == 1 and (cam_direction == CAM_DIRECTION.NORTH or cam_direction == CAM_DIRECTION.SOUTH)):
                # cv2.rectangle(frame, (1, 764), (141, 799), Y, -1)
                # cv2.rectangle(frame, (142, 764), (419, 799), Y, -1)
                cv2.rectangle(frame, (420, 764), (598, 799), Y, -1)
            elif (prevAction == 2 and (cam_direction == CAM_DIRECTION.EAST or cam_direction == CAM_DIRECTION.WEST)):
                cv2.rectangle(frame, (1, 764), (141, 799), Y, -1)
                cv2.rectangle(frame, (142, 764), (419, 799), Y, -1)
                # cv2.rectangle(frame, (420, 764), (598, 799), Y, -1)
            elif (prevAction == 3 and (cam_direction == CAM_DIRECTION.EAST or cam_direction == CAM_DIRECTION.WEST)):
                # cv2.rectangle(frame, (1, 764), (141, 799), Y, -1)
                # cv2.rectangle(frame, (142, 764), (419, 799), Y, -1)
                cv2.rectangle(frame, (420, 764), (598, 799), Y, -1)

        result = get_prediction(frame, model)

        detections = result.to_coco_annotations()        
        
        # Filter detections based on whether they are inside the ROI
        detection_results = []        
        for object in detections:
            # Extract bounding box coordinates (xmin, ymin, width, height)
            xmin, ymin, width, height = object['bbox']
            xmax, ymax = xmin + width, ymin + height

            # Check if the center of the bounding box is inside the ROI polygon
            bbox_center = Point((xmin + xmax) / 2, (ymin + ymax) / 2)
            if roi_polygon_whole_lanes.contains(bbox_center) and (object['category_id'] in allowed_class_ids):
                detection_results.append([xmin, ymin, xmax, ymax, object['score'], object['category_id']])
                
                # Convert bounding box coordinates to integers
                pt1 = (int(xmin) + 10, int(ymin) + 10)
                pt2 = (int(xmax) - 10, int(ymax) - 10)

                # # yolo detection 결과를 노란색 박스로 그리기
                cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(100, 255, 255), thickness=1)
                cv2.putText(frame, '', (pt2[0], pt2[1] + 3), cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 255, 255), 1)
                
        cv2.putText(frame, f"{cam_direction.name}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)

        detection_results = np.array(detection_results)  # Convert list to NumPy array

        online_targets = []
        # Check the shape of detection_results
        if detection_results.size > 0:
            # Apply ByteTrack for tracking only within ROI
            online_targets = self.byte_trackers_reward[cam_direction.value].update(
                output_results=detection_results, 
                img_info=[frame.shape[0], frame.shape[1]], 
                img_size=[frame.shape[0], frame.shape[1]]
            )

        sum_lasted_frames_by_lanes = [0,0,0]
        
        # Loop through tracked objects and draw bounding boxes with track IDs
        for track in online_targets:
            track_id = track.track_id
            score = track.score
            bbox = [int(i) for i in track.tlbr] # (top, left, bottom, right)  # Convert to int for drawing
            bbox_center_x, bbox_center_y  = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

            ############################################################################
            lasted_frames = ( track.end_frame - track.start_frame ) * frame_interval + 1
            ############################################################################

            # Check if the center of the bounding box is inside the ROI polygon
            bbox_center = Point(bbox_center_x, bbox_center_y)

            # 교차로 진입하는 방향에서 봤을때 가장 왼쪽 차선(1차선) : lane1, 중앙차선(2차선) : lane2, 가장 우측 차선(3차선) : lane3
            if roi_polygon_lane1.contains(bbox_center):
                sum_lasted_frames_by_lanes[0] += lasted_frames
            elif roi_polygon_lane2.contains(bbox_center):
                sum_lasted_frames_by_lanes[1] += lasted_frames
            elif roi_polygon_lane3.contains(bbox_center):
                sum_lasted_frames_by_lanes[2] += lasted_frames

            # Draw bounding box and track ID
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 170), 2)
            text_size, _ = cv2.getTextSize('ID{track_id}({lasted_frames}fr)', cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            textlen = len('ID{track_id}({lasted_frames}fr)')/10
            cv2.rectangle(frame, (bbox[0],bbox[1]), (int(bbox[0]+textlen*40-12), int(bbox[1]-text_size[1]*1.2)), (0,0,0,0.5), -1)

            cv2.putText(frame, f'ID{track_id}({lasted_frames}fr)', (bbox[0], bbox[1] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1) # {score:.2f}

        cv2.putText(frame, f"{sum_lasted_frames_by_lanes[2]}fr", (11, 765), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"{sum_lasted_frames_by_lanes[1]}fr", (226, 765), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"{sum_lasted_frames_by_lanes[0]}fr", (491, 765), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        resized_image = cv2.resize(frame, (513, 684))  # resize to fit the display monitor

        return sum_lasted_frames_by_lanes, resized_image