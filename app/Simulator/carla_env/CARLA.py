import glob
import os
import sys
import time
import torch
import torchvision.transforms.functional as F
import numpy as np

from .CAM_DIRECTION import CAM_DIRECTION
from .object_detection import *

try:
    sys.path.append(glob.glob('C:/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import logging
import numpy as np
from numpy import random

import cv2

GLOBAL_TICK_LIMIT = 3360   ## 한 에피소드당 tick수

MAX_VEHICLE_NUM = 400

RANDOM_SPAWN_PROBABILITY = 0.1
RANDOM_SPAWN_TICK = 80

IM_WIDTH = 600
IM_HEIGHT = 800

class Env:
    def __init__(self):

        # initialize client, traffic_manager, command setting

        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.synchronous_master = True        
        self.settings.fixed_delta_seconds = 1/5
        self.settings.no_rendering_mode = False
        self.world.apply_settings(self.settings)
        
        self.traffic_manager = self.client.get_trafficmanager() # 8000
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(3)
        
        self.traffic_manager.set_respawn_dormant_vehicles(False)
        
        self.SpawnActor = carla.command.SpawnActor
        self.SetAutopilot = carla.command.SetAutopilot
        self.FutureActor = carla.command.FutureActor

        # initialize variables

        ## initialize target(traffic light actor) list

        self.target_actor_lists = [
            [1, 2, 4, 5, 6, 7],
            [0, 3],
            [10, 11, 12, 13, 14, 15],
            [8, 9]
        ]
        
        ## initialize camera position as transform

        self.camera_positions = [
            [-15, 5, 9, -50, 180, 0],  # Camera 1
            [5, 15, 9, -50, 90, 0],  # Camera 2
            [15, -5, 9, -50, 0, 0],  # Camera 3
            [-5, -15, 9, -50, -90, 0]   # Camera 4
        ]

        ## initialize shock flag
        self.shock = True
        
        # get blueprints

        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_blueprints = []
        self.vehicle_blueprints = self.get_actor_blueprints(self.world, 'vehicle.*', 'All')
        self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        self.vehicle_blueprints = [x for x in self.vehicle_blueprints if x.get_attribute('base_type') == 'car']
        self.vehicle_blueprints = sorted(self.vehicle_blueprints, key=lambda bp: bp.id)
        
        # get spawn_points

        self.spawn_points = self.world.get_map().get_spawn_points()
    
        self.number_of_spawn_points = len(self.spawn_points)

        # set probability for each directions

        self.spawn_probability_north_sr = RANDOM_SPAWN_PROBABILITY
        self.spawn_probability_north_l = RANDOM_SPAWN_PROBABILITY
        self.spawn_probability_south_sr = RANDOM_SPAWN_PROBABILITY
        self.spawn_probability_south_l = RANDOM_SPAWN_PROBABILITY
        self.spawn_probability_east_sr = RANDOM_SPAWN_PROBABILITY
        self.spawn_probability_east_l = RANDOM_SPAWN_PROBABILITY
        self.spawn_probability_west_sr = RANDOM_SPAWN_PROBABILITY
        self.spawn_probability_west_l = RANDOM_SPAWN_PROBABILITY

        self.spawn_probability_list = [RANDOM_SPAWN_PROBABILITY] * 8
        
        self.random_spawn_tick_each_spawn_points = [random.random_integers(0, RANDOM_SPAWN_TICK) for x in range(self.number_of_spawn_points)]
        
        ## initialize and spawn camera actors

        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT

        self.cameras = []
        self.camera_images = [None, None, None, None]
        
        for i, pos in enumerate(self.camera_positions):
            cam_bp = self.blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x", f"{self.im_width}")
            cam_bp.set_attribute("image_size_y", f"{self.im_height}")
            cam_bp.set_attribute("fov", "70")
            
            transform = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                                        carla.Rotation(pitch=pos[3], yaw=pos[4], roll=pos[5]))
            camera = self.world.spawn_actor(cam_bp, transform)
            self.cameras.append(camera)
            camera.listen(lambda data, idx=i: self.process_image(data, idx))
        
    def main(self):
        # apply world setting for simulation
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.synchronous_master = True
        self.settings.no_rendering_mode = False
        self.world.apply_settings(self.settings)
        
        self.vehicles_list = []
        self.list_traffic_light_actor = []
        
        self.elapsed_global_tick = 0
        self.camera_images = [None, None, None, None]

        self.global_tick_limit = GLOBAL_TICK_LIMIT

        ## batch for spawn vehicles
        self.batch = []

        print(self.spawn_probability_list)

        self.random_spawn_tick_each_spawn_points = [random.random_integers(0, RANDOM_SPAWN_TICK) for x in range(self.number_of_spawn_points)]

        print("\t\t>> getting Traffic Light Actors...")

        self.get_traffic_light_actor()
        
        print("\t\t>> succesful..")
        
    def reset(self):
        print(">> clean up episode..")
        # apply world setting for clean up
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = False
        self.settings.no_rendering_mode = False
        self.world.apply_settings(self.settings)
        
        # destroy all actors
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.vehicles_list = []
        self.list_traffic_light_actor = []
        self.camera_images = [None, None, None, None]
        self.batch = []
        
        time.sleep(0.5)
    
    # image process (raw_date --> (IM_HEIGHT, IM_WIDTH) sized RGB image)
    def process_image(self, image, camera_index):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        self.camera_images[camera_index] = i3
    
    # get all images from camera
    def get_camera_images(self):
        return self.camera_images
    
    # get all images as array
    def get_images_array(self):
        img_array = []
        images = self.get_camera_images()
        
        idx = 0
        for img in images:
            while img is None:
                time.sleep(0.0001)

            if img is not None:
                img_array.append(img)
                idx += 1

        img_array = np.uint8(img_array)

        return img_array
    
    # get filtered actor blueprints
    def get_actor_blueprints(self, world, filter, generation):
        self.bps = world.get_blueprint_library().filter(filter)

        if generation.lower() == "all":
            return self.bps

        if len(self.bps) == 1:
            return self.bps

        try:
            self.int_generation = int(generation)
            if self.int_generation in [1, 2, 3]:
                self.bps = [x for x in self.bps if int(x.get_attribute('generation')) == self.int_generation]
                return self.bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []

    #####################
    #   Traffic Light   #
    #####################
    
    def get_traffic_light_actor(self):
        self.list_actor = self.world.get_actors()
        for actor_ in self.list_actor:
            if isinstance(actor_, carla.TrafficLight) and actor_.type_id == "traffic.traffic_light":
                self.list_traffic_light_actor.append(actor_)

    def set_traffic_light_green(self, target_actor):
        target_actor.set_state(carla.TrafficLightState.Green)

    def set_traffic_light_red(self, target_actor):
        target_actor.set_state(carla.TrafficLightState.Red)

    def set_traffic_light_group_green(self, group: int):
        for i in self.target_actor_lists[group]:
            self.set_traffic_light_green(self.list_traffic_light_actor[i])

    def set_traffic_light_group_red(self, target_actors):
        for i in target_actors:
            self.set_traffic_light_red(self.list_traffic_light_actor[i])

    def set_traffic_light_all_red(self):
        for actor_ in self.list_actor:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_state(carla.TrafficLightState.Red)
                actor_.set_red_time(1000.0)

    ##############
    #    Tick    #
    ##############

    def Tick(self, tick_num):
        elapsed_tick = 0
        
        while elapsed_tick < tick_num:
            print(f"\t\t>> tick : {elapsed_tick}\t\t>> global_tick : {self.elapsed_global_tick}")
            self.world.tick()            
            
            # spawn rule
            if (self.elapsed_global_tick) % 1120 == 0 and self.shock:
                self.spawn_probability_list[random.choice(range(8))] = 1

                (self.spawn_probability_north_sr,
                self.spawn_probability_north_l,
                self.spawn_probability_south_sr,
                self.spawn_probability_south_l,
                self.spawn_probability_east_sr,
                self.spawn_probability_east_l,
                self.spawn_probability_west_sr,
                self.spawn_probability_west_l) = self.spawn_probability_list

                self.shock = False

                print(self.spawn_probability_list)

            elif (self.elapsed_global_tick) % 560 == 0 and not self.shock:
                self.spawn_probability_list = [RANDOM_SPAWN_PROBABILITY] * 8

                (self.spawn_probability_north_sr,
                self.spawn_probability_north_l,
                self.spawn_probability_south_sr,
                self.spawn_probability_south_l,
                self.spawn_probability_east_sr,
                self.spawn_probability_east_l,
                self.spawn_probability_west_sr,
                self.spawn_probability_west_l) = self.spawn_probability_list
                
                self.shock = True

                print(self.spawn_probability_list)

            for i, spawn_tick in enumerate(self.random_spawn_tick_each_spawn_points):
                if self.elapsed_global_tick == spawn_tick:

                    if self.spawn_probability_north_sr > random.random() and i in (11, 0):
                        self.spawn_random(i)

                    if self.spawn_probability_north_l > random.random() and i == 1:
                        self.spawn_random(i)

                    if self.spawn_probability_south_sr > random.random() and i in (7, 6):
                        self.spawn_random(i)

                    if self.spawn_probability_south_l > random.random() and i == 5:
                        self.spawn_random(i)
                    
                    if self.spawn_probability_east_sr > random.random() and i in (8, 9):
                        self.spawn_random(i)

                    if self.spawn_probability_east_l > random.random() and i == 10:
                        self.spawn_random(i)

                    if self.spawn_probability_west_sr > random.random() and i in (4, 3):
                        self.spawn_random(i)

                    if self.spawn_probability_west_l > random.random() and i == 2:
                        self.spawn_random(i)

                    self.random_spawn_tick_each_spawn_points[i] += RANDOM_SPAWN_TICK

            self.elapsed_global_tick += 1
            elapsed_tick += 1

    def spawn_random(self, i):
        self.batch = []

        if len(self.vehicles_list) >= MAX_VEHICLE_NUM:
            print(f"Maximum number of vehicles ({MAX_VEHICLE_NUM}) already spawned.")
            return

        self.vehicle_blueprint = random.choice(self.vehicle_blueprints)

        if self.vehicle_blueprint.has_attribute('color'):
            self.color = random.choice(self.vehicle_blueprint.get_attribute('color').recommended_values)
            self.vehicle_blueprint.set_attribute('color', self.color)

        self.vehicle_blueprint.set_attribute('role_name', 'autopilot')

        self.batch.append(
            self.SpawnActor(self.vehicle_blueprint, self.spawn_points[i])
                .then(self.SetAutopilot(self.FutureActor, True, self.traffic_manager.get_port()))
        )

        self.response = self.client.apply_batch_sync(self.batch, self.synchronous_master)

        if self.response[0].error:
            logging.error(self.response[0].error)
        else:
            self.vehicles_list.append(self.response[0].actor_id)

        print(f"Spawned vehicle(s), Total: {len(self.vehicles_list)} vehicles.")

class CARLA:
    def __init__(self):

        self.a = 1.0004
        self.lastAction = -1
        self.env = Env()
        self.ObjectTracking = ObjectTracking()
        self.sum_lasted_frames_by_lanes_whole = []
        self.currentAction = 0

    def start(self):
        self.env.main()

        self.lastAction = -1
        self.currentAction = 0

        self.ObjectTracking.start()
        self.sum_lasted_frames_by_lanes_whole = []


        print(f"\t\t\t>> start.. <<")
        self.env.world.tick(200)
        self.env.set_traffic_light_all_red()

        state = self.get_state()
        return state

    def step(self, action):

        if self.lastAction == -1:
            pass
        elif self.lastAction != action:
            self.env.set_traffic_light_all_red()  ## "노란불" 역할
            for i in range(20):
                self.env.Tick(4)
                self.update_traffic_status(yellowFlag=True)

        self.currentAction = action

        print()

        if action == 0:
            self.env.set_traffic_light_group_green(0)

        elif action == 1:
            self.env.set_traffic_light_group_green(1)

        elif action == 2:
            self.env.set_traffic_light_group_green(2)

        elif action == 3:
            self.env.set_traffic_light_group_green(3)

        print()

        next_state = self.get_state()
        reward = self.get_reward()
        done = self.is_done()
        
        self.lastAction = action

        return next_state, reward, done
    
    def get_state(self):

        image_tensors = []

        # 기본 신호 길이 틱 진행
        for i in range(10):
            self.env.Tick(4)

            self.update_traffic_status(yellowFlag=False)

        # get_image (4 * H * W)
        image_array = self.env.get_images_array()

        for image in image_array:

            img_tensor = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)
            img_tensor = F.resize(img_tensor, (256, 192))
            img_tensor = F.rgb_to_grayscale(img_tensor)
            image_tensors.append(img_tensor)

        top_row = torch.cat((image_tensors[0], image_tensors[1]), dim=2) 
        bottom_row = torch.cat((image_tensors[2], image_tensors[3]), dim=2) 
        state = torch.cat((top_row, bottom_row), dim=1)
        return state
    
    def update_traffic_status(self, yellowFlag):
        image_array = []
        image_array = self.env.get_images_array()         

        frames_for_display_list = []

        i = 0
        temp = []
        self.sum_lasted_frames_by_lanes_whole = []

        for cam_direction in CAM_DIRECTION:
            sum_lasted_frames_by_lanes_partial, frame = self.ObjectTracking.calculate_lasted_frames(image_array[i], cam_direction=cam_direction, frame_interval=4, currentAction = self.currentAction, yellowFlag=yellowFlag, prevAction=self.lastAction)  
            frames_for_display_list.append(frame)
            temp.extend(sum_lasted_frames_by_lanes_partial)
            i += 1

        self.sum_lasted_frames_by_lanes_whole = temp

        top_row = np.hstack((frames_for_display_list[0], frames_for_display_list[1]))
        bottom_row = np.hstack((frames_for_display_list[2], frames_for_display_list[3]))
        combined_image = np.vstack((top_row, bottom_row))

        # # Show the combined image in one window
        cv2.imshow('4 Directions View', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #out.release()
            cv2.destroyAllWindows()
            

    def get_reward(self):

        # reward method #1 : get the total sum of lasted frames
        sum = 0
        for lasted_frames in self.sum_lasted_frames_by_lanes_whole:
            sum += lasted_frames
        total_sum_lasted_frames = sum 

        reward = -1 * total_sum_lasted_frames


        # reward method #2 : get exponential sum of lasted frames by each traffic directions

        # lasted_frames_NS_SR = self.sum_lasted_frames_by_lanes_whole[1] + self.sum_lasted_frames_by_lanes_whole[2] +self.sum_lasted_frames_by_lanes_whole[7] + self.sum_lasted_frames_by_lanes_whole[8]
        # lasted_frames_NS_L= self.sum_lasted_frames_by_lanes_whole[0] + self.sum_lasted_frames_by_lanes_whole[6]
        # lasted_frames_EW_SR= self.sum_lasted_frames_by_lanes_whole[4] + self.sum_lasted_frames_by_lanes_whole[5] +self.sum_lasted_frames_by_lanes_whole[10] + self.sum_lasted_frames_by_lanes_whole[11]
        # lasted_frames_EW_L= self.sum_lasted_frames_by_lanes_whole[3] + self.sum_lasted_frames_by_lanes_whole[9]
        # reward = -((self.a ** lasted_frames_NS_SR) + (self.a ** lasted_frames_NS_L) + (self.a ** lasted_frames_EW_SR) + (self.a ** lasted_frames_EW_L))

        return reward

    def get_reward_count(self):
        image_array = []
        image_array = self.env.get_images_array()         

        vehicle_count_NS_SR = 0
        vehicle_count_NS_L = 0
        vehicle_count_EW_SR = 0
        vehicle_count_EW_L = 0

        frames_for_display_list = []

        i = 0
        for cam_direction in CAM_DIRECTION:
            _, vehicle_count_straight_and_right, vehicle_count_left, frame = self.ObjectTracking.calculate_vehicle_count(image_array[i], cam_direction=cam_direction)

            if cam_direction == CAM_DIRECTION.NORTH or cam_direction == CAM_DIRECTION.SOUTH:
                vehicle_count_NS_SR += vehicle_count_straight_and_right
                vehicle_count_NS_L += vehicle_count_left
            elif cam_direction == CAM_DIRECTION.WEST or cam_direction == CAM_DIRECTION.EAST:
                vehicle_count_EW_SR += vehicle_count_straight_and_right
                vehicle_count_EW_L += vehicle_count_left                

            frames_for_display_list.append(frame)
            i += 1

        top_row = np.hstack((frames_for_display_list[0], frames_for_display_list[1]))
        bottom_row = np.hstack((frames_for_display_list[2], frames_for_display_list[3]))
        combined_image = np.vstack((top_row, bottom_row))

        # # Show the combined image in one window
        cv2.imshow('4 Directions View', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #out.release()
            cv2.destroyAllWindows()

        reward = -((self.a ** vehicle_count_NS_SR) + (self.a ** vehicle_count_NS_L) + (self.a ** vehicle_count_EW_SR) + (self.a ** vehicle_count_EW_L))
        # reward = -((vehicle_count_NS_SR) + (vehicle_count_NS_L) + (vehicle_count_EW_SR) + (vehicle_count_EW_L))

        return reward
        
    
    def is_done(self):
        if self.env.elapsed_global_tick > self.env.global_tick_limit:
            return True
        else:
            return False
    
    def reset(self):
        self.env.reset()
