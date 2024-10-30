import torch
import sumolib
from .utils import *
from .enums import CAM_DIRECTION
import traci

class SUMO:
    def __init__(self):
        # self.config
        self.sumo_config = "Simulator\sumo_env\cross.sumocfg"
        self.sumo_binary = sumolib.checkBinary('sumo-gui')  # sumo 실행 파일 경로 설정
        self.sumo_process = None
        self.prev_action = -1    

    def start(self):
        # 시뮬레이터 환경설정 및 가동
        if self.sumo_process is not None:
            generateRandomRoutes2() 
            self.prev_action = -1
            traci.load(["-c", self.sumo_config, "--start", "--step-length", "1", "--delay", "0", "--window-size", "1000,1000"])
        else:
            generateRandomRoutes2() 
            self.prev_action = -1
            if self.sumo_process is None:
                sumo_cmd = [
                    self.sumo_binary, 
                    "-c", self.sumo_config,
                    "--start",
                    "--step-length", "1",
                    "--delay", "0",
                    "--window-size", "1000,1000"
                ]
                self.sumo_process = traci.start(sumo_cmd)

        # time.sleep(5)
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        
        state = self.get_state()
        return state
    
    def step(self, action):
        # Implement the logic to apply the action and advance the simulation
        
        tlsID = traci.trafficlight.getIDList()[0]  # 첫 번째 신호등의 ID를 가져옴(어차피 이 network에서는 교차로가 하나라, 신호등 id도 이것 하나뿐입니다)
        
        # Get the current simulation time
        current_time = traci.simulation.getTime()

        # Calculate the fractional step time (steps x step-length = seconds)
        yellow_steps = 2
        step_length = 1
        partial_time = yellow_steps * step_length
        target_time = current_time + partial_time
        
        phasesStr = ['PHASE 0(↑↓)', 'PHASE 2(↖↘)', 'PHASE 4(←→)', 'PHASE 6(↗↙)']  # ↕ ↔
        
        if action == 0: # change to phase 0                    
            if self.prev_action == 0 or self.prev_action == -1:  # if previous was phase 0
                setSig2(tlsID, 0)
            elif self.prev_action == 1:  # if previous was phase 2
                setSig2(tlsID, 3)  # yellow light after phase 2  
                traci.simulationStep(target_time)
                setSig2(tlsID, 0)
            elif self.prev_action == 2:  # if previous was phase 4
                setSig2(tlsID, 5)  # yellow light after phase 4  
                traci.simulationStep(target_time)                             
                setSig2(tlsID, 0)
            elif self.prev_action == 3:  # if previous was phase 6
                setSig2(tlsID, 7)  # yellow light after phase 6  
                traci.simulationStep(target_time)                              
                setSig2(tlsID, 0)
        elif action == 1: # change to phase 2                            
            if self.prev_action == 0:  # if previous was phase 0
                setSig2(tlsID, 1)  # yellow light after phase 0 
                traci.simulationStep(target_time)                
                setSig2(tlsID, 2)
            elif self.prev_action == 1 or self.prev_action == -1:  # if previous was phase 2
                setSig2(tlsID, 2)
            elif self.prev_action == 2:  # if previous was phase 4
                setSig2(tlsID, 5)  # yellow light after phase 4  
                traci.simulationStep(target_time)                             
                setSig2(tlsID, 2)
            elif self.prev_action == 3:  # if previous was phase 6
                setSig2(tlsID, 7)  # yellow light after phase 6  
                traci.simulationStep(target_time)                              
                setSig2(tlsID, 2)        
        elif action == 2: # change to phase 4                            
            if self.prev_action == 0:  # if previous was phase 0
                setSig2(tlsID, 1)  # yellow light after phase 0 
                traci.simulationStep(target_time)                
                setSig2(tlsID, 4)
            elif self.prev_action == 1:  # if previous was phase 2
                setSig2(tlsID, 3)  # yellow light after phase 2 
                traci.simulationStep(target_time)                 
                setSig2(tlsID, 4)
            elif self.prev_action == 2 or self.prev_action == -1:  # if previous was phase 4                             
                setSig2(tlsID, 4)
            elif self.prev_action == 3:  # if previous was phase 6
                setSig2(tlsID, 7)  # yellow light after phase 6  
                traci.simulationStep(target_time)                              
                setSig2(tlsID, 4)   
        elif action == 3: # change to phase 6                            
            if self.prev_action == 0:  # if previous was phase 0
                setSig2(tlsID, 1)  # yellow light after phase 0 
                traci.simulationStep(target_time)                
                setSig2(tlsID, 6)
            elif self.prev_action == 1:  # if previous was phase 2
                setSig2(tlsID, 3)  # yellow light after phase 2 
                traci.simulationStep(target_time)                 
                setSig2(tlsID, 6)
            elif self.prev_action == 2:  # if previous was phase 4 
                setSig2(tlsID, 5)  # yellow light after phase 4 
                traci.simulationStep(target_time)                                              
                setSig2(tlsID, 6)
            elif self.prev_action == 3 or self.prev_action == -1:  # if previous was phase 6                            
                setSig2(tlsID, 6)                                           
        
        print(f'{phasesStr[self.prev_action]} ==> {phasesStr[action]}')
        self.prev_action = action        

        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()

        next_state = self.get_state()
        
        reward = self.get_reward()
        done = self.is_done()
        return next_state, reward, done
    
    def get_state(self):
        # Implement the logic to get the current state of the environment

        # junction_id = "6"  # 교차로 ID
        # output = getSimStatebyNumbers(junction_id) 
        # 다음과 같은 배열 리턴 
        # [
        #   [ 현재신호주기, 다음신호남은시간?, 현신호지난시간? ], 
        #   [ 신호주기0길이, 신호주기1길이, 신호주기2길이, 신호주기3길이, 신호주기4길이, 신호주기5길이, 신호주기6길이, 신호주기7길이 ], 
        #   [ 0, lane0 전체 차량수, 정지 차량수, 평균 속도, 평균 가속도 ], ...(및11개 lane)..., 
        # ]  
        # state = np.array(output)
        
        state = getSimStatebyImage() 
        return state
    
    def get_reward(self):
        # Implement the logic to calculate the reward
        # total_vehicle_count = 0
        # total_sum_lasted_frame = 0
        
        # for cam_direction in CAM_DIRECTION:
        #     vehicle_count, sum_lasted_frame = caculate_reward_from_carla(cam_direction=cam_direction, frame_interval=15)
        #     total_vehicle_count += vehicle_count
        #     total_sum_lasted_frame += sum_lasted_frame
        
        # avg_lasted_frame =  total_sum_lasted_frame / total_vehicle_count
        # reward = (-1) * avg_lasted_frame
        
        reward = calculateReward()
        return reward
    
    def is_done(self):
        # 시나리오 종료 체크
        # Implement the logic to check if the episode is done
        global stop_simulation
        step = traci.simulation.getTime()
        
        if step <= 1000 and not stop_simulation:  # 2000
            return False
        else:
            return True

    def close(self):
        if self.sumo_process is not None:
            traci.close()
            self.sumo_process = None

    def reset(self):
   
        return