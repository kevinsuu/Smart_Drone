import pygame
import numpy as np
from djitellopy import Tello
import cv2, math, time
import argparse
from read_yolo import YOLO


# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120

class FrontEnd(object):
    def adjust_tello_position(self,tello,offset_x, offset_y, z_area):

        if self.z_area >=2500 and self.z_area<=6500:
            if self.offset_y>=-200 and self.offset_y<=20:
                if self.offset_x <=150 and self.offset_x >=-150:
                    pass
                elif self.offset_x < 150:
                    self.tello.send_rc_control(0,0,0,-15)
                elif self.offset_x > -150:
                    self.tello.send_rc_control(0,0,0,15)
            elif self.offset_y == -30:
                pass
            elif self.offset_y < -200:
                self.tello.send_rc_control(0,0,10,0)
            elif self.offset_y > 50:
                self.tello.send_rc_control(0,0,-20,0)
        elif self.z_area == 0:
                pass
        elif self.z_area > 6500  :
            self.tello.send_rc_control(0,-20,0,0)
        elif self.z_area < 2000:
            self.tello.send_rc_control(0,10,0,0)
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tello video stream")

        self.screen = pygame.display.set_mode([960, 720])
        self.tello = Tello()
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.send_rc_control = False

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self):
        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
        ap = argparse.ArgumentParser()
        ap.add_argument('-d', '--device', default=0, help='Device to use')
        ap.add_argument('-s', '--size', default=416, help='Size for yolo')
        ap.add_argument('-c', '--confidence', default=0.8, help='Confidence for yolo')
        args = ap.parse_args()
        label_names=["stop","right","left"]
        yolo = YOLO("C:\\Users\\wmslab\\Desktop\\SmartDrone\\cfg\\weights\\yolov4-tiny.cfg",
         "C:\\Users\\wmslab\\Desktop\\SmartDrone\\cfg\\weights\\yolov4-tiny_final.weights", label_names)
        yolo.size = int(args.size)
        yolo.confidence = float(args.confidence)

        self.tello.connect()
        self.tello.set_speed(self.speed)
        self.tello.streamoff()
        self.tello.streamon()
    
        frame_read = self.tello.get_frame_read()
        should_stop = False
        count=0
        hand_count=0

        while not should_stop:
            count=count+1
            hand_count=hand_count+1
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame

            height = 960
            width = 720
            self.width, self.height, self.inference_time, self.results = yolo.inference(frame)
            center_x = int(height/2)
            center_y = int(width/2)
            
            cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)
            face_center_x = center_x
            face_center_y = center_y
            self.z_area = 0
            print(hand_count)
            #print (self.results)
            if (bool(self.results)) is True:
                for detection in self.results:
                    id, name, confidence, x, y, w, h = detection    
                    cx = x + (w / 2)
                    cy = y + (h / 2)
                    cv2.rectangle(frame,(x, y),(x + w, y + h),(255, 0, 0), 2)
                    face_center_x = x + int(h/2)
                    face_center_y = y + int(w/2)
                    self.z_area = w * h
        
                    if hand_count>=10:                 
                        if name =="right":
                            self.tello.send_rc_control(-30,0,0,25)
                            hand_count=0
                        elif name =="left":
                            self.tello.send_rc_control(30,0,0,-25)
                            hand_count=0
            else:
                for face in faces:
                    (x, y, w, h) = face
                    cv2.rectangle(frame,(x, y),(x + w, y + h),(255, 255, 0), 2)
                    face_center_x = x + int(h/2)
                    face_center_y = y + int(w/2)
                    self.z_area = w * h
                    cv2.circle(frame, (face_center_x, face_center_y), 10, (0, 0, 255))

            self.offset_x = face_center_x - center_x
            self.offset_y = face_center_y - center_y - 30
            
            cv2.putText(frame, f'[{self.offset_x}, {self.offset_y}, {self.z_area}]', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
             (255,255,255), 2, cv2.LINE_AA)                

            if count>=3:
                self.adjust_tello_position(self.tello,self.offset_x, self.offset_y, self.z_area)
                count=0
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)
        self.tello.end()

    def keydown(self, key):

        if key == pygame.K_b:
            print("Battery： "+str(self.tello.get_battery()))
        elif key == pygame.K_t:
            self.tello.takeoff()
            self.tello.move_up(120)
        elif key == pygame.K_l:
            not self.tello.land()
            self.send_rc_control = False
            
    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)


def main():
    frontend = FrontEnd()
    frontend.run()


if __name__ == '__main__':
    main()