import cv2
import mediapipe as mp
import time
import math
import pyautogui

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
                
        return img
    def findPosition(self, img, handNum = 0,draw = True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            _hand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(_hand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)

                #print(id,cx,cy)

                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(0,0,0),cv2.FILLED)

        return lmList
    def fingerDistance(self,img,lmList,finger1,finger2):
        f1 = (0,0)
        f2 = (0,0)
        for id,cx,cy in lmList:
            if id==finger1:
                f1 = (cx,cy)
            if id == finger2:
                f2 = (cx,cy)
        return f1,f2,math.sqrt((f1[0]-f2[0])*(f1[0]-f2[0]) + (f1[1] -f2[1])*(f1[1] -f2[1]))


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img,0,False)
        # distanct = 0
        #print(lmList)
        f1,f2,distanct = detector.fingerDistance(img,lmList,4,8)
        cv2.line(img,f1,f2,(255,0,0),2,3)
        if distanct >= 180 and distanct <300:
            print('CROUCH')      
            pyautogui.press('down')
        elif distanct <= 50 and distanct >= 10:
            print('JUMP')
            pyautogui.press('up')
        elif distanct > 50 and distanct < 180:     
            print('RUN')

        cv2.putText(img,str(int(distanct)),(30,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0,200),3)

        cTime = time.time()
        if cTime - pTime != 0:
            fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

        cv2.imshow("Image",img)
        cv2.waitKey(1)

    

if __name__ == "__main__":
    main()