import cv2
import mediapipe as mp
import math
import time

mpHands = mp.solutions.hands
# <module 'mediapipe.python.solutions.hands' from ..\\mediapipe\\python\\solutions\\hands.py

my_hands = mpHands.Hands(False, 2, 0.6, 0.5)
# <mediapipe.python.solutions.hands.Hands object at 0x00000184FDD44C18>

mpDraw = mp.solutions.drawing_utils
# ..\mediapipe\\python\\solutions\\drawing_utils.py

def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1-x2,2)) + math.sqrt(math.pow(y1-y2,2))


# landmark의 list들을 위의 이미지처럼 전역변수 이름으로 보기 쉽게 바꿔준다.
def position_data(lmlist):
    global wrist, thumb_tip, index_mcp, index_tip, middle_mcp, middle_tip, ring_tip, pinky_tip
    # 팔목 중심(0) 좌표 x,y
    wrist = (lmlist[0][0], lmlist[0][1]) 
    # 엄지손가락 끝 좌표(4) x,y
    thumb_tip = (lmlist[4][0], lmlist[4][1])
    
    # 검지손가락 시작 좌표(5) x,y
    index_mcp = (lmlist[5][0], lmlist[5][1])
    # 검지손가락 끝 좌표(8) x,y
    index_tip = (lmlist[8][0], lmlist[8][1])
    
    # 중지손가락 시작 좌표(9) x,y
    middle_mcp = (lmlist[9][0], lmlist[9][1])
    # 중지손가락 끝 좌표(12) x,y
    middle_tip = (lmlist[12][0], lmlist[12][1])
    
    # 약지손가락 끝 좌표(16) x,y
    ring_tip = (lmlist[16][0], lmlist[16][1])
    
    # 새끼손가락 끝 좌표(20) x,y
    pinky_tip = (lmlist[20][0], lmlist[20][1])

# 두 점을 line으로 그려주는 함수
def draw_line(p1, p2, size=3):
    cv2.line(frame, p1, p2, (255,50,50), size)
    cv2.line(frame, p1, p2, (255,255,255), round(size/2))

def distance(p1, p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    # 거리 공식
    length = ( (x2-x1)**2+(y1-y2)**2 )**(1.0/2)
    return length

def transparent(targetImg, x, y , size=None):
    if size is not None:
        targetImg = cv2.resize(targetImg, size)
    
    # 일단 비디오 프레임 복사하기
    newFrame = frame.copy()
    # blue, green, red, alpha로 layer 분리
    b, g, r, a = cv2.split(targetImg)
    
    # blue, green, red layer만 merge하기
    overlay_color = cv2.merge((b,g,r))
    # alpha 부분 Blur줘서 mask로 생성
    mask = cv2.medianBlur(a,1)
    over_h, over_w, _ = overlay_color.shape
    
    # 관심영역 분리하기
    roi = newFrame[y:y+over_h, x:x+over_w]
    
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask = cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
    newFrame[y:y+over_h,x:x+over_w] = cv2.add(img1_bg, img2_fg)
    
    return newFrame
    



video = cv2.VideoCapture(0)
#print(video) # <VideoCapture 0000018485E3E4D0>

video.set(3, 1280)
video.set(4, 720)

img_1 = cv2.imread('magic_circles/kaist_circle_2.png',-1)
img_2 = cv2.imread('magic_circles/kaist_circle_1.png',-1)

deg=0
year=1850
counter=0

while True:
    # video(카메라)에서 img 가져오기, success는 사용 여부
    success, frame = video.read()
    
    
    # frame 상을 뒤집기
    frame = cv2.flip(frame,1)
    # frame의 색공간 BGR -> RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # RGB 이미지를 mediapipe 손 인식 함수로 처리 결과
    results = my_hands.process(frameRGB)
    # print(results), <class 'mediapipe.python.solution_base.SolutionOutputs'>
    
    if results.multi_hand_landmarks:
        # hand landmarks
        for handlms in results.multi_hand_landmarks:
            lmList = []
            
            for id, handlm in enumerate(handlms.landmark):
                # frame의 height, width 정보
                h,w,c = frame.shape
                coor_x, coor_y = int(handlm.x*w), int(handlm.y*h) # handlm의 좌표값은 0~1로 표현
                # landmark List에 점 리스트 추가하기
                lmList.append([coor_x,coor_y])
                
                # finger point에 점찍기
                # cv2.circle(frame,(coor_x,coor_y), 6, (50,50,255),-1)
            
            # 내가 정의한 위에서 정의한 함수들
            position_data(lmList) # data -> global variable에 저장해줌
            
            # 손바닥 길이를 재보자
            palm = distance(wrist, index_mcp) # 손목에서 중지 시작까지
            i2p = distance(index_tip, pinky_tip) # 검지 끝에서 새끼 끝까지
            ratio = i2p/palm
            # 접으면 1보다 작고, 손을 펴면 1보다 크다 print(ratio)
            
            # 손의 중심 좌표 구하기
            hand_center_x = middle_mcp[0]
            hand_center_y = middle_mcp[1]
            
            
            # 접힌 경우
            if(0.2<ratio<1.2):
                draw_line(wrist, thumb_tip) # 위에서 저장한 point 사이를 이어줌
                draw_line(wrist, index_tip)
                draw_line(wrist, middle_tip)
                draw_line(wrist, pinky_tip)
                draw_line(wrist, thumb_tip)
                draw_line(thumb_tip, index_tip)
                draw_line(thumb_tip, middle_tip)
                draw_line(thumb_tip, ring_tip)
                draw_line(thumb_tip, pinky_tip)
                
                
                fname = "data/"+str(year)+".png"

                if fname!="data/.png" and year<=2100 and year>=1850:
                    data = cv2.imread(fname, cv2.IMREAD_COLOR)
                    #print(year)
                    cv2.namedWindow("ClimateChange",flags=cv2.WINDOW_FREERATIO)
                    cv2.resizeWindow("ClimateChange",1920,1080)
                    cv2.putText(data, str(year), (150,200), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, int((year-1000)/200), (0,0,0),int((year-1000)/60))
                    cv2.putText(data, "CO2: "+"{:.2f}".format((year-1850)*17.37)+"kg", (1800,200), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 4, (0,0,int((year-1850))),12)
                    cv2.imshow("ClimateChange",data) 
                
                # print(counter)
                # 년도가 감소하는 경우 : 왼손
                if 640 < hand_center_x :
                    if (year-counter) > 1850:
                        year-=counter
                    else:
                        year = 1850
                        
                # 년도가 감소하는 경우 : 오른손
                else:
                    if (year+counter) < 2100:
                        year+=counter
                    else:
                        year = 2100
                
        
            
            # 손바닥 편 경우
            if(ratio > 1.2):
                # counter
                counter = int(2**round(palm/70)/3) # 접은 경우에 편 사이즈로 update하기 위함
                print(counter)
                shield_size = 3.0
                diameter = round(palm * shield_size)
                
                x1 = round(hand_center_x - (diameter / 2))
                y1 = round(hand_center_y - (diameter / 2))
                h, w, c = frame.shape
                if x1 < 0:
                    x1 = 0
                elif x1 > w:
                    x1 = w
                if y1 < 0:
                    y1 = 0
                elif y1 > h:
                    y1 = h
                if x1 + diameter > w:
                    diameter = w - x1
                if y1 + diameter > h:
                    diameter = h - y1
                    
                shield_size = diameter, diameter
                ang_vel = 4.0
                
                deg = deg + ang_vel
                
                if deg > 360:
                    deg = 0
                
                hei, wid, col = img_1.shape
                
                center = (wid // 2, hei // 2)
                
                M1 = cv2.getRotationMatrix2D(center, round(deg), 1.0)
                M2 = cv2.getRotationMatrix2D(center, round(360 - deg), 1.0)
                
                if 640 < hand_center_x:
                    img_1 = cv2.imread('magic_circles/kaist_circle_red_2.png',-1)
                    img_2 = cv2.imread('magic_circles/kaist_circle_red_1.png',-1)
                else :
                    img_1 = cv2.imread('magic_circles/kaist_circle_2.png',-1)
                    img_2 = cv2.imread('magic_circles/kaist_circle_1.png',-1)

                    
                
                rotated1 = cv2.warpAffine(img_1, M1, (wid, hei))
                rotated2 = cv2.warpAffine(img_2, M2, (wid, hei))
                
                if (diameter != 0):
                    frame = transparent(rotated1, x1, y1, shield_size)
                    frame = transparent(rotated2, x1, y1, shield_size)
                
            
            #################################################
            # (기본함수)frame에 다가 handlms object에 landmark(점)들을 연결시켜서 보여준다.
            # mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)
        
    
    
    # 마지막 단계 frame을 image_show 해주기
    cv2.imshow("Control",frame)
    
    # key 입력으로 while 문 탈출, 1은 키보드 입력 대기시간(ms)
    if cv2.waitKey(1)>0:
        break;
    
    
    
    #results = my_hands.process(imgRGB)
    
    #if results.multi_hand_landmarks:
    #    for handLms in results.multi_hand_landmarks:
            


video.release()
# 사용했던 카메라 object 해제
cv2.destroyAllWindows()
# 현재 실행중인 창 다 닫기
