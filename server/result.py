import imutils
import mediapipe as mp
from torchvision import datasets, models, transforms
import torch
from PIL import Image, ImageOps
import numpy as np
import dlib
import cv2
import numpy as np

# 얼굴형 분류해서 return 에는 얼굴형 인덱스가 출력될 것임!
def faceline(image_front, PATH):
    class_names = ["각진형", "계란형 ", "둥근형", "마름모형", "하트형"]

    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cpu")

    # 모델 초기화
    model = torch.load(PATH, map_location=torch.device('cpu'))
    # 모델 불러오기
    # 드롭아웃 및 배치 정규화를 평가 모드로 설정
    model.eval()
    # 이미지 불러오기
    image2 = Image.fromarray(image_front)
    image2 = transforms_test(image2).unsqueeze(0).to(device)

    # 불러온 이미지를 얼굴형 분류 모델에 집어넣기
    with torch.no_grad():
        outputs = model(image2)
        _, preds = torch.max(outputs, 1)
        num = preds[0].tolist()

    return num

def facedetection(image_front):
    gray = cv2.cvtColor(image_front, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for face in rects:
        shape = predictor(gray, face)

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[ALL]):
            pt_pos = (pt[0], pt[1])
            cv2.circle(image_front, pt_pos, 2, (0, 255, 0), -1)

    p = (list_points[NOSE][0] + list_points[RIGHT_EYEBROW][4]) / 2 + 1
    center = (list_points[NOSE][6] - p)[1]
    low = (list_points[JAWLINE][8] - list_points[NOSE][6])[1]


    return  center, low , list_points

def hair_up (img1,list_points):
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    def get_head_mask(img):
        """
        Get the mask of the head
        Cuting  BG
        :param img: source image
        :return:   Returns the mask with the cut out BG
        """
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))  # Find faces
        if len(faces) != 0:
            x, y, w, h = faces[0]
            (x, y, w, h) = (x - 40, y - 100, w + 80, h + 200)
            rect1 = (x, y, w, h)
            cv2.grabCut(img, mask, rect1, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)  # Crop BG around the head
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # Take the mask from BG

        return mask2
    mask = get_head_mask(img1)
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnt = cnts[0]
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])

    def is_bold(pnt, hair_mask):
        """
        Check band or not
        :param pnt: The upper point of the head
        :param hair_mask: Mask with hair
        :return: True if Bald, else False
        """
        roi = hair_mask[pnt[1]:pnt[1] + 40, pnt[0] - 40:pnt[0] + 40]  # Select the rectangle under the top dot
        cnt = cv2.countNonZero(roi)  # Count the number of non-zero points in this rectangle
        # If the number of points is less than 25%, then we think that the head is bald
        if cnt < 800:
            # print("Bald human on phoro")
            return True
        else:
            # print("Not Bold")
            return False


    if is_bold(topmost, mask):
        cv2.rectangle(img1, topmost, topmost, (0, 0, 255), 5)
        print(topmost)

    # Otherwise we write that we are not bald and display the coordinates of the largest contour
    else:
        cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        len_cnts = len(cnts[0])
        max_y = 0
        for c in range(len_cnts):
            point = (cnts[0][c][0][0], cnts[0][c][0][1])

            if (point[0] >= list_points[NOSE][0][0] - 5 and point[0] <= (list_points[NOSE][0][0]) + 5):

                if (max_y < point[1]):
                    max_y = point[1]


    hair_line_point = list_points[NOSE][0][0], max_y
    up = (list_points[RIGHT_EYEBROW][4] - hair_line_point)[1]

    return up

def face_length_ratio (up, low, center):


    # 상중하안부 비율의 기준 정하기
    if (up < center):
        if (up < low):
            criteria = up
        else:
            criteria = low

    elif (center < low):
        if (center < up):
            criteria = center
        else:
            criteria = up

    elif (low < up):
        if (low < center):
            criteria = low
        else:
            criteria = center

    # 상중하안부 비율
    upper_ratio = round(abs(up / criteria), 1)
    center_ratio = round(abs(center / criteria), 1)
    lower_ratio = round(abs(low / criteria), 1)
    print(upper_ratio, center_ratio, lower_ratio)

    if (upper_ratio == center_ratio or upper_ratio == lower_ratio):
        ratio = 0  # 1:1:1
    elif (upper_ratio > center_ratio and upper_ratio > lower_ratio):
        ratio = 1  # 상안부 길 때
    elif (center_ratio > lower_ratio and center_ratio > upper_ratio):
        ratio = 2  # 중안부 길 때
    elif (lower_ratio > upper_ratio and lower_ratio > center_ratio):
        ratio = 3  # 하안부 길 때

    return ratio

# 옆광대 여부
def side_cheekbone_have(list_points):
    flag = 0
    x = (list_points[JAWLINE][1] - list_points[JAWLINE][2])[0]
    y = (list_points[JAWLINE][1] - list_points[JAWLINE][2])[1]
    # print(abs(y/x))
    if (abs(y / x) >= 7.0): flag = 1

    ''' <광대 여부 있나 확인> 
    if (flag == 1):
        print("옆광대 여부 : O")
    else:
        print("옆광대 여부 : X")
    '''
    return flag

# 앞광대 여부
def front_cheekbone_have(list_points,image_side):

    pTime = 0

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    imgRGB = cv2.cvtColor(image_side, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    cheekbone = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id,lm in enumerate(faceLms.landmark):
                ih, iw, ic = image_side.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                if (id == 116 or id == 123 or id == 147 or id == 192 or id == 213):  # test
                    # print(id,x,y)
                    #pt_pos2 = (x, y)
                    # cheekbone.append([id, pt_pos2])
                    cheekbone.append([id, x, y])
                    cv2.circle(image_side, (x, y), 2, (0, 255, 0), -1)

    m = (cheekbone[3][2] - cheekbone[1][2]) / (cheekbone[3][1] - cheekbone[1][1])  # 123번-192번 기울기

    #print("기울기", m)

    if (m <= 3.4):
        #print("앞광대 O")
        cheek_side = 1
    else:
        #print("앞광대 X")
        cheek_side = 0

    return cheek_side


# 이미지 읽어오기

#image_front = cv2.imread("dani2.jpg")
#image_side = cv2.imread("dani_is_Kayoung's_bf.jpg")


image_front_origin = cv2.imread("img2/1.jpg")
image_side_origin = cv2.imread("side/13.jpg")

# 얼굴형 분류 모델의 위치 = PATH
PATH = 'model_76.pt'

image_front = imutils.resize(image_front_origin, height=500)  # image 크기 조절
image_side = imutils.resize(image_side_origin, height=500)  # image 크기 조절


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))  # index 1, 15 = 옆광대

index = ALL

faceline_index = faceline(image_front,PATH)

center, low, list_points = facedetection(image_front)

# 얼굴 비율 구할 때 필요한 맨 위 점 (헤어 라인 점 )
up = hair_up(image_front,list_points)

''' <ratio 의미>
0 # 1:1:1
1 # 상안부 길 때
2 # 중안부 길 때
3 # 하안부 길 때
'''
ratio = face_length_ratio(up, center, low)

# 옆광대 유무 1: 있음 , 0: 없음
cheek_side = side_cheekbone_have(list_points)

# 앞광대 유무 1: 있음 , 0: 없음
cheek_front = front_cheekbone_have(list_points,image_side)


print("얼굴형_인덱스", faceline_index)
print("얼굴 비", ratio)
print("얼굴 옆광대 ", cheek_side)
print("얼굴 앞광대 ", cheek_front)