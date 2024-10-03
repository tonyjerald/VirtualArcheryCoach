import cv2
import mediapipe as mp
import time
import math
from pymongo import MongoClient

# Configure MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['archeryDataBase']
collection = db['userDataCollection']

class poseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.model = model_complexity
        self.landmarks = smooth_landmarks
        self.enableSegmentation = enable_segmentation
        self.smoothSegmentaion = smooth_segmentation
        self.detection = min_detection_confidence
        self.tracking = min_tracking_confidence


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model, self.landmarks,
                                     self.enableSegmentation, self.smoothSegmentaion, self.detection,
                                     self.tracking)

    def findPose(self, img , draw=False):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                #if id and cx and cy is not None:
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (225, 0, 0), cv2.FILLED)

        return lmList

    def calculate_angle(A, B, C):
        # Vectors AB and BC
        AB = (B[0] - A[0], B[1] - A[1])
        BC = (C[0] - B[0], C[1] - B[1])

        # Dot product of AB and BC
        dot_product = AB[0] * BC[0] + AB[1] * BC[1]

        # Magnitude of AB and BC
        magnitude_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
        magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

        # Calculate the cosine of the angle
        cos_angle = dot_product / (magnitude_AB * magnitude_BC)

        # Get the angle in radians and convert to degrees
        angle_radians = math.acos(cos_angle)
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees
    def findAngle(self, img, p1, p2, p3, lmList, draw, color):

        # Get the landmarks
        x1, y1 = lmList[p1][1:]
        x2, y2 = lmList[p2][1:]
        x3, y3 = lmList[p3][1:]


        angle = poseDetector.calculate_angle((x1,y1),(x2,y2), (x3,y3))

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, color, 2)
            cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, color, 2)
            cv2.circle(img, (x3, y3), 10, color, cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, color, 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (127, 0, 255), 2)
        return angle

def main(filepath, color, draw, position_id1, position_id2, position_id3, frames_per_second):
    cap = cv2.VideoCapture(filepath)
    previous_time = 0
    detector = poseDetector()
    red = (0, 0, 255)
    blue = (135, 206, 235)
    if color=="red":
        color=red
    else:
        color=blue

    if draw=="yes":
        draw=True
    else:
        draw=False

    # Calculate delay between frames in milliseconds
    delay = int(1000 / frames_per_second)

    while cap.isOpened():
        sucess, frame = cap.read()
        if not sucess:
            # print("exit1")
            break
        else:
            img = detector.findPose(frame)
            lmList = detector.findPosition(img, draw=False)
            # 14 is elbow point
            if len(lmList) != 0:
                # print(lmList[14])
                # cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, red, cv2.FILLED)
                angle = detector.findAngle(img, position_id1, position_id2, position_id3, lmList, draw, color)
                print(angle)
            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            cv2.waitKey(delay)
            cv2.putText(img, str(int(fps)) + "FPS" , (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (225, 0, 0), 3)

            #Tranfering the edited video
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # print("exit2")

    cap.release()
    # Store file path and feedback in MongoDB
    collection.insert_one({'filepath': filepath, 'Angle': angle, 'Enter_Three_Position_IDs_To_Draw_Angle': {'position_id1': position_id1, 'position_id2': position_id2, 'position_id3': position_id3 }, 'frames_per_second': frames_per_second})


if __name__ == "__main__":
    main()