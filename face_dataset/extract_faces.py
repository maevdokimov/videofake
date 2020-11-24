import cv2
from typing import Tuple
import os

BboxType = Tuple[int, int, int, int]


def adjust_bbox_size(scale: float, bbox: BboxType) -> BboxType:
    x, y, w, h = bbox
    center_x, center_y = x + w / 2, y + h / 2
    w, h = int(w * scale), int(h * scale)
    x, y = max(int(center_x - w / 2), 0), max(int(center_y - h / 2), 0)
    return x, y, w, h


def process_video(vid_path: str, xml_path: str, out_path: str):
    cap = cv2.VideoCapture(vid_path)
    face_cascade = cv2.CascadeClassifier(xml_path)

    file_cnt = len(os.listdir(out_path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for bbox in face_cascade.detectMultiScale(gray_frame):
            x, y, w, h = adjust_bbox_size(1.5, bbox)

            filename = os.path.join(out_path, f'{file_cnt}.jpg')
            file_cnt += 1
            cv2.imwrite(filename, frame[y:y+h, x:x+w])

    cap.release()


if __name__ == '__main__':
    path = '/home/maxim/python/videofake/data/face_videos'
    xml_path = '/home/maxim/python/videofake/data/cascade_xml/haarcascade_frontalface_default.xml'
    out_path = '/home/maxim/python/videofake/data/face_videos/faces'
    
    for entry in os.listdir(path):
        file_path = os.path.join(path, entry)
        if os.path.isfile(file_path):
            process_video(file_path, xml_path, out_path)
