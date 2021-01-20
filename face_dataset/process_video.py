import cv2
from pathlib import Path
import os
import argparse


def extract_frames(video_path, destination_path, output_extention='jpg'):
    if output_extention not in ['jpg', 'png']:
        raise ValueError(f'Output extention should be .jpg or .png and not {output_extention}')

    destination_path = Path(destination_path)
    if not destination_path.exists():
        raise ValueError(f'{destination_path} does not exist')
    if any(destination_path.iterdir()):
        raise ValueError(f'{destination_path} is not empty')

    cap = cv2.VideoCapture(video_path)
    count = 0
    ret, frame = cap.read()

    while ret:
        cv2.imwrite(f'{destination_path}{os.sep}{count}.{output_extention}', frame)
        count += 1
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--ext', default='jpg')
    args = parser.parse_args()

    extract_frames(args.src, args.dst, args.ext)
