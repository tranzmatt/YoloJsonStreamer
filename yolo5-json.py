import os
import torch
import cv2
import json
import time
import numpy as np
import argparse
import urllib.request
from collections import OrderedDict
from PIL import Image
import yaml

from yolov5.models.yolo import Model

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'protocol_whitelist;file,rtp,udp,rtsp'

def load_yolo_model(model_weights, confidence_threshold):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = confidence_threshold
    return model

def load_classes(model_classes):
    with open(model_classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def detect_objects(frame, model):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(img)
    return results

def draw_bounding_boxes(frame, results, classes):
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = classes[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {round(conf.item() * 100)}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def generate_json(results, classes):
    detected_objects = []

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = classes[int(cls)]
        timestamp = time.time()

        detected_object = OrderedDict([
            ('label', label),
            ('bbox', {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}),
            ('confidence', conf.item()),
            ('timestamp', timestamp)
        ])
        detected_objects.append(detected_object)

    return json.dumps(detected_objects)

def download_file(url, local_path):
    with urllib.request.urlopen(url) as response, open(local_path, 'wb') as out_file:
        out_file.write(response.read())

def get_yolo():
    model_weights = 'models/yolov5s.pt'
    model_classes = 'models/coco.names'

    if not os.path.exists(model_weights):
        print(f"Downloading {model_weights} ...")
        os.makedirs('models', exist_ok=True)
        download_file('https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt', model_weights)

    if not os.path.exists(model_classes):
        print(f"Downloading {model_classes} ...")
        os.makedirs('models', exist_ok=True)
        download_file('https://github.com/AlexeyAB/darknet/raw/master/data/coco.names', model_classes)

    return model_weights, model_classes

def process_video(video_source, model, classes, print_json=False, display_video=False):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(frame, model)
        annotated_frame = draw_bounding_boxes(frame, results, classes)
        json_output = generate_json(results, classes)

        if display_video:
            cv2.imshow('Video', annotated_frame)

        if print_json:
            print(json_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_yolo_on_video(video_source, print_json=False, display_video=False, confidence_threshold=0.25):
    model_weights, model_classes = get_yolo()

    model = load_yolo_model(model_weights, confidence_threshold)
    classes = load_classes(model_classes)

    process_video(video_source, model, classes, print_json, display_video)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input video stream (RTSP/UDP link, file path, or device path)', required=True)
    parser.add_argument('-p', '--print', help='Print JSON output', action='store_true')
    parser.add_argument('-c', '--confidence', type=float, default=0.25, help='Confidence threshold for object detection (default: 0.25)')
    parser.add_argument('-d', '--display', help='Display video with bounding boxes', action='store_true')
    args = parser.parse_args()

    run_yolo_on_video(args.input, args.print, args.display, args.confidence)
