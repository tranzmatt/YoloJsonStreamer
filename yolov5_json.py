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

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'protocol_whitelist;file,rtp,udp,rtsp,tcp'

def download_file(url, local_path):
    with urllib.request.urlopen(url) as response, open(local_path, 'wb') as out_file:
        out_file.write(response.read())

def get_yolo(yolo_model):
    models_dir = 'models'
    model_weights = f'{yolo_model}.pt'
    model_classes = 'coco.names'

    # Create the models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Construct the paths to the model_weights and model_classes files
    model_weights_path = os.path.join(models_dir, model_weights)
    model_classes_path = os.path.join(models_dir, model_classes)

    if not os.path.exists(model_weights_path):
        print(f"Downloading {model_weights} ...")
        download_file(f'https://github.com/ultralytics/yolov5/releases/download/v5.0/{yolo_model}.pt', model_weights_path)

    if not os.path.exists(model_classes_path):
        print(f"Downloading {model_classes} ...")
        download_file('https://github.com/AlexeyAB/darknet/raw/master/data/coco.names', model_classes_path)

    return model_weights_path, model_classes_path

def load_yolo_model(model_weights, confidence_threshold):
    # Extract the model type from the model weights file name
    model_type = os.path.basename(model_weights).split('.')[0]

    if os.path.isfile(model_weights):
        try:
            # Load model architecture
            print(f"Attempting to locally load {model_weights}")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights, source='local')  # local repo
            # Set confidence threshold
            model.conf = confidence_threshold
        except Exception as e:
            print(f"Error loading {model_weights} locally: {e}")
            print(f"Loading {model_weights} from torch hub with force_reload=True ...")
            model = torch.hub.load('ultralytics/yolov5', model_type, force_reload=True)
            model.conf = confidence_threshold
            torch.save(model.state_dict(), model_weights)
    else:
        print(f"{model_weights} not found locally. Loading {model_type} from torch hub ...")
        model = torch.hub.load('ultralytics/yolov5', model_type)
        model.conf = confidence_threshold
        torch.save(model.state_dict(), model_weights)

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

def run_yolo_on_video(video_source, print_json=False, display_video=False, confidence_threshold=0.5, yolo_model='yolov5s'):

    # DL model weights if not found
    model_weights, model_classes = get_yolo(yolo_model)

    # load YOLO models
    model = load_yolo_model(model_weights, confidence_threshold)

    # Load classes
    classes = load_classes(model_classes)

    process_video(video_source, model, classes, print_json, display_video)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input video stream (RTSP/UDP link, file path, or device path)', required=True)
    parser.add_argument('-p', '--print', help='Print JSON output', action='store_true')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Confidence threshold for object detection (default: 0.5)')
    parser.add_argument('-d', '--display', help='Display video with bounding boxes', action='store_true')
    parser.add_argument('-y', '--yolo_model', type=str, choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'], default='yolov5s', help='YOLOv5 model to use (default: yolov5s)')
    args = parser.parse_args()

    run_yolo_on_video(args.input, args.print, args.display, args.confidence, args.yolo_model)