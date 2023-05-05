import cv2
import json
import time
import numpy as np
import argparse
import os
import urllib.request
from collections import OrderedDict

def load_yolo_model(model_config, model_weights):
    net = cv2.dnn.readNet(model_config, model_weights)

    # Check if GPU acceleration is supported and enable it
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("Enabling GPU acceleration")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    output_layers = net.getUnconnectedOutLayersNames()
    return net, output_layers


def load_classes(model_classes):
    with open(model_classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def detect_objects(frame, net, output_layers, confidence_threshold):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    return indexes, boxes, class_ids, confidences

def draw_bounding_boxes(frame, indexes, boxes, class_ids, confidences, classes):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {round(confidence * 100)}%", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def generate_json(indexes, boxes, class_ids, confidences, classes):
    detected_objects = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            timestamp = time.time()

            detected_object = OrderedDict([
                ('label', label),
                ('bbox', {'x': x, 'y': y, 'w': w, 'h': h}),
                ('confidence', confidence),
                ('timestamp', timestamp)
            ])
            detected_objects.append(detected_object)

    return json.dumps(detected_objects)


def download_file(url, local_path):
    with urllib.request.urlopen(url) as response, open(local_path, 'wb') as out_file:
        out_file.write(response.read())

def get_yolo():
    model_config = 'yolov4.cfg'
    model_weights = 'yolov4.weights'
    model_classes = 'coco.names'

    if not os.path.exists(model_config):
        print(f"Downloading {model_config} ...")
        download_file('https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4.cfg', model_config)

    if not os.path.exists(model_weights):
        print(f"Downloading {model_weights} ...")
        download_file('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights', model_weights)

    if not os.path.exists(model_classes):
        print(f"Downloading {model_classes} ...")
        download_file('https://github.com/AlexeyAB/darknet/raw/master/data/coco.names', model_classes)

    return model_config, model_weights, model_classes


def process_video(video_source, net, output_layers, classes, confidence_threshold, print_json=False, display_video=False):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break

            indexes, boxes, class_ids, confidences = detect_objects(frame, net, output_layers, confidence_threshold)
            annotated_frame = draw_bounding_boxes(frame, indexes, boxes, class_ids, confidences, classes)
            json_output = generate_json(indexes, boxes, class_ids, confidences, classes)

            if display_video:
                cv2.imshow('Video', annotated_frame)

            if print_json:
                print(json_output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("Interrupted by user")
            break

    cap.release()
    cv2.destroyAllWindows()


def run_yolo_on_video(video_source, confidence_threshold, print_json=False, display_video=False):
    model_config, model_weights, model_classes = get_yolo()

    net, output_layers = load_yolo_model(model_config, model_weights)
    classes = load_classes(model_classes)

    process_video(video_source, net, output_layers, classes, confidence_threshold, print_json, display_video)
    del net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input video stream (RTSP/UDP link, file path, or device path)', required=True)
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Confidence threshold for object detection (default: 0.5)')
    parser.add_argument('-p', '--print', action='store_true', help='Print JSON messages for detected objects')
    parser.add_argument('-d', '--display', action='store_true', help='Display the video with bounding boxes')
    args = parser.parse_args()

    run_yolo_on_video(args.input, args.confidence, args.print, args.display)

