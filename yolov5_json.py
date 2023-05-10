import os
import re
import torch
import cv2
import json
import time
import argparse
import urllib.request
from collections import OrderedDict
from PIL import Image

def load_yolo_model(model=None, model_weights=None, confidence_threshold=0.5, the_torch_hub='ultralytics/yolov5'):

    the_model = None
    model_loaded = False

    # If we've provided a custom model weights file
    if model_weights:
        try:
            print(f"Checking for {model_weights}")
            # Attempt to load the model from the local file
            if os.path.isfile(model_weights):
                print(f"Local {models_weights} found, attempting to load")
                the_model = torch.load(model_weights)
                the_model.conf = confidence_threshold
                model_loaded = True
        except Exception as e:
            print(f"Error loading {model_weights} local model: {e}")

    if not model_loaded and model is not None:
        available_models = torch.hub.list(the_torch_hub)
        print(available_models)

        if model not in available_models:
            print(f"The model {model} is not available from {the_torch_hub}")
            return the_model

        try:
            print(f"Attempting to load {model} from cache")
            the_model = torch.hub.load(the_torch_hub, model, force_reload=False)
            the_model.conf = confidence_threshold
            model_loaded = True
        except Exception as e:
            print(f"Loading from cache failed {e}.  Redownloading")
            for attempt in range(3):
                try:
                    # Load the model from torch.hub and force redownload
                    the_model = torch.hub.load(the_torch_hub, model, force_reload=True)
                    the_model.conf = confidence_threshold
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Error downloading {model} from torch.hub: {e}")
                    print("Model download was interrupted or incomplete. Please try again.")

    if not model_loaded or the_model is None:
        print("Unable to load any model!")
    elif torch.cuda.is_available():
        print(f"CUDA capable device found")
        the_model.to('cuda:0')

    return the_model


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
        cv2.putText(frame, f"{label} {round(conf.item() * 100)}%", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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


def open_video_stream(input_source):
    is_rtsp = re.match(r"rtsp://", input_source)
    is_local_device = re.match(r"/dev/video\d+", input_source)
    is_other_protocol = re.match(r"(udp|http|https)://", input_source)

    if is_rtsp:
        transport_methods = ['udp', 'tcp']

        for transport in transport_methods:

            try:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'protocol_whitelist;file,rtp,udp,rtsp,tcp|rtsp_transport;{transport}|loglevel;error'
                cap = cv2.VideoCapture(input_source)
 
                # Check if the stream is opened successfully
                if cap.isOpened():
                    print(f"Connected to RTSP stream using {transport} transport method.")
                    return cap
    
            except Exception as e:
                    print(f"RTSP stream {transport} transport failed.")


        print("Failed to connect to RTSP stream using both UDP and TCP transport methods.")
        return None

    elif is_local_device or os.path.isfile(input_source) or is_other_protocol:
        cap = cv2.VideoCapture(input_source)

        if cap.isOpened():
            print("Connected to the local device, file, or other protocol.")
            return cap

        print("Failed to connect to the local device, file, or other protocol.")
        return None

    else:
        print(f"{input_source} is an invalid input source.")
        return None



def process_video(video_source, model, classes, print_json=False, display_video=False):

    cap = open_video_stream(video_source)

    while cap and cap.isOpened():
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


def run_yolo_on_video(video_source, print_json=False, display_video=False,
                      confidence_threshold=0.5, yolo_model='yolov5s', model_weights=None):

    model_classes = 'coco.names'

    if not os.path.exists(model_classes):
        print(f"Downloading {model_classes} ...")
        download_file('https://github.com/AlexeyAB/darknet/raw/master/data/coco.names', model_classes)

    model = load_yolo_model(yolo_model, model_weights, confidence_threshold)

    classes = load_classes(model_classes)

    process_video(video_source, model, classes, print_json, display_video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='Input video stream (RTSP/UDP link, file path, or device path)', required=True)
    parser.add_argument('-p', '--print', help='Print JSON output', action='store_true')
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                        help='Confidence threshold for object detection (default: 0.5)')
    parser.add_argument('-d', '--display', help='Display video with bounding boxes', action='store_true')
    parser.add_argument('-w', '--model_weights',  type=str, help='Provide custom weight name')
    parser.add_argument('-y', '--yolo_model', type=str, default='yolov5s',
                        help='YOLOv5 model to use (default: yolov5s)')
    args = parser.parse_args()

    run_yolo_on_video(args.input, args.print, args.display, args.confidence,
                      args.yolo_model, args.model_weights)
