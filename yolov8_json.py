import argparse
import json
from datetime import datetime
import cv2
import paho.mqtt.client as mqtt


import torch
from yolov5.models.experimental import attempt_load
#from yolov5.utils.general import non_max_suppression, rescale_boxes
from yolov5.utils.general import non_max_suppression, xyxy2xywh

from yolov5.utils.torch_utils import select_device

def get_yolov8_model(model_name):
    valid_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

    if model_name not in valid_models:
        raise ValueError(f"Invalid model name: {model_name}")

    url = f'https://github.com/ultralytics/yolov5/releases/download/v8.0/{model_name}.pt'
    return torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=url)

def run_yolov8(input_video, model, min_confidence, device, half, mqtt_host,
               mqtt_port, mqtt_topic, show_video, print_json):
    device = select_device(device)
    model = get_yolov8_model(model)
    model.to(device).eval()
    if half:
        model.half()

    cap = cv2.VideoCapture(input_video)
    _, img0 = cap.read()

    mqtt_client = mqtt.Client()
    mqtt_client.connect(mqtt_host, mqtt_port)

    while img0 is not None:
        img = torch.from_numpy(img0).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, min_confidence)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = xyxy2xywh(det[:, :4])

                for *xyxy, conf, cls in det:
                    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                    bbox = [int(x.item()) for x in xyxy]
                    json_data = {
                        "object_type": model.names[int(cls)],
                        "confidence": conf.item(),
                        "bbox": bbox,
                        "timestamp": timestamp
                    }

                    json_string = json.dumps(json_data)

                    if print_json:
                        print(json_string)

                    mqtt_client.publish(mqtt_topic, json_string)

                    if show_video:
                        img0 = \
                            cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])),
                                          (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)

        if show_video:
            cv2.imshow('YOLOv8 Video', img0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        _, img0 = cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='0', help='input video source')
    parser.add_argument('-y', '--model', type=str, default='yolov8s', help='yolov8 model to use')
    parser.add_argument('-c', '--min_confidence', type=float, default=0.5,
                        help='minimum confidence level for publishing JSON')
    parser.add_argument('-d', '--device', type=str, default='0',
                        help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('-f', '--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('-m', '--mqtt-host', type=str, default='localhost', help='MQTT server host')
    parser.add_argument('-p', '--mqtt-port', type=int, default=1883, help='MQTT server port')
    parser.add_argument('-t', '--mqtt-topic', type=str, default='yolo_bb_message',
                        help='MQTT message topic')
    parser.add_argument('-s', '--show-video', action='store_true',
                        help='display the video with YOLO detections')
    parser.add_argument('-P', '--print-json', action='store_true',
                        help='print and send JSON strings')

    args = parser.parse_args()

    run_yolov8(args.input, args.model, args.min_confidence, args.device, args.half,
               args.mqtt_host, args.mqtt_port, args.mqtt_topic, args.show_video, args.print_json)
