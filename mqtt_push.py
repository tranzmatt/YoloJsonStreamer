import os
import json
import paho-mqtt

object_detection_message = {
    "version": String,
    "messageType": "ObjectDetectionMessage",
    "data": {
        # Standard Metadata
        "deviceTime": "2023-03-06T23:07:21Z",
        "currentTime", "2023-03-06T23:07:21Z",
        "deviceName": "MCP2",
        "deviceSerialNumber": "gopro1",
        "missionId": String,
        "recordNumber": Integer,

        # Media Reference Metadata
        "width": Integer, [for images or videos]
        "height": Integer, [for images or videos]
        "filename": "MCP2_gopro1_2023-03-06_23-07-21.jpg",
        "displayUrl": "http://imagery.jatf-dev.org:5000/image/MCP2_gopro1/2023/03/06/MCP2_gopro1_2023-03-06_23-07-21.jpg",
        "s3Bucket": "imagery-pipeline",
        "s3Region": "us-gov-east-1",
        "s3Path": "MCP2_gopro1/2023/03/06/MCP2_gopro1_2023-03-06_23-07-21.jpg",
        "detector": "rekognition",
        "labels": [{}, {}, ]
    }
}

