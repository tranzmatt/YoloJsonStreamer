import os
import paho.mqtt.client as mqtt
import socket
import time

object_detection_message = {
    "version": "String",
    "messageType": "ObjectDetectionMessage",
    "data": {
        # Standard Metadata
        "deviceTime": "2023-03-06T23:07:21Z",
        "currentTime": "2023-03-06T23:07:21Z",
        "deviceName": "MCP2",
        "deviceSerialNumber": "gopro1",
        "missionId": "String",
        "recordNumber": 0,

        # Media Reference Metadata
        "width": 0, #[for images or videos]
        "height": 0,# [for images or videos]
        "filename": "MCP2_gopro1_2023-03-06_23-07-21.jpg",
        "displayUrl": "http://imagery.jatf-dev.org:5000/image/MCP2_gopro1/2023/03/06/MCP2_gopro1_2023-03-06_23-07-21.jpg",
        "s3Bucket": "imagery-pipeline",
        "s3Region": "us-gov-east-1",
        "s3Path": "MCP2_gopro1/2023/03/06/MCP2_gopro1_2023-03-06_23-07-21.jpg",
        "detector": "rekognition",
        "labels": [{}, {}, ]
    }
}

def create_client_id(video_source):
    # Get the hostname of the machine
    hostname = socket.gethostname()

    # Get the input source filename or stream identifier
    if video_source.startswith('rtsp://') or video_source.startswith('http://') or video_source.startswith('https://'):
        # Extract the stream identifier from the RTSP URL
        stream_identifier = video_source.split("/")[-1]
    elif video_source.startswith('udp://'):
        # Extract the IP address and port from the UDP URL
        udp_parts = video_source[6:].split(":")
        ip_address = udp_parts[0].replace(".", "-")
        port = udp_parts[1]
        stream_identifier = f"{ip_address}_{port}"
    else:
        # Get the filename from the input source
        stream_identifier = os.path.basename(video_source)

    # Combine the hostname and stream identifier to create the client_id
    client_id = f"{hostname}_{stream_identifier}"

    return client_id


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT client connected to the broker!")
    else:
        print(f"MQTT connection failed with error code {rc}")


# Define the on_publish callback function
def on_publish(client, userdata, mid):
    print(f"Message published successfully (Message ID: {mid})")


def publish_message(client, mqtt_topic, payload):
    
    if not client.is_connected():
        # Reconnect if not connected
        client.reconnect()

    try:
        result, mid = client.publish(mqtt_topic, payload)
        if result == mqtt.MQTT_ERR_SUCCESS:
            print(f"Publish successful (Message ID: {mid})")
        else:
            print(f"Publish failed with result: {result} ({mqtt.error_string(result)}). Retrying...")
    except Exception as e:
        print(f"Error during publish: {e}. Retrying...")

def create_mqtt_client(client_id=None, mqtt_host=None, mqtt_port=None, mqtt_user=None, mqtt_password=None):

    print(f"Connecting to MQTT: {mqtt_host} {mqtt_port} {mqtt_user} {mqtt_password}")

    #client = mqtt.Client(client_id=client_id, clean_session=False)
    client = mqtt.Client()

    # Set the on_connect callback
    client.on_connect = on_connect
    client.on_publish = on_publish

    if mqtt_user and mqtt_password:
        client.username_pw_set(mqtt_user, mqtt_password)

    # Define a variable to keep track of the connection status
    connected = False

    # Retry connection until successful
    while not connected:
        try:
            # Attempt to connect to the MQTT broker
            client.connect(mqtt_host, mqtt_port)
            connected = True
            print("MQTT client connected successfully!")
        except Exception as e:
            # Failed to connect, wait for a few seconds before retrying
            print(f"MQTT connection failed: {str(e)}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    client.loop_start()

    return client
