import json
import paho.mqtt.client as mqtt

BROKER = "127.0.0.1"
PORT = 1883
TOPIC = "smartbuilding/room1/temperature"


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Controller connected to MQTT broker.")
        client.subscribe(TOPIC, qos=1)
        print(f"Subscribed to: {TOPIC}")
    else:
        print(f"MQTT connection failed: {reason_code}")


def on_message(client, userdata, message):
    try:
        reading = json.loads(message.payload.decode("utf-8"))
        print(
            f"Received from {reading['device_id']}: "
            f"{reading['temperature_c']:.2f} C | "
            f"{reading['timestamp']}"
        )
    except (ValueError, KeyError, UnicodeDecodeError) as error:
        print(f"Invalid message received: {error}")


client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="smartbuilding-controller-01",
)
client.on_connect = on_connect
client.on_message = on_message

try:
    print("Waiting for MQTT messages. Press Ctrl+C to stop.")
    client.connect(BROKER, PORT, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("Controller stopped.")
finally:
    if client.is_connected():
        client.disconnect()
