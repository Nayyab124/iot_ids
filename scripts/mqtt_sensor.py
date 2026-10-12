import json
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

BROKER = "127.0.0.1"
PORT = 1883
TOPIC = "smartbuilding/room1/temperature"
connected = False


def on_connect(client, userdata, flags, reason_code, properties):
    global connected
    if reason_code == 0:
        connected = True
        print("Sensor connected to MQTT broker.")
    else:
        print(f"MQTT connection failed: {reason_code}")


client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="temperature-sensor-01",
)
client.on_connect = on_connect

try:
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    for _ in range(10):
        deadline = time.time() + 10
        while not connected and time.time() < deadline:
            time.sleep(0.1)

        if not connected:
            raise RuntimeError("Could not connect to MQTT broker.")

        reading = {
            "device_id": "temperature-sensor-01",
            "temperature_c": 22.0 + (time.time() % 5),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        info = client.publish(TOPIC, json.dumps(reading), qos=1)
        info.wait_for_publish(timeout=5)
        print(f"Published: {reading}")
        time.sleep(2)

except KeyboardInterrupt:
    print("Sensor stopped.")
finally:
    client.loop_stop()
    if client.is_connected():
        client.disconnect()
