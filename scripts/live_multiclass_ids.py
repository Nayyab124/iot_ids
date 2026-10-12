
from pathlib import Path
from datetime import datetime
import argparse
import csv

import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iot_ids_multiclass.joblib"
LOG_PATH = ROOT / "logs" / "live_multiclass_predictions.csv"
FEATURE_LOG_PATH = ROOT / "logs" / "live_multiclass_features.csv"

DEFAULT_INTERFACE = (
    r"\Device\NPF_{53C3EA3F-D2FD-4BE6-B7C3-43A787E2974F}"
)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
features = bundle["features"]
encoder = bundle["label_encoder"]

SERVICE_PORTS = {
    "HTTP": {80, 8080, 8000},
    "HTTPS": {443, 8443},
    "DNS": {53},
    "Telnet": {23},
    "SMTP": {25, 465, 587},
    "SSH": {22},
    "IRC": {6667, 6668, 6697},
    "DHCP": {67, 68},
}


def extract_features(packets):
    sizes, times, ttls, ip_headers, ip_protocols = [], [], [], [], []

    values = {
        "fin_count": 0, "syn_count": 0, "rst_count": 0,
        "psh_count": 0, "ack_count": 0, "ece_count": 0,
        "cwr_count": 0, "HTTP": 0, "HTTPS": 0, "DNS": 0,
        "Telnet": 0, "SMTP": 0, "SSH": 0, "IRC": 0,
        "TCP": 0, "UDP": 0, "DHCP": 0, "ARP": 0,
        "ICMP": 0, "IGMP": 0, "IPv": 0, "LLC": 0,
    }

    for packet in packets:
        sizes.append(float(len(packet)))
        times.append(float(packet.time))

        if IP in packet:
            ip = packet[IP]
            ip_headers.append(float(ip.ihl or 5) * 4)
            ttls.append(float(ip.ttl))
            ip_protocols.append(float(ip.proto))
            values["IPv"] += 1

            if ip.proto == 2:
                values["IGMP"] += 1

        if ARP in packet:
            values["ARP"] += 1

        if ICMP in packet:
            values["ICMP"] += 1

        src_port = dst_port = None

        if TCP in packet:
            tcp = packet[TCP]
            values["TCP"] += 1
            src_port, dst_port = int(tcp.sport), int(tcp.dport)
            flag = int(tcp.flags)

            for bit, name in [
                (1, "fin_count"), (2, "syn_count"),
                (4, "rst_count"), (8, "psh_count"),
                (16, "ack_count"), (64, "ece_count"),
                (128, "cwr_count"),
            ]:
                if flag & bit:
                    values[name] += 1

        elif UDP in packet:
            udp = packet[UDP]
            values["UDP"] += 1
            src_port, dst_port = int(udp.sport), int(udp.dport)

        if src_port is not None:
            for name, ports in SERVICE_PORTS.items():
                if src_port in ports or dst_port in ports:
                    values[name] += 1

    if not sizes:
        return None, 0

    size_array = np.asarray(sizes, dtype=float)
    packet_count = len(sizes)
    duration = max(max(times) - min(times), 0.001)

    gaps = np.diff(np.sort(np.asarray(times, dtype=float)))
    iat = float(np.mean(gaps)) if len(gaps) else 0.0

    # Approximation for live windows; verify against the original
    # dataset feature generator before relying on predictions.
    rate = packet_count / duration

    def fraction(name):
        return values[name] / packet_count

    row = {
        "Header_Length": float(np.mean(ip_headers)) if ip_headers else 0.0,
        "Protocol Type": (
            float(max(set(ip_protocols), key=ip_protocols.count))
            if ip_protocols else 0.0
        ),
        "Time_To_Live": float(np.mean(ttls)) if ttls else 0.0,
        "Rate": rate,

        "fin_flag_number": fraction("fin_count"),
        "syn_flag_number": fraction("syn_count"),
        "rst_flag_number": fraction("rst_count"),
        "psh_flag_number": fraction("psh_count"),
        "ack_flag_number": fraction("ack_count"),
        "ece_flag_number": fraction("ece_count"),
        "cwr_flag_number": fraction("cwr_count"),

        "fin_count": values["fin_count"],
        "syn_count": values["syn_count"],
        "rst_count": values["rst_count"],
        "psh_count": values["psh_count"],
        "ack_count": values["ack_count"],

        "HTTP": fraction("HTTP"),
        "HTTPS": fraction("HTTPS"),
        "DNS": fraction("DNS"),
        "Telnet": fraction("Telnet"),
        "SMTP": fraction("SMTP"),
        "SSH": fraction("SSH"),
        "IRC": fraction("IRC"),
        "TCP": fraction("TCP"),
        "UDP": fraction("UDP"),
        "DHCP": fraction("DHCP"),
        "ARP": fraction("ARP"),
        "ICMP": fraction("ICMP"),
        "IGMP": fraction("IGMP"),
        "IPv": fraction("IPv"),

        # LLC definition is not yet verified; do not invent a value.
        "LLC": 0.0,

        "Tot sum": float(np.sum(size_array)),
        "Min": float(np.min(size_array)),
        "Max": float(np.max(size_array)),
        "AVG": float(np.mean(size_array)),
        "Std": float(np.std(size_array)),
        # Confirmed by the inspected training CSV samples.
        "Tot size": float(np.mean(size_array)),
        "IAT": iat,
        "Number": float(packet_count),
        "Variance": float(np.var(size_array)),
    }

    return {name: row.get(name, 0.0) for name in features}, duration


def predict_window(packets):
    extracted, duration = extract_features(packets)
    if extracted is None:
        return None

    frame = pd.DataFrame([extracted], columns=features)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    X = imputer.transform(frame)
    X = scaler.transform(X)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    label = encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))

    return {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "prediction": str(label),
        "confidence": round(confidence, 6),
        "packets": len(packets),
        "window_duration_seconds": round(duration, 6),
        "status": "Experimental Live Detection",
        "features": extracted,
    }


def save_result(result):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(result)


def save_feature_result(metadata, extracted):
    FEATURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = (
        FEATURE_LOG_PATH.exists()
        and FEATURE_LOG_PATH.stat().st_size > 0
    )

    row = {**metadata, **extracted}
    with FEATURE_LOG_PATH.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default=DEFAULT_INTERFACE)
    parser.add_argument("--window-packets", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERIMENTAL LIVE MULTICLASS IoT IDS")
    print("Classes:", list(encoder.classes_))
    print("Interface:", args.interface)
    print("Output:", LOG_PATH)
    print("Features are approximate; predictions are not ground truth.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        while True:
            packets = sniff(
                iface=args.interface,
                count=max(1, args.window_packets),
                timeout=max(1, args.timeout),
                store=True,
            )

            if not packets:
                print("No packets captured; waiting...")
                continue

            result = predict_window(packets)
            if result:
                extracted = result.pop("features", {})
                save_result(result)
                save_feature_result(result, extracted)
                print(
                    f"{result['prediction']:<18} "
                    f"confidence={result['confidence']:.1%} "
                    f"packets={result['packets']}"
                )

    except KeyboardInterrupt:
        print("\nLive IDS stopped.")
    except Exception as error:
        print(f"\nCapture error: {error}")
        print("Check the interface and try running PowerShell as Administrator.")
        raise


if __name__ == "__main__":
    main()
