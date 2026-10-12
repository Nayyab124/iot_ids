from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse

import joblib
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from scapy.all import sniff, IP, IPv6, TCP, UDP, ARP, ICMP, IGMP, Dot3, LLC

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ciciot_xgboost_binary.json"
PREP_PATH = ROOT / "models" / "ciciot_preprocessing.joblib"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OUTPUT_PATH = ROOT / "logs" / "live_ids_predictions.csv"
WINDOW_SIZE = 10

def packet_features(packets):
    lengths, ttls, headers, proto_values, times = [], [], [], [], []
    flag_rows, ports = [], []
    counts = Counter()

    for p in packets:
        lengths.append(len(p))
        times.append(float(p.time))
        ttl = 0
        proto = 0
        header = 0

        if IP in p:
            layer = p[IP]
            ttl = int(layer.ttl)
            proto = int(layer.proto)
            header = int(layer.ihl or 5) * 4
            counts["IPv"] += 1
        elif IPv6 in p:
            layer = p[IPv6]
            ttl = int(layer.hlim)
            proto = int(layer.nh)
            header = 40

        if TCP in p:
            t = p[TCP]
            header += int(t.dataofs or 5) * 4
            flags = int(t.flags)
            flag_rows.append(flags)
            ports.append((int(t.sport), int(t.dport)))
            counts["TCP"] += 1
            counts["ack_count"] += bool(flags & 0x10)
            counts["syn_count"] += bool(flags & 0x02)
            counts["fin_count"] += bool(flags & 0x01)
            counts["rst_count"] += bool(flags & 0x04)
        elif UDP in p:
            u = p[UDP]
            header += 8
            ports.append((int(u.sport), int(u.dport)))
            flag_rows.append(0)
            counts["UDP"] += 1
        else:
            flag_rows.append(0)
            ports.append((0, 0))

        if ARP in p:
            counts["ARP"] += 1
            header = max(header, 28)
        if ICMP in p:
            counts["ICMP"] += 1
        if IGMP in p:
            counts["IGMP"] += 1
        if Dot3 in p or LLC in p:
            counts["LLC"] += 1

        headers.append(header)
        ttls.append(ttl)
        proto_values.append(proto)

    n = len(packets)
    sizes = np.asarray(lengths, dtype=float)
    times = np.asarray(times, dtype=float)
    iat = float(np.mean(np.diff(times))) if n > 1 else 0.0
    rate = 1.0 / iat if iat > 0 else 0.0

    def fraction(condition):
        return float(np.mean(condition)) if n else 0.0

    def service(port):
        return fraction([
            port in pair for pair in ports
        ])

    flags = np.asarray(flag_rows, dtype=int)
    proto = Counter(proto_values).most_common(1)[0][0] if proto_values else 0

    return {
        "Header_Length": float(np.mean(headers)),
        "Protocol Type": float(proto),
        "Time_To_Live": float(np.mean(ttls)),
        "Rate": rate,
        "fin_flag_number": fraction((flags & 0x01) != 0),
        "syn_flag_number": fraction((flags & 0x02) != 0),
        "rst_flag_number": fraction((flags & 0x04) != 0),
        "psh_flag_number": fraction((flags & 0x08) != 0),
        "ack_flag_number": fraction((flags & 0x10) != 0),
        "ece_flag_number": fraction((flags & 0x40) != 0),
        "cwr_flag_number": fraction((flags & 0x80) != 0),
        "ack_count": int(counts["ack_count"]),
        "syn_count": int(counts["syn_count"]),
        "fin_count": int(counts["fin_count"]),
        "rst_count": int(counts["rst_count"]),
        "HTTP": service(80),
        "HTTPS": service(443),
        "DNS": service(53),
        "Telnet": service(23),
        "SMTP": service(25),
        "SSH": service(22),
        "IRC": service(194),
        "TCP": fraction([TCP in p for p in packets]),
        "UDP": fraction([UDP in p for p in packets]),
        "DHCP": fraction([
            UDP in p and (
                int(p[UDP].sport) in (67, 68)
                or int(p[UDP].dport) in (67, 68)
            ) for p in packets
        ]),
        "ARP": fraction([ARP in p for p in packets]),
        "ICMP": fraction([ICMP in p for p in packets]),
        "IGMP": fraction([IGMP in p for p in packets]),
        "IPv": fraction([IP in p for p in packets]),
        "LLC": fraction([Dot3 in p or LLC in p for p in packets]),
        "Tot sum": float(np.sum(sizes)),
        "Min": float(np.min(sizes)),
        "Max": float(np.max(sizes)),
        "AVG": float(np.mean(sizes)),
        "Std": float(np.std(sizes, ddof=1)) if n > 1 else 0.0,
        "Tot size": float(np.mean(sizes)),
        "IAT": iat,
        "Number": n,
        "Variance": float(np.var(sizes, ddof=1)) if n > 1 else 0.0,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Experimental passive live IoT IDS; firewall blocking is disabled."
    )
    parser.add_argument("--interface", required=True, help="Scapy interface name")
    parser.add_argument("--count", type=int, default=1000, help="Packets to capture")
    parser.add_argument("--timeout", type=int, default=60, help="Capture timeout seconds")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Prediction CSV path")
    args = parser.parse_args()

    if args.count < WINDOW_SIZE or args.timeout < 1:
        parser.error(f"--count must be at least {WINDOW_SIZE}; --timeout must be positive")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    threshold = float(config.get("detection", {}).get("confidence_threshold", 0.80))
    prep = joblib.load(PREP_PATH)
    columns = prep["feature_columns"]
    medians = pd.Series(prep["medians"]).reindex(columns)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    packets = []
    print(f"Passive capture on {args.interface}; max {args.count} packets / {args.timeout}s")
    print("Experimental features only. Payload storage disabled. Firewall blocking DISABLED.")

    sniff(
        iface=args.interface,
        prn=packets.append,
        store=False,
        count=args.count,
        timeout=args.timeout,
    )

    usable = len(packets) // WINDOW_SIZE * WINDOW_SIZE
    if usable == 0:
        print(f"Captured {len(packets)} packets; no complete 10-packet window to evaluate.")
        return

    output_rows = []
    for start in range(0, usable, WINDOW_SIZE):
        window = packets[start:start + WINDOW_SIZE]
        features = packet_features(window)
        X = pd.DataFrame([features]).reindex(columns=columns)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)

        probability = float(model.predict_proba(X)[0, 1])
        attack = probability >= threshold
        timestamp = datetime.now(timezone.utc).isoformat()

        row = {
            "window_number": start // WINDOW_SIZE + 1,
            "evaluated_at_utc": timestamp,
            **features,
            "attack_probability": probability,
            "prediction": "ATTACK" if attack else "BENIGN",
            "alert": bool(attack),
            "confidence_threshold": threshold,
            "feature_method": "experimental_10_packet_window",
        }
        output_rows.append(row)
        print(
            f"Window {row['window_number']:>4}: "
            f"{row['prediction']:<6} probability={probability:.3f}"
        )

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output, index=False)

    print("\nLive IDS run finished.")
    print(f"Packets captured: {len(packets)}")
    print(f"Complete windows evaluated: {len(output_rows)}")
    print(f"Unused packets in final partial window: {len(packets) - usable}")
    print(f"Alerts: {sum(r['alert'] for r in output_rows)}")
    print(f"Predictions saved: {output}")
    print("Firewall blocking: DISABLED")
    print("WARNING: Feature calculations are experimental, not verified against the original dataset extractor.")

if __name__ == "__main__":
    main()
