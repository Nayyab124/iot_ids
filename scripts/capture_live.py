from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv

from scapy.all import ARP, IP, IPv6, TCP, UDP, sniff, get_if_list

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "logs" / "live_packets.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Passively capture packet metadata for IDS development."
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List capture interfaces and exit.",
    )
    parser.add_argument(
        "--interface",
        help="Scapy interface name to capture from.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Maximum packets to capture (default: 100).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Maximum capture time in seconds (default: 30).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.list_interfaces:
        print("Available Scapy interfaces:")
        for name in get_if_list():
            print(f"  {name}")
        return

    if not args.interface:
        parser.error("Provide --interface or use --list-interfaces.")

    if args.count < 1 or args.timeout < 1:
        parser.error("--count and --timeout must both be positive.")

    rows = []

    def record_packet(packet):
        src_ip = ""
        dst_ip = ""
        protocol = "OTHER"
        src_port = ""
        dst_port = ""
        ttl = ""
        tcp_flags = ""

        if IP in packet:
            layer = packet[IP]
            src_ip = layer.src
            dst_ip = layer.dst
            ttl = layer.ttl
            protocol = str(layer.proto)
        elif IPv6 in packet:
            layer = packet[IPv6]
            src_ip = layer.src
            dst_ip = layer.dst
            protocol = "IPv6"
        elif ARP in packet:
            layer = packet[ARP]
            src_ip = layer.psrc
            dst_ip = layer.pdst
            protocol = "ARP"

        if TCP in packet:
            protocol = "TCP"
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            tcp_flags = str(packet[TCP].flags)
        elif UDP in packet:
            protocol = "UDP"
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport

        rows.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "packet_timestamp": float(packet.time),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": protocol,
            "src_port": src_port,
            "dst_port": dst_port,
            "packet_length": len(packet),
            "ttl": ttl,
            "tcp_flags": tcp_flags,
        })

        print(
            f"Captured {len(rows):>4} | {protocol:<5} | "
            f"{src_ip or '-'}:{src_port or '-'} -> "
            f"{dst_ip or '-'}:{dst_port or '-'} | "
            f"{len(packet)} bytes"
        )

    print(f"Starting passive capture on interface: {args.interface}")
    print(f"Maximum packets: {args.count}; timeout: {args.timeout}s")
    print("Payload storage: disabled. Firewall blocking: disabled.")

    try:
        sniff(
            iface=args.interface,
            prn=record_packet,
            store=False,
            count=args.count,
            timeout=args.timeout,
        )
    except Exception as exc:
        raise SystemExit(
            f"Capture failed: {exc}\n"
            "Check the interface name, capture permissions, and Npcap installation."
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp_utc", "packet_timestamp", "src_ip", "dst_ip",
        "protocol", "src_port", "dst_port", "packet_length",
        "ttl", "tcp_flags",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nCapture finished.")
    print(f"Packets recorded: {len(rows)}")
    print(f"Metadata saved to: {output_path}")


if __name__ == "__main__":
    main()
