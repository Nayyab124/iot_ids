from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "CSV"
LIMIT = 10_000

CATEGORIES = [
    "Benign_Final",
    "Backdoor_Malware",
    "BrowserHijacking",
    "CommandInjection",
    "DDoS-ACK_Fragmentation",
]

EVALUATION_FILES = [
    DATA / "Benign_Final" / "BenignTraffic1.pcap.csv",
    DATA / "Benign_Final" / "BenignTraffic2.pcap.csv",
    DATA / "Benign_Final" / "BenignTraffic3.pcap.csv",
    DATA / "DDoS-HTTP_Flood" / "DDoS-HTTP_Flood-.pcap.csv",
]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

lines = [
    "TRAINING / EVALUATION CAPTURE AUDIT",
    f"Training row cap per category: {LIMIT}",
    "Selection below follows sorted CSV order and the per-category cap.",
    "A file is marked as a likely training source if the cap can be filled from it.",
    "",
]

training_candidates = []

for category in CATEGORIES:
    folder = DATA / category
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    lines.append(f"[{category}]")
    if not files:
        lines.append("  No CSV files found.")
        lines.append("")
        continue

    remaining = LIMIT
    for path in files:
        if remaining <= 0:
            break

        # Reading only enough rows to determine how much this file can contribute.
        sample = pd.read_csv(path, nrows=remaining, low_memory=False)
        count = len(sample)
        if count == 0:
            continue

        digest = sha256(path)
        training_candidates.append((path.resolve(), digest))
        lines.append(
            f"  CANDIDATE: {path.name} | contributes up to {count} rows "
            f"| SHA256={digest}"
        )
        remaining -= count

    if remaining > 0:
        lines.append(f"  WARNING: category may have fewer than {LIMIT} usable rows.")
    lines.append("")

lines.append("[Evaluation capture comparison]")
candidate_hashes = {digest: path for path, digest in training_candidates}
for path in EVALUATION_FILES:
    if not path.exists():
        lines.append(f"  MISSING: {path}")
        continue
    digest = sha256(path)
    match = candidate_hashes.get(digest)
    if match:
        lines.append(
            f"  POSSIBLE EXACT-FILE OVERLAP: {path.name} matches "
            f"training candidate {match.name}"
        )
    else:
        lines.append(f"  No exact-file hash match: {path.name} | SHA256={digest}")

lines.extend([
    "",
    "IMPORTANT LIMITATIONS:",
    "- Different hashes do not prove the traffic is independent; duplicate or related",
    "  captures may have different filenames or CSV formatting.",
    "- This audit estimates source files from the known sampling logic; it does not",
    "  prove which rows were used unless the training run logged that information.",
    "- Do not describe a row-random split as independent capture-level validation.",
])

report = ROOT / "reports" / "capture_independence_audit.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved report: {report}")
