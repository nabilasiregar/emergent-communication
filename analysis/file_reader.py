import json
import csv

def json_log_to_csv(
    json_path: str,
    csv_path: str,
    keys: list = None
):
    """
    Read json_path which should have one JSON object per line (from ConsoleLogger).
    Write out a CSV to csv_path with columns = keys (in order). If keys is None,
    use whatever keys appear in the first JSON line (in sorted order).
    """
    with open(json_path, "r") as fin, open(csv_path, "w", newline="") as fout:
        writer = None
        header = keys

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # skip any non‐JSON lines (e.g. extra log text)
                continue

            if header is None:
                header = sorted(record.keys())

            if writer is None:
                writer = csv.writer(fout)
                writer.writerow(header)

            row = [record.get(k, "") for k in header]
            writer.writerow(row)

if __name__ == "__main__":
    json_log_to_csv(json_path="logs/json/human_gs_seed2025.log", csv_path="results/human_gs_seed2025.csv")