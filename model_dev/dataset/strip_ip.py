import os
import re
from pathlib import Path

RAW_DIR = Path('./raw')
OUT_DIR = Path('./ip_removed')
OUT_DIR.mkdir(exist_ok=True)

ipv4_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

for txt_file in RAW_DIR.glob('*.log'):
    with txt_file.open('r', encoding='utf-8') as f:
        content = f.read()

    matches = set(re.findall(ipv4_pattern, content))
    if not matches:
        continue

    print(f"\nFile: {txt_file.name}")
    print("Found IPs:")
    for ip in sorted(matches):
        print("  ", ip)

    if len(matches) <= 4:
        print(f"Auto-replacing {len(matches)} IPs.")
    else:
        confirm = input("Delete? [y/n]: ").strip().lower()
        if confirm != 'y':
            continue

    for ip in matches:
        content = content.replace(ip, 'x.x.x.x')

    out_file = OUT_DIR / txt_file.name
    with out_file.open('w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved: {out_file}")
