import sqlite3
from pathlib import Path

DB_PATH = Path("player_data.sqlite")
IN_DIR = Path("ip_removed")
OUT_DIR = Path("id_stripped")
OUT_DIR.mkdir(exist_ok=True)

# Load player data
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT pid, ingame_id FROM Player")
id_to_pid = {ingame_id: str(pid) for pid, ingame_id in cursor.fetchall() if ingame_id}
conn.close()

for log_file in IN_DIR.glob("*.log"):
    with log_file.open("r", encoding="utf-8") as f:
        content = f.read()

    found = {ingame_id for ingame_id in id_to_pid if ingame_id in content}
    if not found:
        continue

    print(f"\nFile: {log_file.name}")
    print("Found IDs:")
    for ingame_id in sorted(found):
        print(f"  {ingame_id} → {id_to_pid[ingame_id]}")

    if len(found) <= 4:
        print(f"Auto-replacing {len(found)} IDs.")
    else:
        confirm = input("Replace IDs? [y/n]: ").strip().lower()
        if confirm != 'y':
            continue

    for ingame_id in found:
        content = content.replace(ingame_id, f"player_pid_{id_to_pid[ingame_id]}")

    out_file = OUT_DIR / log_file.name
    with out_file.open("w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved: {out_file}")
