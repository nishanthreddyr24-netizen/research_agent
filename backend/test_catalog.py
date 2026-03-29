from pathlib import Path
import json

path = Path(r"c:\Users\R Nishanth Reddy\Desktop\vs_code_researcg\backend\storage\papers.json")
print(f"Path: {path}")
print(f"Exists: {path.exists()}")
if path.exists():
    content = path.read_text(encoding="utf-8")
    print(f"Content length: {len(content)}")
    data = json.loads(content)
    print(f"Data: {data}")