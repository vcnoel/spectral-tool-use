# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
import os
import re
from pathlib import Path
from collections import defaultdict

INPUT_FILE = "data/validation_set.jsonl"
OUTPUT_DIR = "data/categories"

# Keyword mapping for categories
CATEGORIES = {
    "Finance": ["tax", "tip", "loan", "currency", "discount", "invoice", "shipping", "finance", "budget", "investment", "interest"],
    "Math": ["average", "sum", "math", "calculator", "area", "volume", "geometry", "random", "number"],
    "Personal": ["age", "bmi", "date", "time", "personal", "alarm", "schedule"],
    "Search": ["search", "weather", "news", "find", "locate"],
    "Travel": ["flight", "hotel", "travel", "booking", "trip"]
}

def get_category(system_prompt):
    s = system_prompt.lower()
    for cat, keywords in CATEGORIES.items():
        for k in keywords:
            if k in s:
                return cat
    return "Other"

def main():
    print(f"Reading from {INPUT_FILE}...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    counts = defaultdict(int)
    writers = {}
    
    # Pre-open handles? No, let's just collect in lists or open on demand. 
    # Lists are safer for 1000 items.
    data_by_cat = defaultdict(list)
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            try:
                ex = json.loads(line)
                sys = ex.get('system', '')
                cat = get_category(sys)
                data_by_cat[cat].append(line)
                counts[cat] += 1
            except:
                continue
                
    total = 0
    for cat, lines in data_by_cat.items():
        out_path = Path(OUTPUT_DIR) / f"{cat.lower()}.jsonl"
        with open(out_path, 'w') as f:
            for line in lines:
                f.write(line)
        print(f"Saved {len(lines)} examples to {out_path}")
        total += len(lines)
        
    print(f"Total processed: {total}")
    print("Breakdown:")
    for cat in sorted(counts.keys()):
        print(f"  {cat}: {counts[cat]}")

if __name__ == "__main__":
    main()
