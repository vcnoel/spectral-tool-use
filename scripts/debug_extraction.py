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
import re

def extract_json(s):
    """Attempt to extract a JSON object from a string."""
    s = s.strip()
    # Try finding the first { and last }
    start = s.find('{')
    end = s.rfind('}')
    
    if start != -1 and end != -1:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            print(f"JSON extract failed on candidate: {e}")
            pass
            
    # Try generic load
    try:
        return json.loads(s)
    except:
        return None

def main():
    with open('data/qwen_ablate_pool_50.jsonl', 'r') as f:
        # Check first line
        line = f.readline()
        ex = json.loads(line)
        gen = ex['generated']
        print(f"RAW GEN: {gen[:100]}...")
        extracted = extract_json(gen)
        print(f"EXTRACTED: {extracted}")
        if extracted is None:
            print("EXTRACTION RETURNED NONE")
        else:
            print(f"KEYS: {extracted.keys()}")

if __name__ == "__main__":
    main()
