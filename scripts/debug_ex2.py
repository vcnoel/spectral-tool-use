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

def extract_json(s):
    s = s.strip()
    # Try finding the first { and last }
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except:
            pass
    try:
        return json.loads(s)
    except:
        return None

def main():
    with open('data/qwen_mining_1000.jsonl', 'r') as f:
        for line in f:
            if "John012" in line:
                print("FOUND EXAMPLE")
                ex = json.loads(line)
                gt = ex['ground_truth']
                gen = ex['generated']
                
                print(f"RAW GT REPR: {repr(gt)}")
                
                gt_json = extract_json(gt)
                gen_json = extract_json(gen)
                
                print(f"GT EXTRACTED: {gt_json is not None}")
                if gt_json: print(f"GT KEYS: {gt_json.keys()}")
                
                print(f"GEN EXTRACTED: {gen_json is not None}")
                
                if gt_json is not None and gen_json is not None:
                    print("BOTH EXTRACTED. Checking Match...")
                    # Replicate logic
                    if gt_json == gen_json:
                         print("MATCH: YES")
                    else:
                         print("MATCH: NO")
                else:
                    print("ONE MISSING.")
                break

if __name__ == "__main__":
    main()
