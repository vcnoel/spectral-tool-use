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

def main():
    input_file = "data/pilot_run_50.jsonl"
    print(f"Inspecting {input_file}...")
    
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
        
    print(f"Loaded {len(data)} lines.")
    
    count = 0
    for i, ex in enumerate(data):
        # We want to see what the 'generated' looked like vs 'ground_truth'
        # The generation script labeled them as 1 (Hallucination) mostly.
        # Let's inspect a few.
        
        gt = ex.get('ground_truth', '')
        gen = ex.get('generated', '')
        label = ex.get('label', -1)
        
        if label == 1:
            print(f"\n--- Sample {i} (Gen Label: {label}) ---")
            print(f"GT: {repr(gt)}")
            print(f"GEN: {repr(gen)}")
            count += 1
            if count >= 10:
                break
                
if __name__ == "__main__":
    main()
