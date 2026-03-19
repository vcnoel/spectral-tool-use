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

from datasets import load_dataset
import json

def main():
    print("Loading Stream...")
    # Try tuandunghcmut/toolbench-v1 or other variants if needed
    try:
        ds = load_dataset("tuandunghcmut/toolbench-v1", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading tuandunghcmut/toolbench-v1: {e}")
        return

    print("Iterating...")
    for i, ex in enumerate(ds):
        if i >= 3: break
        print(f"--- Example {i} ---")
        print("Keys:", ex.keys())
        # Print first level details
        for k, v in ex.items():
            if isinstance(v, list) and len(v) > 0:
                print(f"{k}: [List length {len(v)}]")
                print(f"  First item: {v[0]}")
            elif isinstance(v, str):
                print(f"{k}: {v[:200]}...")
            else:
                print(f"{k}: {v}")
        print("\n")

if __name__ == "__main__":
    main()
