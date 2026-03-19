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
    print("Loading dataset...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    
    print(f"Features: {ds.features}")
    
    # Print first few examples
    for i in range(3):
        print(f"\n--- Example {i} ---")
        ex = ds[i]
        print(f"Keys: {ex.keys()}")
        print(f"System: {ex.get('system', 'N/A')}")
        chat = ex.get('chat', 'N/A')
        print(f"Chat Type: {type(chat)}")
        print(f"Chat Content: {chat}")

if __name__ == "__main__":
    main()
