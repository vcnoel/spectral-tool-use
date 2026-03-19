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
    with open('data/qwen_mining_1000.jsonl', 'r') as f:
        count = 0
        for line in f:
            ex = json.loads(line)
            if ex['label'] == 1:
                print(f"LABEL: {ex['label']}")
                print(f"GT: {ex['ground_truth'][:100]}")
                print(f"GEN: {ex['generated'][:100]}")
                print("-" * 40)
                count += 1
                if count >= 3:
                    break

if __name__ == "__main__":
    main()
