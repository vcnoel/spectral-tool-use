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

#!/usr/bin/env python3
"""
Sample 100 examples from existing datasets for temperature comparison.
"""
import json
import random

# For T=1.0, sample from the existing validation set
random.seed(42)

with open("data/qwen_validation_2300_relabeled.jsonl", 'r') as f:
    t1_data = [json.loads(line) for line in f]

# Sample 100
t1_sample = random.sample(t1_data, 100)

with open("data/qwen_temp_1.0.jsonl", 'w') as f:
    for item in t1_sample:
        f.write(json.dumps(item) + "\n")

print(f"Sampled 100 from T=1.0 dataset")
print(f"Hallucinations: {sum(1 for x in t1_sample if x['label'] == 1)}/100")
