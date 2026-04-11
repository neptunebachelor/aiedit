import json
import os

with open('temp_index.json', 'r', encoding='utf-8') as f:
    index_data = json.load(f)

candidates = [f for f in index_data['frames'] if f.get('candidate')]

for i in range(220, 300):
    if i >= len(candidates): break
    f = candidates[i]
    print(f'Idx {i}: frame {f["frame_number"]}')
