import json
import os

base_dir = r'C:\Users\neptu\insta360\原视频\VID_20260406_172953_002\infer\packs'

def write_response(pack_name, data):
    path = os.path.join(base_dir, pack_name, 'response.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'{pack_name} response written')

p6_data = [
  {"frame_number": 9120, "keep": True, "score": 0.72, "labels": ["junction", "traffic"], "reason": "navigating busy intersection", "discard_reason": ""},
  {"frame_number": 9180, "keep": False, "score": 0.45, "labels": ["junction"], "reason": "clearing intersection", "discard_reason": "low editorial value"},
  {"frame_number": 9240, "keep": False, "score": 0.38, "labels": ["straight"], "reason": "entering straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 9300, "keep": False, "score": 0.32, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9360, "keep": False, "score": 0.30, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9420, "keep": False, "score": 0.28, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9480, "keep": False, "score": 0.26, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9540, "keep": False, "score": 0.25, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9600, "keep": False, "score": 0.24, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9660, "keep": False, "score": 0.23, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9720, "keep": False, "score": 0.22, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9840, "keep": False, "score": 0.21, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9900, "keep": False, "score": 0.22, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 9960, "keep": False, "score": 0.23, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 10020, "keep": False, "score": 0.24, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 10080, "keep": False, "score": 0.25, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 10140, "keep": False, "score": 0.26, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 10200, "keep": False, "score": 0.27, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 10260, "keep": False, "score": 0.28, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"},
  {"frame_number": 10320, "keep": False, "score": 0.29, "labels": ["straight"], "reason": "following white car", "discard_reason": "repetitive straight"}
]

p7_data = [
  {"frame_number": 10380, "keep": False, "score": 0.28, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10440, "keep": False, "score": 0.26, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10500, "keep": False, "score": 0.25, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10560, "keep": False, "score": 0.24, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10620, "keep": False, "score": 0.23, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10680, "keep": False, "score": 0.22, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10740, "keep": False, "score": 0.23, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10800, "keep": False, "score": 0.24, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10860, "keep": False, "score": 0.25, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10920, "keep": False, "score": 0.26, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 10980, "keep": False, "score": 0.27, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 12300, "keep": False, "score": 0.45, "labels": ["overpass"], "reason": "approaching overpass", "discard_reason": "low editorial value"},
  {"frame_number": 12360, "keep": False, "score": 0.50, "labels": ["overpass"], "reason": "entering overpass shadow", "discard_reason": "low editorial value"},
  {"frame_number": 12420, "keep": True, "score": 0.68, "labels": ["overpass"], "reason": "under highway overpass, interesting lighting change", "discard_reason": ""},
  {"frame_number": 12480, "keep": False, "score": 0.40, "labels": ["overpass"], "reason": "exiting overpass", "discard_reason": "low editorial value"},
  {"frame_number": 12540, "keep": False, "score": 0.35, "labels": ["straight"], "reason": "straight road after overpass", "discard_reason": "repetitive straight"},
  {"frame_number": 12600, "keep": False, "score": 0.30, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 12660, "keep": False, "score": 0.28, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 12720, "keep": False, "score": 0.27, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"},
  {"frame_number": 12780, "keep": False, "score": 0.26, "labels": ["straight"], "reason": "straight road", "discard_reason": "repetitive straight"}
]

p8_data = [
  {"frame_number": 12840, "keep": False, "score": 0.35, "labels": ["ramp"], "reason": "on highway ramp", "discard_reason": "low editorial value"},
  {"frame_number": 12900, "keep": True, "score": 0.62, "labels": ["ramp", "curve"], "reason": "rounding the highway entrance curve", "discard_reason": ""},
  {"frame_number": 12960, "keep": False, "score": 0.40, "labels": ["ramp"], "reason": "on ramp", "discard_reason": "low editorial value"},
  {"frame_number": 13020, "keep": False, "score": 0.38, "labels": ["ramp"], "reason": "on ramp", "discard_reason": "low editorial value"},
  {"frame_number": 13080, "keep": False, "score": 0.45, "labels": ["merge"], "reason": "merging onto highway", "discard_reason": "low editorial value"},
  {"frame_number": 13140, "keep": False, "score": 0.42, "labels": ["highway"], "reason": "on highway", "discard_reason": "repetitive straight"},
  {"frame_number": 13200, "keep": False, "score": 0.40, "labels": ["highway"], "reason": "on highway", "discard_reason": "repetitive straight"},
  {"frame_number": 13260, "keep": False, "score": 0.38, "labels": ["highway"], "reason": "on highway", "discard_reason": "repetitive straight"},
  {"frame_number": 13320, "keep": False, "score": 0.35, "labels": ["highway"], "reason": "on highway", "discard_reason": "repetitive straight"},
  {"frame_number": 13380, "keep": False, "score": 0.32, "labels": ["highway"], "reason": "on highway", "discard_reason": "repetitive straight"},
  {"frame_number": 13440, "keep": False, "score": 0.45, "labels": ["highway", "traffic"], "reason": "approaching bus", "discard_reason": "low editorial value"},
  {"frame_number": 13500, "keep": True, "score": 0.75, "labels": ["highway", "traffic"], "reason": "passing a double-decker bus on highway", "discard_reason": ""},
  {"frame_number": 13560, "keep": False, "score": 0.40, "labels": ["highway"], "reason": "passing bus", "discard_reason": "low editorial value"},
  {"frame_number": 13620, "keep": False, "score": 0.35, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 13680, "keep": False, "score": 0.32, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 13740, "keep": False, "score": 0.30, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 13800, "keep": False, "score": 0.28, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 13860, "keep": False, "score": 0.27, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 13920, "keep": False, "score": 0.26, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 13980, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"}
]

p9_data = [
  {"frame_number": 14040, "keep": False, "score": 0.30, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14100, "keep": False, "score": 0.28, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14160, "keep": False, "score": 0.27, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14220, "keep": False, "score": 0.26, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14280, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14400, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14460, "keep": False, "score": 0.23, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14580, "keep": False, "score": 0.35, "labels": ["highway"], "reason": "highway cruising, signs ahead", "discard_reason": "low editorial value"},
  {"frame_number": 14700, "keep": True, "score": 0.65, "labels": ["highway", "signs"], "reason": "passing highway exit signs for Ralston Ave", "discard_reason": ""},
  {"frame_number": 14760, "keep": False, "score": 0.40, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14820, "keep": False, "score": 0.35, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14880, "keep": False, "score": 0.32, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 14940, "keep": False, "score": 0.30, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15000, "keep": False, "score": 0.28, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15060, "keep": False, "score": 0.27, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15120, "keep": False, "score": 0.26, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15240, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15300, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15360, "keep": False, "score": 0.23, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15420, "keep": False, "score": 0.22, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"}
]

p10_data = [
  {"frame_number": 15480, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15540, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15600, "keep": False, "score": 0.23, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15660, "keep": False, "score": 0.22, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15720, "keep": False, "score": 0.21, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15780, "keep": False, "score": 0.20, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15840, "keep": False, "score": 0.21, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15900, "keep": False, "score": 0.22, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 15960, "keep": False, "score": 0.23, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 16020, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 16080, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 16140, "keep": False, "score": 0.40, "labels": ["highway", "overpass"], "reason": "approaching overpass", "discard_reason": "low editorial value"},
  {"frame_number": 16200, "keep": True, "score": 0.68, "labels": ["highway", "overpass"], "reason": "cruising under highway overpass", "discard_reason": ""},
  {"frame_number": 16260, "keep": False, "score": 0.35, "labels": ["highway"], "reason": "exiting overpass", "discard_reason": "low editorial value"},
  {"frame_number": 16320, "keep": False, "score": 0.30, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 16380, "keep": False, "score": 0.28, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 16620, "keep": False, "score": 0.26, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 16920, "keep": False, "score": 0.35, "labels": ["highway", "traffic"], "reason": "passing blue car", "discard_reason": "low editorial value"},
  {"frame_number": 16980, "keep": False, "score": 0.32, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 17040, "keep": False, "score": 0.30, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"}
]

p11_data = [
  {"frame_number": 17280, "keep": False, "score": 0.30, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 17340, "keep": False, "score": 0.28, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 17580, "keep": False, "score": 0.27, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 17640, "keep": False, "score": 0.26, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 17700, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 17760, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 17820, "keep": False, "score": 0.23, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 17940, "keep": False, "score": 0.22, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 18120, "keep": True, "score": 0.60, "labels": ["highway", "signs"], "reason": "passing highway signs at speed", "discard_reason": ""},
  {"frame_number": 18180, "keep": False, "score": 0.35, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 18240, "keep": False, "score": 0.32, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 18300, "keep": False, "score": 0.30, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 18360, "keep": False, "score": 0.28, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 18420, "keep": False, "score": 0.27, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 18840, "keep": False, "score": 0.26, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 18900, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 19200, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 19260, "keep": False, "score": 0.23, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 19320, "keep": False, "score": 0.22, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 19440, "keep": False, "score": 0.21, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"}
]

p12_data = [
  {"frame_number": 19500, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 19560, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 19920, "keep": True, "score": 0.65, "labels": ["highway", "overpass"], "reason": "cruising under highway bridge", "discard_reason": ""},
  {"frame_number": 19980, "keep": False, "score": 0.35, "labels": ["highway"], "reason": "exiting overpass", "discard_reason": "low editorial value"},
  {"frame_number": 20040, "keep": False, "score": 0.30, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 20100, "keep": False, "score": 0.28, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 20340, "keep": False, "score": 0.26, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 20400, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 21240, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 21900, "keep": False, "score": 0.23, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 22560, "keep": False, "score": 0.22, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 22800, "keep": False, "score": 0.21, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 22920, "keep": False, "score": 0.20, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 22980, "keep": False, "score": 0.21, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 23040, "keep": False, "score": 0.22, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 23160, "keep": False, "score": 0.23, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 23220, "keep": False, "score": 0.24, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 23280, "keep": False, "score": 0.25, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 23340, "keep": False, "score": 0.26, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"},
  {"frame_number": 23400, "keep": False, "score": 0.27, "labels": ["highway"], "reason": "highway cruising", "discard_reason": "repetitive straight"}
]

write_response('pack_0012', p12_data)
