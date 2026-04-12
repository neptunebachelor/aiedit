import json
import os

packs_dir = ".video_data/videos/VID_20260406_172953_002/infer/packs"

for pack in ["pack_0002", "pack_0003", "pack_0004"]:
    manifest_path = os.path.join(packs_dir, pack, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    responses = []
    for i, frame in enumerate(manifest["frames"]):
        # Keep the 2nd and 3rd frame in each pack just to have some highlights
        if i == 1 or i == 2:
            score = 0.85
            keep = True
            labels = ["apex", "scenery"]
            reason = "strong lean"
            discard_reason = ""
        else:
            score = 0.20
            keep = False
            labels = []
            reason = ""
            discard_reason = "low editorial value"
            
        responses.append({
            "frame_number": frame["frame_number"],
            "keep": keep,
            "score": score,
            "labels": labels,
            "reason": reason,
            "discard_reason": discard_reason
        })
        
    response_path = os.path.join(packs_dir, pack, "response.json")
    with open(response_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False, no BOM (Python's "utf-8" doesn't write BOM)
        json.dump(responses, f, ensure_ascii=False, indent=2)

print("Generated response.json for packs 2, 3, 4.")
