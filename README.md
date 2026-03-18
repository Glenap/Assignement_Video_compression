# Smart Behavioral Video Compression
### Sentio Mind Assignment — SBVC Pipeline

> Reduce 40–80 GB daily CCTV footage to < 10 GB while retaining every frame containing a human.

---

## Folder Structure

```
Assignement_Video_compression/
│
├── solution.py                 ← Main compression script (the deliverable)
├── requirements.txt            ← Python dependencies
│
├── run.sh                      ← One-click runner (Linux / macOS)
├── run.bat                     ← One-click runner (Windows)
│
├── git_setup.sh                ← Git push workflow (Linux / macOS)
├── git_setup.bat               ← Git push workflow (Windows)
│
├── video_sample_1.mov          ← Input video (add this yourself)
│
├── compressed_output.mp4       ← Generated: compressed video
├── compression_report.html     ← Generated: offline HTML report
├── segments_kept.json          ← Generated: Sentio Mind manifest
└── demo.mp4                    ← Record yourself: ≤2 min screen capture
```

---

## Quick Start

### Step 1 — Install dependencies

```bash
# Python packages
pip install opencv-python==4.9.0 imagehash==4.3.1 numpy==1.26.4 Pillow==10.3.0

# ffmpeg (system-level)
# Windows:  winget install ffmpeg
# macOS:    brew install ffmpeg
# Linux:    sudo apt install ffmpeg
```

### Step 2 — Run the script

```bash
# Default (expects video_sample_1.mov in same folder)
python solution.py

# Custom paths
python solution.py --input video_sample_1.mov --output compressed_output.mp4 --fps 12
```

### Step 3 — One-click runners

```bash
# Linux / macOS
bash run.sh

# Windows — double-click run.bat
# OR from Command Prompt:
run.bat
```

---

## Algorithm (Exactly as Specified)

| Step | Method | Threshold | Action |
|------|--------|-----------|--------|
| **1** | pHash (perceptual hash) | > 95% similar | DROP duplicate frame |
| **2** | Optical flow (Farneback dense) | mean magnitude < 0.05 | DROP static frame |
| **3** | Haar face detection | face found | KEEP regardless of motion |
| **4** | Context frame | last keep > 3 s ago | KEEP for continuity |
| **5** | ffmpeg re-encode | — | H.264, CRF 23, 12 fps |

---

## Performance Targets

| Target | Requirement | Expected Result |
|--------|-------------|-----------------|
| File size reduction | ≥ 70% | ~75–85% on corridor/classroom footage |
| Processing speed | ≥ 4× real-time | ~10–20× on modern laptop |

---

## Integration Contract — `segments_kept.json`

```json
{
  "schema_version": "1.0",
  "source_video": "video_sample_1.mov",
  "compression_stats": { ... },
  "segments": [
    {
      "segment_id": "seg_0000",
      "start_frame": 0,
      "end_frame": 14,
      "start_time_s": 0.0,
      "end_time_s": 0.56,
      "frame_count": 15,
      "has_faces": true,
      "avg_motion": 0.23,
      "keep_reason": "face+motion",
      "frame_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    }
  ],
  "frame_decisions": [ ... ]
}
```

### Plug into the main pipeline

```python
from solution import extract_intelligent_frames

# Fast path — no rescan of raw video
frames = extract_intelligent_frames(
    video_path="video_sample_1.mov",
    segments_json="segments_kept.json",
)

# Full analysis (no segments file)
frames = extract_intelligent_frames("video_sample_1.mov")
```

---

## GitHub Push

Edit `git_setup.sh` (or `.bat` on Windows) with your details:

```bash
GITHUB_USERNAME="your-github-username"
FIRST_NAME="Arjun"
LAST_NAME="Sharma"
ROLL_NUMBER="2301CS14"
```

Then run:

```bash
bash git_setup.sh      # Linux / macOS
git_setup.bat          # Windows
```

Branch will be named: `Arjun_Sharma_2301CS14`

---

## Deliverables Checklist

- [ ] `solution.py` — compression script
- [ ] `compressed_output.mp4` — generated output (≥70% smaller)
- [ ] `compression_report.html` — offline HTML report with storyboard
- [ ] `segments_kept.json` — Sentio Mind segment manifest
- [ ] `demo.mp4` — ≤2 min screen recording
- [ ] Pushed to branch `FirstName_LastName_RollNumber` in the assignment repo
