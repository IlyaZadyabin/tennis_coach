# Tennis AI Coach

Turn raw tennis footage into instant analytics and coaching insights with Google Gemini.

## Key Features
- Shot detection (forehand, backhand, serve, volley, …)
- Make / miss stats & cumulative score
- AI-generated technical feedback
- Optional video overlay with live stats

## Install
```bash
# (optional) python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-key"   # get one at makersuite.google.com/app/apikey
```

## Quick Start
```bash
python run_demo.py path/to/your_video.mp4  # runs analysis + creates annotated video
```
Outputs:
* `tennis.json` – structured shot-by-shot data
* `your_video_analysis_output.mp4` – video with overlays

## Separate Steps
```bash
# 1. Generate analysis only
python tennis_coach.py video.mp4            # → tennis.json
# 2. Visualise existing analysis
python tennis_visualizer.py video.mp4       # → video_analysis_output.mp4
```

That’s it – hit the court and level-up! 