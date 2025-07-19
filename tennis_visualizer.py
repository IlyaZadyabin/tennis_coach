import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import textwrap
import time
from typing import Any, cast, Tuple, Optional
import argparse
import os

# -------------------------------------------------------------
# FreeType text rendering helper (full Unicode support)
# -------------------------------------------------------------
try:
    _ft2 = cv2.freetype.createFreeType2()  # type: ignore[attr-defined]

    # Common system fonts that cover a wide Unicode range
    _font_candidates = [
        "/Library/Fonts/Arial Unicode.ttf",  # macOS (some installs)
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS Ventura+
        "/System/Library/Fonts/SFNS.ttf",  # macOS (older)
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ]

    _font_path = next((p for p in _font_candidates if os.path.exists(p)), None)
    if _font_path:
        _ft2.loadFontData(fontFileName=_font_path, id=0)
    else:
        # If no font found, disable FreeType to fall back gracefully
        _ft2 = None
except (AttributeError, cv2.error):
    # FreeType module not available in this OpenCV build
    _ft2 = None


def _draw_text(img, text: str, org: Tuple[int, int], font_scale: float, color: Tuple[int, int, int],
               thickness: int, border_color: Optional[Tuple[int, int, int]] = None,
               border_thickness: int = 0, line_type: int = cv2.LINE_AA) -> None:
    """Render text using FreeType when available; fallback to Hershey fonts.

    Args:
        img: destination image (modified in place)
        text: string to draw
        org: bottom-left corner (x, y)
        font_scale: same meaning as cv2.putText when falling back; when using
                     FreeType it is converted to a pixel height (approx 40*scale).
        color: BGR color tuple for main text
        thickness: thickness of main text strokes
        border_color: optional BGR color for outline
        border_thickness: stroke thickness for outline
        line_type: cv2 line type
    """

    if _ft2 is not None:
        pixel_height = max(1, int(font_scale * 40))  # rough mapping
        if border_color and border_thickness > 0:
            _ft2.putText(img, text, org, pixel_height, border_color, border_thickness, line_type, False)
        _ft2.putText(img, text, org, pixel_height, color, thickness, line_type, False)
    else:
        # Border (draw first so main text overlays it)
        if border_color and border_thickness > 0:
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        border_color, border_thickness, line_type)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, thickness, line_type)

# -----------------------------
# CLI argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Visualize tennis AI analysis overlays on video")
parser.add_argument('video', help='Path to input tennis video file')
parser.add_argument('--output', '-o', help='Path to save annotated video (defaults to <input>_analysis_output.<ext>)')
args = parser.parse_args()

input_video_path = args.video
# Derive default output path if not provided
if args.output:
    output_video_path = args.output
else:
    base, ext = os.path.splitext(input_video_path)
    output_video_path = f"{base}_analysis_output{ext}"

# Validate input file exists early
if not os.path.exists(input_video_path):
    raise FileNotFoundError(f"Input video file not found: {input_video_path}")

# Load shot data from JSON
with open('tennis.json', 'r') as f:
    shot_data = json.load(f)

# Initialize MediaPipe Pose
mp_solutions = cast(Any, mp.solutions)
mp_pose = mp_solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video file paths - update these with your actual video files
# input_video_path = 'tennis_video.mov'  # Your input tennis video
# output_video_path = 'tennis_analysis_output.mov'  # Final output with analysis

# Open the video file
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# List to store all processed frames
processed_frames = []

# Animation variables
last_shot_time = None
animation_duration = 1.5  # seconds for shot feedback animation
current_color = (255, 255, 255)  # Start with white

def parse_timestamp(timestamp):
    # Convert timestamp (e.g., "0:15.7") to seconds
    minutes, seconds = timestamp.split(':')
    return float(minutes) * 60 + float(seconds)

def timestamp_to_frame(timestamp, fps):
    # Convert timestamp to frame number
    seconds = parse_timestamp(timestamp)
    return int(seconds * fps)

def wrap_text(text, font, scale, thickness, max_width):
    # Calculate the maximum number of characters that can fit in max_width
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        text_size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        
        if text_size[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_animation_color(elapsed_time, is_made):
    if elapsed_time >= animation_duration:
        return (255, 255, 255)  # Return to white
    
    # Calculate progress through animation (0 to 1)
    progress = elapsed_time / animation_duration
    
    if progress < 0.5:
        # Fade to color
        if is_made:
            # Fade to green (BGR format)
            return (
                int(255 * (1 - progress * 2)),  # B
                255,                            # G
                int(255 * (1 - progress * 2))   # R
            )
        else:
            # Fade to red (BGR format)
            return (
                int(255 * (1 - progress * 2)),  # B
                int(255 * (1 - progress * 2)),  # G
                255                             # R
            )
    else:
        # Fade back to white
        if is_made:
            # Fade from green to white
            return (
                int(255 * ((progress - 0.5) * 2)),  # B
                255,                                # G
                int(255 * ((progress - 0.5) * 2))   # R
            )
        else:
            # Fade from red to white
            return (
                int(255 * ((progress - 0.5) * 2)),  # B
                int(255 * ((progress - 0.5) * 2)),  # G
                255                                 # R
            )

# Convert timestamps to frame numbers and add feedback display duration
if 'shots' in shot_data:
    for shot in shot_data['shots']:
        shot['frame_number'] = timestamp_to_frame(shot['timestamp_of_outcome'], fps)
        shot['feedback_end_frame'] = shot['frame_number'] + (4 * fps)  # Show feedback for 4 seconds

last_player_pos = None
frame_count = 0
process_every_n_frames = max(1, int(fps / 20))  # Process at reduced rate for performance
last_shot_result = None

print("Processing tennis video...")
print(f"Found {len(shot_data.get('shots', []))} shots to visualize")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    frame_count += 1
    
    # Only process pose detection every nth frame for performance
    if frame_count % process_every_n_frames == 0:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get the head landmark (landmark 0 is the top of the head)
            head = results.pose_landmarks.landmark[0]
            # Scale coordinates to frame resolution
            head_x = int(head.x * width)
            head_y = int(head.y * height)
            last_player_pos = (head_x, head_y)

    # Draw player tracking arrow and label if we have a position
    if last_player_pos is not None:
        head_x, head_y = last_player_pos
        arrow_height = 35
        arrow_width = 50
        arrow_tip_y = max(0, head_y - 120)  # Position arrow above player
        
        # Triangle points for the arrow
        pt1 = (head_x, arrow_tip_y + arrow_height)  # tip
        pt2 = (head_x - arrow_width // 2, arrow_tip_y)  # left
        pt3 = (head_x + arrow_width // 2, arrow_tip_y)  # right
        pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], (0, 100, 255))  # Orange arrow for tennis
        
        # Draw player name above the arrow
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "PLAYER"
        text_size = cv2.getTextSize(text, font, 2.2, 5)[0]
        text_x = head_x - text_size[0] // 2
        text_y = arrow_tip_y - 15
        
        # Draw player label with border (supports Unicode via FreeType)
        _draw_text(frame, text, (text_x, text_y), 2.2, (255, 255, 255), 5,
                   border_color=(0, 0, 0), border_thickness=12)

    # Calculate current shot statistics
    current_shots_made = 0
    current_shots_missed = 0
    current_forehands_made = 0
    current_backhands_made = 0
    current_serves_made = 0
    current_volleys_made = 0
    current_feedback = None
    
    # Process shots up to current frame
    if 'shots' in shot_data:
        for shot in shot_data['shots']:
            if shot['frame_number'] <= frame_count:
                if shot['frame_number'] == frame_count:
                    # New shot detected - trigger animation
                    last_shot_time = time.time()
                    last_shot_result = shot['result']
                
                # Update cumulative stats
                if shot['result'] == 'made':
                    current_shots_made = shot.get('total_shots_made_so_far', 0)
                else:
                    current_shots_missed = shot.get('total_shots_missed_so_far', 0)
                
                current_forehands_made = shot.get('total_forehands_made_so_far', 0)
                current_backhands_made = shot.get('total_backhands_made_so_far', 0)
                current_serves_made = shot.get('total_serves_made_so_far', 0)
                current_volleys_made = shot.get('total_volleys_made_so_far', 0)
                
                # Check if we should show feedback for this shot
                if shot['frame_number'] <= frame_count <= shot.get('feedback_end_frame', shot['frame_number']):
                    current_feedback = shot.get('feedback', '')

    # Display shot statistics in top left
    stats_font = cv2.FONT_HERSHEY_SIMPLEX
    stats_border = (0, 0, 0)  # Black border
    stats_scale = 0.8
    stats_thickness = 2
    stats_border_thickness = 4
    stats_spacing = 35
    white_color = (255, 255, 255)

    # Position for stats (top left with padding)
    stats_x = 20
    stats_y = 60

    # Calculate animation color if needed
    if last_shot_time is not None:
        elapsed_time = time.time() - last_shot_time
        if elapsed_time < animation_duration:
            current_color = get_animation_color(elapsed_time, last_shot_result == 'made')
        else:
            current_color = white_color
            last_shot_time = None

    # Draw overall statistics
    stats_lines = [
        f"Shots Made: {current_shots_made}",
        f"Shots Missed: {current_shots_missed}",
        f"Forehands: {current_forehands_made}",
        f"Backhands: {current_backhands_made}",
        f"Serves: {current_serves_made}",
        f"Volleys: {current_volleys_made}"
    ]
    
    for i, stat_text in enumerate(stats_lines):
        y_pos = stats_y + (i * stats_spacing)
        
        # Determine color (animate first two lines for overall stats)
        text_color = current_color if i < 2 else white_color
        
        # Draw statistic line with border
        _draw_text(frame, stat_text, (stats_x, y_pos), stats_scale, text_color, stats_thickness,
                   border_color=stats_border, border_thickness=stats_border_thickness)

    # Display coaching feedback if available
    if current_feedback:
        feedback_font = cv2.FONT_HERSHEY_SIMPLEX
        feedback_scale = 0.7
        feedback_color = (255, 255, 255)  # White text
        feedback_border = (0, 0, 0)  # Black border
        feedback_thickness = 2
        feedback_border_thickness = 4
        feedback_spacing = 30

        # Wrap text to fit within 80% of screen width
        max_width = int(width * 0.8)
        # If FreeType unavailable, replace em-dash to avoid '???' glyphs
        sanitized_feedback = current_feedback
        if _ft2 is None:
            sanitized_feedback = sanitized_feedback.replace('\u2014', '-')
        wrapped_lines = wrap_text(sanitized_feedback, feedback_font, feedback_scale, 
                                  feedback_thickness, max_width)

        # Calculate total height of wrapped text
        total_height = len(wrapped_lines) * feedback_spacing
        start_y = height - 50 - total_height

        # Draw each line centered
        for i, line in enumerate(wrapped_lines):
            text_size = cv2.getTextSize(line, feedback_font, feedback_scale, feedback_thickness)[0]
            feedback_x = (width - text_size[0]) // 2
            feedback_y = start_y + (i * feedback_spacing)

            # Draw feedback text with border (supports Unicode)
            _draw_text(frame, line, (feedback_x, feedback_y), feedback_scale, feedback_color, feedback_thickness,
                       border_color=feedback_border, border_thickness=feedback_border_thickness)

    # Store the processed frame
    processed_frames.append(frame.copy())

    # Optional: Display frame during processing (comment out for faster processing)
    # cv2.imshow('Tennis Analysis', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release video capture
cap.release()
cv2.destroyAllWindows()

print("Creating final tennis analysis video...")

# Create the final video
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write all frames to the final video
for frame in processed_frames:
    out.write(frame)

out.release()

print(f"✅ Tennis analysis complete!")
print(f"🎾 Final video saved to: {output_video_path}")
print(f"📊 Processed {len(processed_frames)} frames")
print(f"🏆 Analyzed {len(shot_data.get('shots', []))} shots")

# Print summary statistics
if 'shots' in shot_data and shot_data['shots']:
    total_shots = len(shot_data['shots'])
    made_shots = sum(1 for shot in shot_data['shots'] if shot['result'] == 'made')
    missed_shots = total_shots - made_shots
    accuracy = (made_shots / total_shots * 100) if total_shots > 0 else 0
    
    print(f"\n🎯 FINAL STATS:")
    print(f"   Total Shots: {total_shots}")
    print(f"   Made: {made_shots}")
    print(f"   Missed: {missed_shots}")
    print(f"   Accuracy: {accuracy:.1f}%") 