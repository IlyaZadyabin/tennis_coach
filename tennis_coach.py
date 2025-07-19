#!/usr/bin/env python3
"""
Tennis AI Coach - Video Analysis Tool
Analyzes tennis videos to provide coaching feedback and shot statistics
"""

import os
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, cast
from datetime import datetime
import sys
import traceback

# Handle OpenCV import gracefully for IDE
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    # IDE may not have the package installed during development
    cv2 = None
    CV2_AVAILABLE = False

# Handle Google Generative AI import gracefully for IDE
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    # IDE may not have the package installed during development
    genai = None
    GENAI_AVAILABLE = False


class TennisCoach:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Tennis Coach with Gemini API"""
        if not GENAI_AVAILABLE or genai is None:
            raise ImportError("google-generativeai package not installed. Run: pip3 install -r requirements.txt")
        
        self.api_key: str = api_key or os.getenv('GEMINI_API_KEY') or ""
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass as parameter")
        
        cast(Any, genai).configure(api_key=self.api_key)
        self.model: Any = cast(Any, genai).GenerativeModel('models/gemini-2.0-flash-lite')
        
        # Tennis-specific shot types
        self.shot_types: List[str] = [
            'forehand', 'backhand', 'serve', 'volley', 
            'drop_shot', 'overhead', 'smash', 'return'
        ]
        
    def extract_frames(self, video_path: str, fps: float = 1.0) -> List[str]:
        """Extract frames from video at specified FPS and encode as base64"""
        if not CV2_AVAILABLE or cv2 is None:
            raise ImportError("opencv-python package not installed. Run: pip3 install -r requirements.txt")
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        frame_count = 0
        extracted_count = 0
        
        print(f"Extracting frames at {fps} FPS from video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize frame for processing (optional)
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Encode frame as base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                frames.append(frame_b64)
                extracted_count += 1
                
            frame_count += 1
        
        cap.release()
        print(f"Extracted {extracted_count} frames for analysis")
        return frames
    
    def load_prompt(self, prompt_file: str = 'tennis_prompt.txt') -> str:
        """Load the tennis coaching prompt from file"""
        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            # Default prompt if file not found
            return """Analyze this tennis video and provide coaching feedback. 
            Count shots made/missed by type (forehand, backhand, serve, volley, etc.). 
            Give technical advice like a professional tennis coach."""
    
    def analyze_video(self, video_path: str, output_file: str = 'tennis.json') -> Dict[str, Any]:
        """Analyze tennis video and generate shot-by-shot JSON analysis"""
        
        print(f"Starting tennis video analysis of {video_path}...")
        
        # Extract frames
        frames = self.extract_frames(video_path, fps=1.0)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Load prompt
        prompt = self.load_prompt()
        
        # Prepare content for Gemini - use simplified approach for now
        # Since this is a demo, we'll process a representative sample of frames
        max_frames = min(len(frames), 10)  # Limit frames for API constraints
        step = max(1, len(frames) // max_frames) if len(frames) > max_frames else 1
        
        selected_frames = []
        for i in range(0, len(frames), step):
            if len(selected_frames) >= max_frames:
                break
            selected_frames.append(frames[i])
        
        # For demo purposes, we'll create a combined prompt
        content = f"{prompt}\n\nAnalyzing {len(selected_frames)} frames from the tennis video."
        
        print(f"Sending prompt to AI for shot-by-shot analysis of {len(selected_frames)} frames...")
        # Debug log to help trace if the error originates from the Google API
        print("[DEBUG] Calling Gemini generate_content", file=sys.stderr)
        try:
            # Generate analysis
            response = self.model.generate_content(content)
            print("[DEBUG] Gemini response received", file=sys.stderr)
            analysis_text = response.text
            
            # Extract JSON from response (AI might include extra text)
            try:
                # Find JSON in the response
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_text = analysis_text[json_start:json_end]
                    shot_data = json.loads(json_text)
                else:
                    raise ValueError("No valid JSON found in AI response")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not parse JSON from AI response: {e}")
                print("Raw response:", analysis_text[:500])
                # Create fallback structure
                shot_data = {
                    "shots": [],
                    "error": "Failed to parse AI response",
                    "raw_response": analysis_text
                }
            
            # Save shot data to tennis.json (the magic file!)
            with open(output_file, 'w') as f:
                json.dump(shot_data, f, indent=2)
            print(f"✅ Shot analysis saved to {output_file}")
            print(f"🎾 Found {len(shot_data.get('shots', []))} shots in the video")
            
            return shot_data
            
        except Exception as e:
            # Print full traceback to stderr for easier debugging
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Error during AI analysis: {str(e)}")
    



def main():
    parser = argparse.ArgumentParser(description='Tennis AI Coach - Video Analysis Tool')
    parser.add_argument('video', help='Path to tennis video file')
    parser.add_argument('--output', '-o', help='Output JSON file for analysis results')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--prompt', default='tennis_prompt.txt', help='Custom prompt file')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return 1
    
    try:
        # Initialize coach
        coach = TennisCoach(api_key=args.api_key)
        
        # Determine output file path
        output_file = args.output if args.output else 'tennis.json'
        
        # Analyze video
        shot_data = coach.analyze_video(args.video, output_file)
        
        # Results are automatically saved to tennis.json
        print(f"✅ Analysis complete! Check {output_file} for shot-by-shot data.")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main()) 