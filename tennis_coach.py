#!/usr/bin/env python3
"""
Tennis AI Coach - Video Analysis Tool
Analyzes tennis videos to provide coaching feedback and shot statistics
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, cast
from datetime import datetime
import sys
import traceback

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
        
        # Upload the entire video file to Gemini for proper analysis
        print(f"Uploading video file to Gemini...")
        try:
            # Upload video file using Files API
            myfile = cast(Any, genai).upload_file(video_path)
            print(f"✅ Video uploaded successfully: {myfile.name}")
            
            # Wait for video to be processed
            import time
            while myfile.state.name == "PROCESSING":
                print("⏳ Video processing...")
                time.sleep(10)
                myfile = cast(Any, genai).get_file(myfile.name)
            
            if myfile.state.name == "FAILED":
                raise RuntimeError(f"Video processing failed: {myfile.state}")
                
        except Exception as e:
            print(f"Error uploading video: {e}")
            raise RuntimeError(f"Failed to upload video: {str(e)}")
        
        # Load prompt
        prompt = self.load_prompt()
        
        print(f"Sending video to AI for shot-by-shot analysis...")
        # Debug log to help trace if the error originates from the Google API
        print("[DEBUG] Calling Gemini generate_content with video file", file=sys.stderr)
        try:
            # Generate analysis using the uploaded video file
            response = self.model.generate_content([myfile, prompt])
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
            
            # Clean up uploaded file
            try:
                cast(Any, genai).delete_file(myfile.name)
                print("🗑️ Cleaned up uploaded video file")
            except Exception as e:
                print(f"Warning: Could not delete uploaded file: {e}")
            
            # Save shot data to tennis.json (the magic file!)
            with open(output_file, 'w') as f:
                json.dump(shot_data, f, indent=2)
            print(f"✅ Shot analysis saved to {output_file}")
            print(f"🎾 Found {len(shot_data.get('shots', []))} shots in the video")
            
            return shot_data
            
        except Exception as e:
            # Clean up uploaded file on error
            try:
                cast(Any, genai).delete_file(myfile.name)
            except:
                pass
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