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
import warnings
import re
import shutil

# Suppress SSL warnings for macOS LibreSSL compatibility
warnings.filterwarnings("ignore", message=".*urllib3.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*", category=UserWarning)

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
        self.model: Any = cast(Any, genai).GenerativeModel('gemini-2.5-pro')
        
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
    
    def _parse_json_response(self, text: str) -> Optional[Any]:
        """Robustly extract and parse JSON from the model response.

        The Gemini model sometimes wraps JSON in Markdown code fences, e.g.:
            ```json
            { ... }
            ```
        or includes explanatory text. This helper removes code fences and
        extraneous text before attempting json.loads.
        Returns the parsed JSON (dict/list) on success, or None on failure.
        """
        # Trim whitespace early
        stripped = text.strip()

        # 1) Attempt to extract any fenced code block that looks like JSON
        fence_regex = r"```(?:json)?\s*(.*?)\s*```"  # non-greedy match across the full string
        m = re.search(fence_regex, stripped, flags=re.DOTALL | re.IGNORECASE)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate.strip())
            except Exception:
                # Fall through to other strategies
                pass

        # 2) Try direct JSON parsing of the whole text
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # 3) Fallback: find the first '{' and the last '}' and try parsing that substring
        start = stripped.find('{')
        end = stripped.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except Exception:
                pass

        # Parsing failed
        return None

    def analyze_video(self, video_path: str, output_file: str = 'tennis.json') -> bool:
        """Analyze tennis video and generate shot-by-shot JSON analysis"""
        
        print(f"Starting tennis video analysis of {video_path}...")
        
        # Upload the entire video file to Gemini for proper analysis
        print(f"Uploading video file to Gemini...")
        try:
            # Upload video file using Files API
            myfile = cast(Any, genai).upload_file(video_path)
            print(f"✅ Video uploaded successfully: {myfile.name}")
            
            # Wait for video to be processed with timeout
            import time
            max_wait_time = 300  # 5 minutes timeout
            wait_time = 0
            while myfile.state.name == "PROCESSING":
                if wait_time >= max_wait_time:
                    raise RuntimeError(f"Video processing timeout after {max_wait_time} seconds")
                print(f"⏳ Video processing... ({wait_time}s)")
                time.sleep(10)
                wait_time += 10
                myfile = cast(Any, genai).get_file(myfile.name)
            
            if myfile.state.name == "FAILED":
                raise RuntimeError(f"Video processing failed: {myfile.state}")
                
        except Exception as e:
            print(f"Error uploading video: {e}")
            raise RuntimeError(f"Failed to upload video: {str(e)}")
        
        # Load prompt
        prompt = self.load_prompt()
        
        print(f"Sending video to AI for shot-by-shot analysis...")
        try:
            # Generate analysis using the uploaded video file
            response = self.model.generate_content([myfile, prompt])
            analysis_text = response.text
            
            print(f"\n[INFO] Model response length: {len(analysis_text)} characters")
            
            # Save complete model response to file
            with open('model_response_debug.txt', 'w', encoding='utf-8') as f:
                f.write(f"Model: {self.model.model_name if hasattr(self.model, 'model_name') else 'gemini-2.5-flash-lite-preview-06-17'}\n")
                f.write(f"Response length: {len(analysis_text)} characters\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("="*50 + "\n")
                f.write("FULL MODEL RESPONSE:\n")
                f.write("="*50 + "\n")
                f.write(analysis_text)
                f.write("\n" + "="*50 + "\n")
                f.write("END OF MODEL RESPONSE\n")
            
            print(f"✅ Model response saved to model_response_debug.txt")
            
            # Parse the JSON response directly
            parsed_data = self._parse_json_response(analysis_text)
            
            # Create the final result
            if parsed_data and isinstance(parsed_data, dict) and 'shots' in parsed_data:
                # Successfully parsed the tennis analysis - save only the parsed data
                result = parsed_data
                print(f"✅ Successfully parsed {len(parsed_data.get('shots', []))} shots")
                success = True
            else:
                # Fallback: save raw response if parsing failed
                result = {
                    "raw_response": analysis_text,
                    "note": "Check model_response_debug.txt for full raw response"
                }
                print(f"❌ Failed to parse JSON - saved raw response")
                success = False
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ Structured result saved to {output_file}")
            
            # Clean up uploaded file
            try:
                cast(Any, genai).delete_file(myfile.name)
                print("🗑️ Cleaned up uploaded video file")
            except Exception as e:
                print(f"Warning: Could not delete uploaded file: {e}")
            
            return success
            
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
    parser.add_argument('--local', action='store_true', help='Use existing tennis.json instead of calling Gemini')
    parser.add_argument('--prompt', default='tennis_prompt.txt', help='Custom prompt file')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return 1
    
    # If local mode, skip API and reuse existing tennis.json
    if args.local:
        source_json = 'tennis.json'
        if not os.path.exists(source_json):
            print(f"Error: Local mode requested but '{source_json}' not found in current directory")
            return 1

        # Determine output file path
        output_file = args.output if args.output else source_json
        if output_file != source_json:
            try:
                shutil.copyfile(source_json, output_file)
                print(f"✅ Copied existing analysis from {source_json} to {output_file}")
            except Exception as e:
                print(f"Error copying file: {e}")
                return 1
        else:
            print(f"✅ Using existing analysis file: {source_json}")

        return 0

    try:
        
        coach = TennisCoach(api_key=args.api_key)
        output_file = args.output if args.output else 'tennis.json'
        success = coach.analyze_video(args.video, output_file)
        if success:
            print(f"✅ Analysis complete! Model response saved to model_response_debug.txt")
        else:
            print(f"❌ Analysis failed! Check model_response_debug.txt for details")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main()) 