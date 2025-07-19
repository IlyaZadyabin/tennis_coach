#!/usr/bin/env python3
"""
Tennis AI Coach Demo
Demonstrates the complete tennis analysis pipeline
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import cv2
        import numpy as np
        import google.generativeai as genai
        import mediapipe as mp
        _modules = (cv2, np, genai, mp)  # Prevent unused-import warnings
        del _modules
        print("✅ All requirements are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if Gemini API key is set"""
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("✅ Gemini API key found")
        return True
    else:
        print("❌ Gemini API key not found")
        print("Set environment variable: export GEMINI_API_KEY='your-key-here'")
        print("Get your key from: https://makersuite.google.com/app/apikey")
        return False

def run_analysis(video_path):
    """Run the tennis analysis pipeline"""
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print(f"\n🎾 Starting tennis analysis for: {video_path}")
    
    # Step 1: Generate analysis (creates tennis.json)
    print("\n📊 Step 1: Generating AI analysis...")
    result = subprocess.run([
        sys.executable, 'tennis_coach.py', video_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Analysis failed: {result.stderr}")
        return False
    
    print("✅ Analysis complete - tennis.json created")
    
    # Step 2: Create visualization (creates tennis_analysis_output.mp4)
    if Path('tennis.json').exists():
        print("\n🎥 Step 2: Creating video visualization...")
        result = subprocess.run([
            sys.executable, 'tennis_visualizer.py', video_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Visualization failed: {result.stderr}")
            return False
        
        print("✅ Video visualization complete - tennis_analysis_output.mp4 created")
        return True
    else:
        print("❌ tennis.json not found - analysis may have failed")
        return False

def main():
    """Main demo function"""
    print("🎾 Tennis AI Coach Demo")
    print("=" * 40)
    
    # Check setup
    if not check_requirements():
        return 1
    
    if not check_api_key():
        return 1
    
    # Get video file
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("\nUsage: python run_demo.py <tennis_video.mp4>")
        print("\nExample:")
        print("  python run_demo.py my_tennis_match.mp4")
        return 1
    
    # Run the complete pipeline
    success = run_analysis(video_path)
    
    if success:
        print("\n🎉 Demo Complete!")
        print("\nFiles created:")
        print("  📄 tennis.json - Shot-by-shot analysis (THE MAGIC)")
        print("  🎥 tennis_analysis_output.mp4 - Video with overlays")
        print("\nNext steps:")
        print("  1. Open tennis.json to see detailed shot analysis")
        print("  2. Play tennis_analysis_output.mp4 to see the visualization")
        print("  3. Use the feedback to improve your tennis game!")
        return 0
    else:
        print("\n❌ Demo failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main()) 