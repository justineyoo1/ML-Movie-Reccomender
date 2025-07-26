#!/usr/bin/env python3
"""
🎬 Movie Recommender - Easy Start Script

This script starts both the backend API and frontend interface
for the streamlined movie recommendation experience.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print startup banner."""
    print("\n" + "="*60)
    print("MOVIE RECOMMENDER - STREAMLINED VERSION")
    print("="*60)
    print("✨ Simple, clean movie recommendations just for you!")
    print("🎯 Rate movies → Get personal recommendations")
    print("="*60 + "\n")

def check_data():
    """Check if data is loaded."""
    data_dir = Path("data/ml-100k")
    if not data_dir.exists():
        print("📁 Setting up movie data...")
        print("   This will download the MovieLens dataset (first time only)")
        subprocess.run([sys.executable, "data/setup.py", "download", "100k"], 
                      capture_output=True)
        print("✅ Data ready!")
    else:
        print("✅ Movie data already available")

def start_backend():
    """Start the backend API server."""
    print("🚀 Starting backend API server...")
    backend = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return backend

def start_frontend():
    """Start the frontend server."""
    print("🌐 Starting frontend interface...")
    frontend = subprocess.Popen(
        [sys.executable, "frontend_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return frontend

def main():
    """Main startup function."""
    print_banner()
    
    try:
        # Check data availability
        check_data()
        
        # Start backend
        backend = start_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start frontend
        frontend = start_frontend()
        time.sleep(2)  # Give frontend time to start
        
        print("\n🎉 READY TO USE!")
        print("="*40)
        print("🌐 Open: http://127.0.0.1:3000")
        print("📱 Mobile-friendly responsive design")
        print("⚡ Fast, streamlined experience")
        print("="*40)
        print("\n💡 QUICK START:")
        print("   1. Rate 5 movies you've seen")
        print("   2. Click 'Get My Recommendations'")
        print("   3. Discover your next favorite movie!")
        print("\n🔧 To stop: Press Ctrl+C\n")
        
        # Auto-open browser after a moment
        time.sleep(1)
        try:
            webbrowser.open("http://127.0.0.1:3000")
            print("🚀 Opening browser automatically...")
        except:
            print("💻 Please open http://127.0.0.1:3000 in your browser")
        
        # Wait for user to stop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping servers...")
            backend.terminate()
            frontend.terminate()
            print("✅ All servers stopped. Thanks for using Movie Recommender!")
            
    except Exception as e:
        print(f"\n❌ Error starting servers: {e}")
        print("💡 Try running manually:")
        print("   Terminal 1: python app.py")
        print("   Terminal 2: python frontend_app.py")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 