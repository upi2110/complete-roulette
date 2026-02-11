#!/usr/bin/env python3
"""
AI European Roulette Prediction System - Entry Point
Start the Flask + SocketIO server.
"""

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HOST, PORT, DEBUG, DATA_DIR, SESSIONS_DIR, MODELS_DIR

# Create data directories
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    print("=" * 60)
    print("  AI European Roulette Prediction System v1.0")
    print("=" * 60)
    print(f"  Server:    http://localhost:{PORT}")
    print(f"  Data dir:  {DATA_DIR}")
    print(f"  Debug:     {DEBUG}")
    print("=" * 60)
    print()

    socketio.run(app, host=HOST, port=PORT, debug=DEBUG, use_reloader=False)
