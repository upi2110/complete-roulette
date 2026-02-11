"""
HTTP Routes - Serve dashboard and API endpoints.
"""

from flask import Blueprint, render_template, jsonify, send_from_directory

import sys
sys.path.insert(0, '.')

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'AI Roulette Predictor'})
