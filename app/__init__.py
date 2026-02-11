"""
Flask Application Factory with SocketIO initialization.
"""

from flask import Flask
from flask_socketio import SocketIO

import sys
sys.path.insert(0, '.')
from config import SECRET_KEY

socketio = SocketIO()


def create_app():
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')

    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable static file caching in dev

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    socketio.init_app(app, cors_allowed_origins="*", async_mode='eventlet')

    from app import socketio_handlers  # noqa: F401

    @app.after_request
    def add_no_cache_headers(response):
        """Prevent browser caching of static files during development."""
        if 'text/html' in response.content_type or 'javascript' in response.content_type or 'text/css' in response.content_type:
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

    return app
