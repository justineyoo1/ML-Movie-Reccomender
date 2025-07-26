#!/usr/bin/env python3
"""
Streamlined Frontend Flask Application for ML Movie Recommender.

Serves the single-page Tinder-style movie rating and recommendation interface.
"""

from flask import Flask, render_template


def create_frontend_app():
    """Create the streamlined frontend Flask application."""
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Streamlined movie recommender interface."""
        return render_template('simple.html')
    
    return app


if __name__ == '__main__':
    app = create_frontend_app()
    
    # Run on port 3000 to avoid conflict with backend API
    app.run(
        host='127.0.0.1',
        port=3000,
        debug=True
    ) 