#!/usr/bin/env python3
"""
Main Flask Application for ML Movie Recommender.

This module creates and configures the main Flask application,
initializes the recommender system, and sets up all routes and error handlers.
"""

import os
from flask import Flask, jsonify, render_template
from flask_cors import CORS

# Import configuration
from config import get_config

# Import route blueprints
from routes.recommendations import recommendations_bp
from routes.models import models_bp
from routes.user import user_bp
from routes.data import data_bp

# Import error classes for error handlers
from report_objects.errors.model_not_trained_error import ModelNotTrainedError
from report_objects.errors.user_not_found_error import UserNotFoundError
from report_objects.errors.data_validation_error import DataValidationError
from report_objects.errors.api_connection_error import APIConnectionError

# Import the recommender system
from report_objects.recommender_manager import RecommenderManager


def create_app(config_name='development'):
    """
    Create and configure the Flask application.

    Args:
    ----
        config_name (str): Configuration environment name.

    Returns:
    -------
        Flask: Configured Flask application instance.

    """
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Enable CORS for frontend integration
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://127.0.0.1:3000", "http://localhost:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Initialize recommender system
    app.recommender_manager = RecommenderManager(
        data_path=config.DATA_PATH,
        output_path=config.OUTPUT_PATH,
        tmdb_api_key=config.TMDB_API_KEY
    )
    
    # Auto-load data and train model for immediate functionality
    try:
        print("🔄 Initializing recommendation system...")
        
        # Check if data is already loaded
        status = app.recommender_manager.get_system_status()
        
        if not status['data_loaded']:
            print("📊 Loading MovieLens data...")
            app.recommender_manager.load_and_prepare_data()
            print("✅ Data loaded successfully")
        else:
            print("✅ Data already loaded")
        
        if not status['model_trained']:
            print("🧠 Training recommendation model...")
            app.recommender_manager.train_model()
            print("✅ Model trained successfully")
        else:
            print("✅ Model already trained")
            
        print("🎬 Recommendation system ready!")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize recommendation system: {e}")
        print("📝 The app will still work, but recommendations may require manual setup")
    
    # Register blueprints with the app
    from routes.recommendations import recommendations_bp
    from routes.models import models_bp
    from routes.user import user_bp
    from routes.data import data_bp
    
    app.register_blueprint(recommendations_bp, url_prefix='/api/recommendations')
    app.register_blueprint(models_bp, url_prefix='/api/models')
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(data_bp, url_prefix='/api/data')
    
    # Import and register TMDB blueprint
    from routes.tmdb import tmdb_bp
    app.register_blueprint(tmdb_bp, url_prefix='/api/tmdb')
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register general routes
    register_general_routes(app)
    
    return app


def register_error_handlers(app):
    """Register custom error handlers for the application."""
    
    @app.errorhandler(ModelNotTrainedError)
    def handle_model_not_trained(error):
        return jsonify({
            'status': 'error',
            'error_type': 'ModelNotTrainedError',
            'message': str(error),
            'suggestion': 'Train a model first using /api/models/train endpoint'
        }), 400
    
    @app.errorhandler(UserNotFoundError)
    def handle_user_not_found(error):
        return jsonify({
            'status': 'error',
            'error_type': 'UserNotFoundError',
            'message': str(error),
            'suggestion': 'Check if the user ID exists in the dataset'
        }), 404
    
    @app.errorhandler(DataValidationError)
    def handle_data_validation(error):
        return jsonify({
            'status': 'error',
            'error_type': 'DataValidationError',
            'message': str(error),
            'suggestion': 'Check data loading and configuration'
        }), 400
    
    @app.errorhandler(APIConnectionError)
    def handle_api_connection(error):
        return jsonify({
            'status': 'error',
            'error_type': 'APIConnectionError',
            'message': str(error),
            'suggestion': 'Check your internet connection and API configuration'
        }), 503
    
    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({
            'status': 'error',
            'error_type': 'NotFound',
            'message': 'The requested resource was not found',
            'suggestion': 'Check the URL and available endpoints at /api/docs'
        }), 404
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        return jsonify({
            'status': 'error',
            'error_type': 'InternalServerError',
            'message': 'An internal server error occurred',
            'suggestion': 'Check the server logs for more details'
        }), 500


def register_general_routes(app):
    """Register general application routes."""
    
    @app.route('/')
    def home():
        """Root endpoint that provides API information."""
        return jsonify({
            'name': app.config.get('APP_NAME', 'ML Movie Recommender'),
            'version': app.config.get('VERSION', '1.0.0'),
            'status': 'active',
            'description': 'AI-powered movie recommendation system',
            'documentation': '/api/docs',
            'endpoints': {
                'recommendations': '/api/recommendations',
                'models': '/api/models', 
                'users': '/api/users',
                'data': '/api/data',
                'tmdb': '/api/tmdb'
            }
        })
    
    @app.route('/api/status')
    def api_status():
        """Get current system status."""
        try:
            status = app.recommender_manager.get_system_status()
            return jsonify({
                'status': 'success',
                'data': status
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to get system status: {str(e)}'
            }), 500
    
    @app.route('/api/health')
    def health_check():
        """Health check endpoint for monitoring."""
        return jsonify({
            'status': 'healthy',
            'timestamp': str(app.recommender_manager.get_system_status().get('timestamp', 'unknown')),
            'version': app.config.get('VERSION', '1.0.0')
        })
    
    @app.route('/api/docs')
    def api_docs():
        """API documentation endpoint."""
        return jsonify({
            'title': 'ML Movie Recommender API',
            'version': app.config.get('VERSION', '1.0.0'),
            'description': 'RESTful API for movie recommendations using machine learning',
            'base_url': '/api',
            'endpoints': {
                # Recommendation endpoints
                'GET /api/recommendations/<user_id>': 'Get personalized recommendations',
                'POST /api/recommendations/batch': 'Get batch recommendations',
                'POST /api/recommendations/feedback': 'Submit user feedback',
                'GET /api/recommendations/similar/<movie_id>': 'Find similar movies',
                'GET /api/recommendations/popular': 'Get popular movies',
                
                # Model endpoints
                'POST /api/models/train': 'Train recommendation model',
                'POST /api/models/evaluate': 'Evaluate model performance',
                'POST /api/models/optimize': 'Optimize hyperparameters',
                'POST /api/models/save': 'Save trained model',
                'POST /api/models/load': 'Load saved model',
                'GET /api/models/info': 'Get model information',
                'GET /api/models/types': 'Get available model types',
                'GET /api/models/experiments': 'Get experiment history',
                
                # User endpoints
                'GET /api/users/<user_id>/profile': 'Get user profile',
                'GET /api/users/<user_id>/ratings': 'Get user ratings',
                'GET /api/users/<user_id>/stats': 'Get user statistics',
                'GET /api/users/search': 'Search users',
                'GET /api/users/random': 'Get random users',
                
                # Data endpoints
                'GET /api/data/info': 'Get dataset information',
                'POST /api/data/load': 'Load and prepare dataset',
                'POST /api/data/download/<dataset_size>': 'Download dataset',
                'GET /api/data/available': 'List available datasets',
                'GET /api/data/supported': 'List supported datasets',
                'GET /api/data/stats': 'Get detailed statistics',
                'DELETE /api/data/cleanup/<dataset_size>': 'Clean up dataset',
                'GET /api/data/sample': 'Get data sample',
                
                # TMDB endpoints
                'GET /api/tmdb/search': 'Search movies in TMDB',
                'GET /api/tmdb/movie/<tmdb_id>': 'Get movie details from TMDB',
                
                # System endpoints
                'GET /api/status': 'Get system status',
                'GET /api/health': 'Health check',
                'GET /api/docs': 'This documentation'
            },
            'authentication': 'None required',
            'rate_limiting': 'Basic rate limiting in place',
            'cors': 'Enabled for localhost:3000'
        })


if __name__ == '__main__':
    # Get configuration
    config = get_config('development')
    
    # Create application
    app = create_app('development')
    
    # Print startup information
    print("\n" + "="*50)
    print(f"🚀 Starting {config.APP_NAME} v{config.VERSION}")
    print(f"🔧 Environment: {os.getenv('FLASK_ENV', 'development')}")
    print(f"🌐 Host: {config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"📊 Data path: {config.DATA_PATH}")
    print(f"💾 Output path: {config.OUTPUT_PATH}")
    
    if config.TMDB_API_KEY:
        print("✅ TMDB API: Enabled")
    else:
        print("⚠️  TMDB API: Disabled (no API key)")
    
    print(f"📚 API Documentation: http://{config.FLASK_HOST}:{config.FLASK_PORT}/api/docs")
    print("="*50)
    
    # Run the application on port 8000 - completely different range
    app.run(
        host=config.FLASK_HOST,
        port=8000,  # Use port 8000 - totally different range
        debug=config.DEBUG
    ) 