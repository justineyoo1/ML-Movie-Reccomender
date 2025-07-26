"""
Configuration settings for ML Movie Recommender System.

This module contains all configuration parameters for the application,
including paths, API settings, model parameters, and Flask settings.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration class with default settings."""
    
    # Application Settings
    APP_NAME = "ML Movie Recommender"
    VERSION = "1.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5001))
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.getenv('DATA_PATH', os.path.join(BASE_DIR, 'data'))
    OUTPUT_PATH = os.getenv('OUTPUT_PATH', os.path.join(BASE_DIR, 'outputs'))
    MODELS_PATH = os.path.join(OUTPUT_PATH, 'models')
    LOGS_PATH = os.path.join(OUTPUT_PATH, 'logs')
    RESULTS_PATH = os.path.join(OUTPUT_PATH, 'results')
    
    # TMDB API Settings
    TMDB_API_KEY = os.getenv('TMDB_API_KEY')
    TMDB_ACCESS_TOKEN = os.getenv('TMDB_ACCESS_TOKEN')
    TMDB_BASE_URL = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"
    
    # Dataset Settings
    DEFAULT_DATASET_SIZE = os.getenv('DEFAULT_DATASET_SIZE', '100k')
    SUPPORTED_DATASETS = ['100k', '1m', '10m', '20m', '25m']
    
    # Model Settings
    DEFAULT_MODEL_TYPE = os.getenv('DEFAULT_MODEL_TYPE', 'svd')
    SUPPORTED_MODELS = ['svd', 'user_based', 'item_based']
    
    # SVD Model Parameters
    SVD_N_COMPONENTS = int(os.getenv('SVD_N_COMPONENTS', 50))
    SVD_RANDOM_STATE = int(os.getenv('SVD_RANDOM_STATE', 42))
    
    # Collaborative Filtering Parameters
    SIMILARITY_METRIC = os.getenv('SIMILARITY_METRIC', 'cosine')
    TOP_K_NEIGHBORS = int(os.getenv('TOP_K_NEIGHBORS', 50))
    
    # Recommendation Settings
    DEFAULT_N_RECOMMENDATIONS = int(os.getenv('DEFAULT_N_RECOMMENDATIONS', 10))
    MAX_N_RECOMMENDATIONS = int(os.getenv('MAX_N_RECOMMENDATIONS', 50))
    INCLUDE_METADATA_DEFAULT = os.getenv('INCLUDE_METADATA_DEFAULT', 'True').lower() == 'true'
    
    # Data Processing
    TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
    MIN_RATINGS_PER_USER = int(os.getenv('MIN_RATINGS_PER_USER', 5))
    MIN_RATINGS_PER_MOVIE = int(os.getenv('MIN_RATINGS_PER_MOVIE', 5))
    
    # Hyperparameter Optimization
    OPTUNA_N_TRIALS = int(os.getenv('OPTUNA_N_TRIALS', 50))
    OPTUNA_TIMEOUT = int(os.getenv('OPTUNA_TIMEOUT', 300))  # 5 minutes
    
    # File Cleanup
    LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', 30))
    
    # Rate Limiting (for API endpoints)
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 3600))  # 1 hour
    
    # Caching
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 300))  # 5 minutes
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate configuration settings and return validation results.
        
        Returns:
        -------
            Dict[str, Any]: Validation results with any issues found.
        """
        issues = []
        warnings = []
        
        # Check required directories
        required_dirs = [cls.DATA_PATH, cls.OUTPUT_PATH]
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                warnings.append(f"Directory does not exist: {dir_path}")
        
        # Check TMDB API key
        if not cls.TMDB_API_KEY:
            warnings.append("TMDB_API_KEY not set - movie metadata features will be disabled")
        
        # Validate dataset size
        if cls.DEFAULT_DATASET_SIZE not in cls.SUPPORTED_DATASETS:
            issues.append(f"Invalid DEFAULT_DATASET_SIZE: {cls.DEFAULT_DATASET_SIZE}")
        
        # Validate model type
        if cls.DEFAULT_MODEL_TYPE not in cls.SUPPORTED_MODELS:
            issues.append(f"Invalid DEFAULT_MODEL_TYPE: {cls.DEFAULT_MODEL_TYPE}")
        
        # Validate numeric ranges
        if not (0.1 <= cls.TEST_SIZE <= 0.5):
            issues.append(f"TEST_SIZE should be between 0.1 and 0.5, got {cls.TEST_SIZE}")
        
        if cls.SVD_N_COMPONENTS < 1:
            issues.append(f"SVD_N_COMPONENTS must be positive, got {cls.SVD_N_COMPONENTS}")
        
        if cls.DEFAULT_N_RECOMMENDATIONS < 1:
            issues.append(f"DEFAULT_N_RECOMMENDATIONS must be positive, got {cls.DEFAULT_N_RECOMMENDATIONS}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_PATH,
            cls.OUTPUT_PATH,
            cls.MODELS_PATH,
            cls.LOGS_PATH,
            cls.RESULTS_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_movielens_url(cls, dataset_size: str) -> str:
        """
        Get the download URL for a specific MovieLens dataset.
        
        Args:
        ----
            dataset_size (str): Size of the dataset ('100k', '1m', etc.).
            
        Returns:
        -------
            str: Download URL for the dataset.
        """
        urls = {
            '100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
            '1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            '10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
            '20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip',
            '25m': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
        }
        
        return urls.get(dataset_size, urls['100k'])


class DevelopmentConfig(Config):
    """Development configuration with debugging enabled."""
    DEBUG = True
    FLASK_HOST = '127.0.0.1'
    FLASK_PORT = 5000


class ProductionConfig(Config):
    """Production configuration with security settings."""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = int(os.getenv('PORT', 5000))
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class TestingConfig(Config):
    """Testing configuration for unit tests."""
    TESTING = True
    DEBUG = True
    # Use in-memory or temporary directories for testing
    DATA_PATH = '/tmp/test_data'
    OUTPUT_PATH = '/tmp/test_outputs'


# Configuration factory
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name: str = None) -> Config:
    """
    Get configuration object based on environment.
    
    Args:
    ----
        config_name (str): Name of configuration to use.
        
    Returns:
    -------
        Config: Configuration object.
    """
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    return config_by_name.get(config_name, DevelopmentConfig) 