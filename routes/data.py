"""
Data API endpoints.

This module contains Flask routes for data management and dataset operations.
"""

from flask import Blueprint, request, jsonify, current_app
from report_objects.errors.data_validation_error import DataValidationError
from data.setup import DataSetup

data_bp = Blueprint('data', __name__)


@data_bp.route('/info')
def get_data_info():
    """
    Get information about the currently loaded dataset.
    
    Returns:
    -------
        JSON response with dataset information.
    """
    try:
        status = current_app.recommender_manager.get_system_status()
        
        if not status.get('data_loaded'):
            return jsonify({
                'status': 'warning',
                'message': 'No data currently loaded',
                'data_loaded': False
            })
        
        data_stats = status.get('data_stats', {})
        
        return jsonify({
            'status': 'success',
            'data_loaded': True,
            'dataset_info': data_stats
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get data info: {str(e)}'
        }), 500


@data_bp.route('/load', methods=['POST'])
def load_data():
    """
    Load and prepare a MovieLens dataset.
    
    Request Body:
    ------------
        {
            "dataset_size": "100k",
            "test_size": 0.2
        }
        
    Returns:
    -------
        JSON response with data loading results.
    """
    try:
        data = request.get_json() or {}
        
        dataset_size = data.get('dataset_size', '100k')
        test_size = data.get('test_size', 0.2)
        
        # Validate parameters
        valid_sizes = ['100k', '1m', '10m', '20m', '25m']
        if dataset_size not in valid_sizes:
            return jsonify({
                'status': 'error',
                'message': f'Invalid dataset_size. Must be one of: {valid_sizes}'
            }), 400
        
        if not (0.1 <= test_size <= 0.5):
            return jsonify({
                'status': 'error',
                'message': 'test_size must be between 0.1 and 0.5'
            }), 400
        
        # Load and prepare data
        result = current_app.recommender_manager.load_and_prepare_data(
            dataset_size=dataset_size,
            test_size=test_size
        )
        
        return jsonify({
            'status': 'success',
            'message': f'Dataset {dataset_size} loaded successfully',
            'load_results': result
        })
        
    except DataValidationError as e:
        raise e
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Data loading failed: {str(e)}'
        }), 500


@data_bp.route('/download/<dataset_size>')
def download_dataset(dataset_size):
    """
    Download a MovieLens dataset.
    
    Args:
    ----
        dataset_size (str): Size of dataset to download.
        
    Query Parameters:
    ----------------
        force (bool): Force re-download if dataset exists (default: false)
        
    Returns:
    -------
        JSON response with download results.
    """
    try:
        force_download = request.args.get('force', 'false').lower() == 'true'
        
        # Validate dataset size
        valid_sizes = ['100k', '1m', '10m', '20m', '25m']
        if dataset_size not in valid_sizes:
            return jsonify({
                'status': 'error',
                'message': f'Invalid dataset_size. Must be one of: {valid_sizes}'
            }), 400
        
        # Use DataSetup to download
        data_setup = DataSetup()
        success = data_setup.download_dataset(dataset_size, force_download)
        
        if success:
            # Get dataset info
            info = data_setup.get_dataset_info(dataset_size)
            
            return jsonify({
                'status': 'success',
                'message': f'Dataset {dataset_size} downloaded successfully',
                'dataset_info': info
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to download dataset {dataset_size}'
            }), 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Download failed: {str(e)}'
        }), 500


@data_bp.route('/available')
def list_available_datasets():
    """
    List all available datasets in the data directory.
    
    Returns:
    -------
        JSON response with available datasets.
    """
    try:
        data_setup = DataSetup()
        available_datasets = data_setup.list_available_datasets()
        
        # Get info for each available dataset
        datasets_info = {}
        for dataset_size in available_datasets:
            info = data_setup.get_dataset_info(dataset_size)
            datasets_info[dataset_size] = info
        
        return jsonify({
            'status': 'success',
            'available_datasets': available_datasets,
            'datasets_info': datasets_info,
            'total_available': len(available_datasets)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to list datasets: {str(e)}'
        }), 500


@data_bp.route('/supported')
def get_supported_datasets():
    """
    Get information about all supported MovieLens datasets.
    
    Returns:
    -------
        JSON response with supported dataset information.
    """
    supported_info = {
        '100k': {
            'name': 'MovieLens 100K',
            'description': '100,000 ratings from 943 users on 1,682 movies',
            'size': '~5MB',
            'users': 943,
            'movies': 1682,
            'ratings': 100000,
            'sparsity': '~93.7%',
            'good_for': ['Learning', 'Quick experiments', 'Development']
        },
        '1m': {
            'name': 'MovieLens 1M',
            'description': '1 million ratings from 6,040 users on 3,706 movies',
            'size': '~25MB',
            'users': 6040,
            'movies': 3706,
            'ratings': 1000209,
            'sparsity': '~95.5%',
            'good_for': ['Small to medium experiments', 'Algorithm testing']
        },
        '10m': {
            'name': 'MovieLens 10M',
            'description': '10 million ratings from 71,567 users on 10,681 movies',
            'size': '~265MB',
            'users': 71567,
            'movies': 10681,
            'ratings': 10000054,
            'sparsity': '~98.7%',
            'good_for': ['Medium scale experiments', 'Performance testing']
        },
        '20m': {
            'name': 'MovieLens 20M',
            'description': '20 million ratings from 138,493 users on 27,278 movies',
            'size': '~520MB',
            'users': 138493,
            'movies': 27278,
            'ratings': 20000263,
            'sparsity': '~99.5%',
            'good_for': ['Large scale experiments', 'Production testing']
        },
        '25m': {
            'name': 'MovieLens 25M',
            'description': '25 million ratings from 162,541 users on 62,423 movies',
            'size': '~800MB',
            'users': 162541,
            'movies': 62423,
            'ratings': 25000095,
            'sparsity': '~99.8%',
            'good_for': ['Very large experiments', 'Scalability testing']
        }
    }
    
    return jsonify({
        'status': 'success',
        'supported_datasets': supported_info,
        'default_dataset': '100k',
        'recommendation': 'Start with 100k for development, use 1m+ for serious experiments'
    })


@data_bp.route('/stats')
def get_dataset_stats():
    """
    Get detailed statistics about the current dataset.
    
    Returns:
    -------
        JSON response with detailed dataset statistics.
    """
    try:
        # Check if data is loaded
        if not current_app.recommender_manager.data_loaded:
            return jsonify({
                'status': 'error',
                'message': 'No data currently loaded'
            }), 400
        
        ratings_data = current_app.recommender_manager.ratings_data
        movies_data = current_app.recommender_manager.movies_data
        
        # Calculate detailed statistics
        stats = {
            'basic_stats': {
                'total_ratings': len(ratings_data),
                'unique_users': ratings_data['user_id'].nunique(),
                'unique_movies': ratings_data['movie_id'].nunique(),
                'rating_scale': {
                    'min': float(ratings_data['rating'].min()),
                    'max': float(ratings_data['rating'].max())
                },
                'average_rating': float(ratings_data['rating'].mean()),
                'median_rating': float(ratings_data['rating'].median())
            },
            'rating_distribution': ratings_data['rating'].value_counts().sort_index().to_dict(),
            'sparsity': 1 - (len(ratings_data) / (ratings_data['user_id'].nunique() * ratings_data['movie_id'].nunique())),
            'user_stats': {
                'avg_ratings_per_user': float(ratings_data['user_id'].value_counts().mean()),
                'median_ratings_per_user': float(ratings_data['user_id'].value_counts().median()),
                'max_ratings_per_user': int(ratings_data['user_id'].value_counts().max()),
                'min_ratings_per_user': int(ratings_data['user_id'].value_counts().min())
            },
            'movie_stats': {
                'avg_ratings_per_movie': float(ratings_data['movie_id'].value_counts().mean()),
                'median_ratings_per_movie': float(ratings_data['movie_id'].value_counts().median()),
                'max_ratings_per_movie': int(ratings_data['movie_id'].value_counts().max()),
                'min_ratings_per_movie': int(ratings_data['movie_id'].value_counts().min())
            }
        }
        
        # Add genre information if available
        if movies_data is not None and 'genres' in movies_data.columns:
            genre_counts = {}
            for genres_str in movies_data['genres'].dropna():
                if isinstance(genres_str, str):
                    genres = genres_str.split('|')
                    for genre in genres:
                        genre = genre.strip()
                        if genre:
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            stats['genre_distribution'] = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'status': 'success',
            'dataset_statistics': stats
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get dataset statistics: {str(e)}'
        }), 500


@data_bp.route('/cleanup/<dataset_size>', methods=['DELETE'])
def cleanup_dataset(dataset_size):
    """
    Remove a dataset from the data directory.
    
    Args:
    ----
        dataset_size (str): Size of dataset to remove.
        
    Returns:
    -------
        JSON response with cleanup confirmation.
    """
    try:
        # Validate dataset size
        valid_sizes = ['100k', '1m', '10m', '20m', '25m']
        if dataset_size not in valid_sizes:
            return jsonify({
                'status': 'error',
                'message': f'Invalid dataset_size. Must be one of: {valid_sizes}'
            }), 400
        
        # Check if this is the currently loaded dataset
        status = current_app.recommender_manager.get_system_status()
        if status.get('data_loaded'):
            # This is a simplified check - in a full implementation,
            # you'd want to track which dataset is currently loaded
            return jsonify({
                'status': 'warning',
                'message': f'Cannot delete dataset {dataset_size} - data is currently loaded. Reload different data first.'
            }), 400
        
        # Use DataSetup to cleanup
        data_setup = DataSetup()
        success = data_setup.cleanup_dataset(dataset_size)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Dataset {dataset_size} removed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to remove dataset {dataset_size}'
            }), 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Cleanup failed: {str(e)}'
        }), 500


@data_bp.route('/sample')
def get_data_sample():
    """
    Get a sample of the loaded dataset for inspection.
    
    Query Parameters:
    ----------------
        n_ratings (int): Number of rating samples (default: 10)
        n_movies (int): Number of movie samples (default: 5)
        n_users (int): Number of user samples (default: 5)
        
    Returns:
    -------
        JSON response with data samples.
    """
    try:
        n_ratings = min(int(request.args.get('n_ratings', 10)), 100)
        n_movies = min(int(request.args.get('n_movies', 5)), 50)
        n_users = min(int(request.args.get('n_users', 5)), 50)
        
        # Check if data is loaded
        if not current_app.recommender_manager.data_loaded:
            return jsonify({
                'status': 'error',
                'message': 'No data currently loaded'
            }), 400
        
        sample_data = {}
        
        # Sample ratings
        ratings_data = current_app.recommender_manager.ratings_data
        sample_data['ratings_sample'] = ratings_data.sample(n=min(n_ratings, len(ratings_data))).to_dict('records')
        
        # Sample movies
        movies_data = current_app.recommender_manager.movies_data
        if movies_data is not None:
            sample_data['movies_sample'] = movies_data.sample(n=min(n_movies, len(movies_data))).to_dict('records')
        
        # Sample users (if available)
        users_data = current_app.recommender_manager.users_data
        if users_data is not None:
            sample_data['users_sample'] = users_data.sample(n=min(n_users, len(users_data))).to_dict('records')
        
        return jsonify({
            'status': 'success',
            'data_samples': sample_data,
            'sample_sizes': {
                'ratings': len(sample_data['ratings_sample']),
                'movies': len(sample_data.get('movies_sample', [])),
                'users': len(sample_data.get('users_sample', []))
            }
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid parameter: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get data sample: {str(e)}'
        }), 500 