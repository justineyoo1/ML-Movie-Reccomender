"""
Recommendations API endpoints.

This module contains Flask routes for generating and managing movie recommendations.
"""

from flask import Blueprint, request, jsonify, current_app
from report_objects.errors.model_not_trained_error import ModelNotTrainedError
from report_objects.errors.user_not_found_error import UserNotFoundError

recommendations_bp = Blueprint('recommendations', __name__)


@recommendations_bp.route('/<int:user_id>')
def get_recommendations(user_id):
    """
    Get movie recommendations for a specific user.
    
    Args:
    ----
        user_id (int): ID of the user to get recommendations for.
        
    Query Parameters:
    ----------------
        n_recommendations (int): Number of recommendations (default: 10, max: 50)
        include_metadata (bool): Include TMDB metadata (default: true)
        model_type (str): Model to use ('svd', 'user_based', 'item_based')
        
    Returns:
    -------
        JSON response with recommendations or error message.
    """
    try:
        # Get query parameters
        n_recommendations = min(int(request.args.get('n_recommendations', 10)), 50)
        include_metadata = request.args.get('include_metadata', 'true').lower() == 'true'
        model_type = request.args.get('model_type', None)
        
        # Get recommendations from the manager
        result = current_app.recommender_manager.get_recommendations(
            user_id=user_id,
            n_recommendations=n_recommendations,
            include_metadata=include_metadata,
            model_type=model_type
        )
        
        return jsonify(result)
        
    except ModelNotTrainedError as e:
        raise e  # Will be handled by error handler
    except UserNotFoundError as e:
        raise e  # Will be handled by error handler
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid parameter: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get recommendations: {str(e)}'
        }), 500


@recommendations_bp.route('/batch', methods=['POST'])
def get_batch_recommendations():
    """
    Get recommendations for multiple users in a single request.
    
    Request Body:
    ------------
        {
            "user_ids": [1, 2, 3],
            "n_recommendations": 5,
            "include_metadata": true,
            "model_type": "svd"
        }
        
    Returns:
    -------
        JSON response with recommendations for all users.
    """
    try:
        data = request.get_json()
        
        if not data or 'user_ids' not in data:
            return jsonify({
                'status': 'error',
                'message': 'user_ids field is required'
            }), 400
        
        user_ids = data['user_ids']
        n_recommendations = min(data.get('n_recommendations', 10), 50)
        include_metadata = data.get('include_metadata', True)
        model_type = data.get('model_type', None)
        
        if not isinstance(user_ids, list) or len(user_ids) == 0:
            return jsonify({
                'status': 'error',
                'message': 'user_ids must be a non-empty list'
            }), 400
        
        if len(user_ids) > 100:  # Limit batch size
            return jsonify({
                'status': 'error',
                'message': 'Maximum 100 users per batch request'
            }), 400
        
        # Get recommendations for each user
        results = {}
        errors = {}
        
        for user_id in user_ids:
            try:
                result = current_app.recommender_manager.get_recommendations(
                    user_id=user_id,
                    n_recommendations=n_recommendations,
                    include_metadata=include_metadata,
                    model_type=model_type
                )
                results[str(user_id)] = result
            except (UserNotFoundError, ModelNotTrainedError) as e:
                errors[str(user_id)] = {
                    'error_type': type(e).__name__,
                    'message': str(e)
                }
        
        return jsonify({
            'status': 'success',
            'results': results,
            'errors': errors,
            'summary': {
                'total_users': len(user_ids),
                'successful': len(results),
                'failed': len(errors)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to process batch recommendations: {str(e)}'
        }), 500


@recommendations_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback on movie recommendations.
    
    Request Body:
    ------------
        {
            "user_id": 1,
            "movie_id": 123,
            "rating": 4.5,
            "feedback_type": "explicit"
        }
        
    Returns:
    -------
        JSON response confirming feedback submission.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body is required'
            }), 400
        
        required_fields = ['user_id', 'movie_id', 'rating']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        user_id = data['user_id']
        movie_id = data['movie_id']
        rating = float(data['rating'])
        feedback_type = data.get('feedback_type', 'explicit')
        
        # Validate rating range
        if not (0.5 <= rating <= 5.0):
            return jsonify({
                'status': 'error',
                'message': 'Rating must be between 0.5 and 5.0'
            }), 400
        
        # Record feedback
        feedback_path = current_app.recommender_manager.record_user_feedback(
            user_id=user_id,
            movie_id=movie_id,
            rating=rating
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded successfully',
            'feedback_log': feedback_path,
            'data': {
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'feedback_type': feedback_type
            }
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid data: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to record feedback: {str(e)}'
        }), 500


@recommendations_bp.route('/similar/<int:movie_id>')
def get_similar_movies(movie_id):
    """
    Get movies similar to a specific movie (requires item-based model).
    
    Args:
    ----
        movie_id (int): ID of the reference movie.
        
    Query Parameters:
    ----------------
        n_similar (int): Number of similar movies (default: 10, max: 50)
        include_metadata (bool): Include TMDB metadata (default: true)
        
    Returns:
    -------
        JSON response with similar movies.
    """
    try:
        n_similar = min(int(request.args.get('n_similar', 10)), 50)
        include_metadata = request.args.get('include_metadata', 'true').lower() == 'true'
        
        # Check if item-based model is available
        status = current_app.recommender_manager.get_system_status()
        if not status.get('model_trained'):
            raise ModelNotTrainedError("No trained model available")
        
        # Note: This is a simplified implementation
        # For a full implementation, you'd need to add similar movie functionality
        # to the RecommenderBuilder class
        
        return jsonify({
            'status': 'success',
            'movie_id': movie_id,
            'message': 'Similar movies feature coming soon - requires item-based collaborative filtering implementation'
        })
        
    except ModelNotTrainedError as e:
        raise e
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid parameter: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get similar movies: {str(e)}'
        }), 500


@recommendations_bp.route('/popular')
def get_popular_movies():
    """
    Get most popular movies (highest rated with minimum rating count).
    
    Query Parameters:
    ----------------
        n_movies (int): Number of movies (default: 20, max: 100)
        min_ratings (int): Minimum number of ratings required (default: 50)
        
    Returns:
    -------
        JSON response with popular movies.
    """
    try:
        n_movies = min(int(request.args.get('n_movies', 20)), 100)
        min_ratings = int(request.args.get('min_ratings', 50))
        
        # Check if data is loaded
        status = current_app.recommender_manager.get_system_status()
        if not status.get('data_loaded'):
            return jsonify({
                'status': 'error',
                'message': 'Data must be loaded first'
            }), 400
        
        # This would require additional implementation in the RecommenderManager
        # For now, return a placeholder response
        return jsonify({
            'status': 'success',
            'message': 'Popular movies feature coming soon',
            'parameters': {
                'n_movies': n_movies,
                'min_ratings': min_ratings
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
            'message': f'Failed to get popular movies: {str(e)}'
        }), 500 