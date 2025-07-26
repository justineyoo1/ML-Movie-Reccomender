"""
User API endpoints.

This module contains Flask routes for managing user profiles and rating history.
"""

from flask import Blueprint, request, jsonify, current_app
from report_objects.errors.user_not_found_error import UserNotFoundError
from report_objects.errors.data_validation_error import DataValidationError

user_bp = Blueprint('user', __name__)


@user_bp.route('/<int:user_id>/profile')
def get_user_profile(user_id):
    """
    Get user profile information.
    
    Args:
    ----
        user_id (int): ID of the user.
        
    Returns:
    -------
        JSON response with user profile data.
    """
    try:
        profile = current_app.recommender_manager.get_user_profile(user_id)
        
        return jsonify({
            'status': 'success',
            'user_profile': profile
        })
        
    except UserNotFoundError as e:
        raise e
    except DataValidationError as e:
        raise e
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get user profile: {str(e)}'
        }), 500


@user_bp.route('/<int:user_id>/ratings')
def get_user_ratings(user_id):
    """
    Get user's rating history.
    
    Args:
    ----
        user_id (int): ID of the user.
        
    Query Parameters:
    ----------------
        limit (int): Maximum number of ratings to return (default: 50)
        sort_by (str): Sort by 'rating', 'timestamp', or 'title' (default: 'timestamp')
        order (str): 'asc' or 'desc' (default: 'desc')
        
    Returns:
    -------
        JSON response with user's ratings.
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 500)
        sort_by = request.args.get('sort_by', 'timestamp')
        order = request.args.get('order', 'desc')
        
        # Validate parameters
        valid_sort_fields = ['rating', 'timestamp', 'title']
        if sort_by not in valid_sort_fields:
            return jsonify({
                'status': 'error',
                'message': f'Invalid sort_by. Must be one of: {valid_sort_fields}'
            }), 400
        
        if order not in ['asc', 'desc']:
            return jsonify({
                'status': 'error',
                'message': 'order must be "asc" or "desc"'
            }), 400
        
        # Get user ratings
        user_ratings = current_app.recommender_manager.data_reader.get_user_ratings(user_id)
        
        if user_ratings.empty:
            raise UserNotFoundError(f"User {user_id} not found in the dataset")
        
        # Sort the ratings
        ascending = (order == 'asc')
        if sort_by == 'timestamp':
            user_ratings = user_ratings.sort_values('timestamp', ascending=ascending)
        elif sort_by == 'rating':
            user_ratings = user_ratings.sort_values('rating', ascending=ascending)
        elif sort_by == 'title':
            user_ratings = user_ratings.sort_values('title', ascending=ascending)
        
        # Limit results
        limited_ratings = user_ratings.head(limit)
        
        # Convert to dict format
        ratings_list = limited_ratings.to_dict('records')
        
        # Calculate statistics
        stats = {
            'total_ratings': len(user_ratings),
            'average_rating': float(user_ratings['rating'].mean()),
            'min_rating': float(user_ratings['rating'].min()),
            'max_rating': float(user_ratings['rating'].max()),
            'rating_distribution': user_ratings['rating'].value_counts().to_dict()
        }
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'ratings': ratings_list,
            'statistics': stats,
            'query_info': {
                'limit': limit,
                'sort_by': sort_by,
                'order': order,
                'showing': len(ratings_list),
                'total_available': len(user_ratings)
            }
        })
        
    except UserNotFoundError as e:
        raise e
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid parameter: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get user ratings: {str(e)}'
        }), 500


@user_bp.route('/<int:user_id>/stats')
def get_user_stats(user_id):
    """
    Get detailed statistics for a user.
    
    Args:
    ----
        user_id (int): ID of the user.
        
    Returns:
    -------
        JSON response with detailed user statistics.
    """
    try:
        # Get user ratings
        user_ratings = current_app.recommender_manager.data_reader.get_user_ratings(user_id)
        
        if user_ratings.empty:
            raise UserNotFoundError(f"User {user_id} not found in the dataset")
        
        # Calculate detailed statistics
        stats = {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'unique_movies_rated': user_ratings['movie_id'].nunique(),
            'rating_stats': {
                'mean': float(user_ratings['rating'].mean()),
                'median': float(user_ratings['rating'].median()),
                'std': float(user_ratings['rating'].std()),
                'min': float(user_ratings['rating'].min()),
                'max': float(user_ratings['rating'].max())
            },
            'rating_distribution': user_ratings['rating'].value_counts().sort_index().to_dict(),
            'activity_over_time': {}
        }
        
        # Genre preferences (if available)
        if 'genres' in user_ratings.columns and not user_ratings['genres'].isna().all():
            # Parse genres and count preferences
            genre_counts = {}
            for genres_str in user_ratings['genres'].dropna():
                if isinstance(genres_str, str):
                    genres = genres_str.split('|')
                    for genre in genres:
                        genre = genre.strip()
                        if genre:
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Sort by count and get top genres
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            stats['favorite_genres'] = dict(top_genres)
        
        # Recent activity
        if 'timestamp' in user_ratings.columns:
            recent_ratings = user_ratings.sort_values('timestamp', ascending=False).head(5)
            stats['recent_activity'] = recent_ratings[['movie_id', 'title', 'rating', 'timestamp']].to_dict('records')
        
        return jsonify({
            'status': 'success',
            'user_statistics': stats
        })
        
    except UserNotFoundError as e:
        raise e
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get user statistics: {str(e)}'
        }), 500


@user_bp.route('/search')
def search_users():
    """
    Search for users based on criteria.
    
    Query Parameters:
    ----------------
        min_ratings (int): Minimum number of ratings (default: 1)
        max_ratings (int): Maximum number of ratings (optional)
        avg_rating_min (float): Minimum average rating (optional)
        avg_rating_max (float): Maximum average rating (optional)
        limit (int): Maximum users to return (default: 50, max: 500)
        
    Returns:
    -------
        JSON response with matching users.
    """
    try:
        min_ratings = int(request.args.get('min_ratings', 1))
        max_ratings = request.args.get('max_ratings')
        avg_rating_min = request.args.get('avg_rating_min')
        avg_rating_max = request.args.get('avg_rating_max')
        limit = min(int(request.args.get('limit', 50)), 500)
        
        # Get all ratings data
        if not current_app.recommender_manager.data_loaded:
            return jsonify({
                'status': 'error',
                'message': 'Data must be loaded first'
            }), 400
        
        ratings_data = current_app.recommender_manager.ratings_data
        
        # Calculate user statistics
        user_stats = ratings_data.groupby('user_id').agg({
            'rating': ['count', 'mean'],
            'movie_id': 'nunique'
        }).round(2)
        
        user_stats.columns = ['num_ratings', 'avg_rating', 'unique_movies']
        user_stats = user_stats.reset_index()
        
        # Apply filters
        filtered_users = user_stats[user_stats['num_ratings'] >= min_ratings]
        
        if max_ratings:
            filtered_users = filtered_users[filtered_users['num_ratings'] <= int(max_ratings)]
        
        if avg_rating_min:
            filtered_users = filtered_users[filtered_users['avg_rating'] >= float(avg_rating_min)]
        
        if avg_rating_max:
            filtered_users = filtered_users[filtered_users['avg_rating'] <= float(avg_rating_max)]
        
        # Sort by number of ratings (descending) and limit
        filtered_users = filtered_users.sort_values('num_ratings', ascending=False).head(limit)
        
        # Convert to list
        users_list = filtered_users.to_dict('records')
        
        return jsonify({
            'status': 'success',
            'users': users_list,
            'search_criteria': {
                'min_ratings': min_ratings,
                'max_ratings': max_ratings,
                'avg_rating_min': avg_rating_min,
                'avg_rating_max': avg_rating_max,
                'limit': limit
            },
            'total_found': len(filtered_users),
            'total_users_in_dataset': len(user_stats)
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid parameter: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to search users: {str(e)}'
        }), 500


@user_bp.route('/random')
def get_random_users():
    """
    Get random users from the dataset.
    
    Query Parameters:
    ----------------
        count (int): Number of random users to return (default: 5, max: 50)
        min_ratings (int): Minimum ratings per user (default: 10)
        
    Returns:
    -------
        JSON response with random users.
    """
    try:
        count = min(int(request.args.get('count', 5)), 50)
        min_ratings = int(request.args.get('min_ratings', 10))
        
        # Get user statistics
        if not current_app.recommender_manager.data_loaded:
            return jsonify({
                'status': 'error',
                'message': 'Data must be loaded first'
            }), 400
        
        ratings_data = current_app.recommender_manager.ratings_data
        user_counts = ratings_data['user_id'].value_counts()
        
        # Filter users with minimum ratings
        eligible_users = user_counts[user_counts >= min_ratings].index.tolist()
        
        if len(eligible_users) == 0:
            return jsonify({
                'status': 'error',
                'message': f'No users found with at least {min_ratings} ratings'
            }), 404
        
        # Sample random users
        import random
        sample_size = min(count, len(eligible_users))
        random_user_ids = random.sample(eligible_users, sample_size)
        
        # Get basic info for each user
        random_users = []
        for user_id in random_user_ids:
            user_data = {
                'user_id': int(user_id),
                'num_ratings': int(user_counts[user_id])
            }
            random_users.append(user_data)
        
        return jsonify({
            'status': 'success',
            'random_users': random_users,
            'criteria': {
                'count_requested': count,
                'min_ratings': min_ratings,
                'count_returned': len(random_users)
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
            'message': f'Failed to get random users: {str(e)}'
        }), 500 