#!/usr/bin/env python3
"""
TMDB (The Movie Database) API routes.

Handles movie metadata fetching, poster URLs, and TMDB integration.
"""

from flask import Blueprint, jsonify, request, current_app
import requests
import os
from typing import Dict, Any, Optional

from report_objects.errors.api_connection_error import APIConnectionError
# from utils import validate_required_fields, format_api_response  # Not needed for basic functionality

# Create TMDB blueprint
tmdb_bp = Blueprint('tmdb', __name__)


@tmdb_bp.route('/search', methods=['GET'])
def search_movie():
    """
    Search for a movie in TMDB and return metadata.
    
    Query Parameters:
    ----------------
        title (str): Movie title to search for.
        year (str, optional): Release year to improve search accuracy.
        
    Returns:
    -------
        JSON response with movie metadata including poster URL and TMDB ID.
    """
    try:
        title = request.args.get('title')
        year = request.args.get('year')
        
        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Movie title is required'
            }), 400
        
        # Get TMDB credentials from config
        tmdb_access_token = current_app.config.get('TMDB_ACCESS_TOKEN')
        tmdb_api_key = current_app.config.get('TMDB_API_KEY')
        
        if not tmdb_access_token and not tmdb_api_key:
            return jsonify({
                'status': 'error',
                'message': 'TMDB API not configured'
            }), 503
        
        # Search for movie
        metadata = search_tmdb_movie(title, year)
        
        if metadata:
            return jsonify({
                'status': 'success',
                'data': metadata,
                **metadata  # Flatten for easier access
            })
        else:
            return jsonify({
                'status': 'not_found',
                'message': f'Movie "{title}" not found in TMDB'
            }), 404
            
    except APIConnectionError as e:
        return jsonify({
            'status': 'error',
            'message': f'TMDB API error: {str(e)}'
        }), 502
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }), 500


@tmdb_bp.route('/movie/<int:tmdb_id>', methods=['GET'])
def get_movie_details(tmdb_id: int):
    """
    Get detailed movie information from TMDB.
    
    Path Parameters:
    ---------------
        tmdb_id (int): TMDB movie ID.
        
    Returns:
    -------
        JSON response with detailed movie information.
    """
    try:
        tmdb_api_key = current_app.config.get('TMDB_API_KEY')
        
        if not tmdb_api_key:
            return jsonify({
                'status': 'error',
                'message': 'TMDB API not configured'
            }), 503
        
        # Get movie details
        details = get_tmdb_movie_details(tmdb_id, tmdb_api_key)
        
        if details:
            return jsonify({
                'status': 'success',
                'data': details
            })
        else:
            return jsonify({
                'status': 'not_found',
                'message': f'Movie with TMDB ID {tmdb_id} not found'
            }), 404
            
    except APIConnectionError as e:
        return jsonify({
            'status': 'error',
            'message': f'TMDB API error: {str(e)}'
        }), 502
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }), 500


def search_tmdb_movie(title: str, year: Optional[str] = None, api_key: str = None) -> Optional[Dict[str, Any]]:
    """
    Search for a movie in TMDB and return metadata.
    
    Args:
    ----
        title (str): Movie title to search for.
        year (Optional[str]): Release year to improve search accuracy.
        api_key (str): TMDB API key.
        
    Returns:
    -------
        Optional[Dict]: Movie metadata or None if not found.
        
    Raises:
    ------
        APIConnectionError: If TMDB API request fails.
    """
    try:
        # Clean up title for better search results
        clean_title = clean_movie_title(title)
        
        # Get credentials from Flask config
        access_token = current_app.config.get('TMDB_ACCESS_TOKEN')
        api_key = current_app.config.get('TMDB_API_KEY')
        
        if not access_token and not api_key:
            raise APIConnectionError("TMDB API credentials not configured")
        
        # Build search URL
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            'query': clean_title
        }
        
        # Set up authentication
        headers = {}
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'
        else:
            params['api_key'] = api_key
        
        # Add year filter if provided
        if year:
            params['year'] = year
        
        # Make API request
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('results'):
            movie = data['results'][0]  # Get first result
            
            # Build metadata response
            metadata = {
                'tmdb_id': movie.get('id'),
                'title': movie.get('title'),
                'original_title': movie.get('original_title'),
                'overview': movie.get('overview'),
                'release_date': movie.get('release_date'),
                'vote_average': movie.get('vote_average'),
                'vote_count': movie.get('vote_count'),
                'genre_ids': movie.get('genre_ids', []),
                'poster_path': movie.get('poster_path'),
                'backdrop_path': movie.get('backdrop_path')
            }
            
            # Build full poster URL if available
            if metadata['poster_path']:
                metadata['poster_url'] = f"https://image.tmdb.org/t/p/original{metadata['poster_path']}"
            
            # Build full backdrop URL if available
            if metadata['backdrop_path']:
                metadata['backdrop_url'] = f"https://image.tmdb.org/t/p/w1280{metadata['backdrop_path']}"
            
            # Build TMDB movie page URL
            metadata['tmdb_url'] = f"https://www.themoviedb.org/movie/{metadata['tmdb_id']}"
            
            return metadata
        
        return None
        
    except requests.RequestException as e:
        raise APIConnectionError(f"Failed to search TMDB: {str(e)}")


def get_tmdb_movie_details(tmdb_id: int, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed movie information from TMDB.
    
    Args:
    ----
        tmdb_id (int): TMDB movie ID.
        api_key (str): TMDB API key.
        
    Returns:
    -------
        Optional[Dict]: Detailed movie information or None if not found.
        
    Raises:
    ------
        APIConnectionError: If TMDB API request fails.
    """
    try:
        # Build details URL
        details_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {
            'api_key': api_key,
            'append_to_response': 'credits,videos,similar'  # Get additional data
        }
        
        # Make API request
        response = requests.get(details_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Process and return detailed information
        details = {
            'tmdb_id': data.get('id'),
            'title': data.get('title'),
            'original_title': data.get('original_title'),
            'overview': data.get('overview'),
            'release_date': data.get('release_date'),
            'runtime': data.get('runtime'),
            'vote_average': data.get('vote_average'),
            'vote_count': data.get('vote_count'),
            'genres': [genre['name'] for genre in data.get('genres', [])],
            'poster_path': data.get('poster_path'),
            'backdrop_path': data.get('backdrop_path'),
            'budget': data.get('budget'),
            'revenue': data.get('revenue'),
            'tagline': data.get('tagline'),
            'status': data.get('status')
        }
        
        # Build image URLs
        if details['poster_path']:
            details['poster_url'] = f"https://image.tmdb.org/t/p/original{details['poster_path']}"
        
        if details['backdrop_path']:
            details['backdrop_url'] = f"https://image.tmdb.org/t/p/w1280{details['backdrop_path']}"
        
        # Build TMDB page URL
        details['tmdb_url'] = f"https://www.themoviedb.org/movie/{details['tmdb_id']}"
        
        return details
        
    except requests.RequestException as e:
        raise APIConnectionError(f"Failed to get TMDB movie details: {str(e)}")


def clean_movie_title(title: str) -> str:
    """
    Clean movie title for better TMDB search results.
    
    Args:
    ----
        title (str): Raw movie title.
        
    Returns:
    -------
        str: Cleaned title.
    """
    import re
    
    # Remove year from title
    title = re.sub(r'\(\d{4}\)', '', title)
    
    # Remove common patterns that might interfere with search
    title = re.sub(r',\s*The$', '', title)  # Remove trailing "The"
    title = re.sub(r'^The\s+', '', title)   # Remove leading "The"
    
    # Clean up whitespace
    title = title.strip()
    
    return title 