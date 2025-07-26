"""
Models API endpoints.

This module contains Flask routes for training, evaluating, and managing ML models.
"""

from flask import Blueprint, request, jsonify, current_app
from report_objects.errors.model_not_trained_error import ModelNotTrainedError
from report_objects.errors.data_validation_error import DataValidationError

models_bp = Blueprint('models', __name__)


@models_bp.route('/train', methods=['POST'])
def train_model():
    """
    Train a recommendation model.
    
    Request Body:
    ------------
        {
            "model_type": "svd",
            "n_components": 50,
            "similarity_metric": "cosine",
            "random_state": 42
        }
        
    Returns:
    -------
        JSON response with training results.
    """
    try:
        data = request.get_json() or {}
        
        model_type = data.get('model_type', 'svd')
        
        # Validate model type
        valid_models = ['svd', 'user_based', 'item_based']
        if model_type not in valid_models:
            return jsonify({
                'status': 'error',
                'message': f'Invalid model_type. Must be one of: {valid_models}'
            }), 400
        
        # Prepare training parameters
        training_params = {}
        
        if model_type == 'svd':
            training_params['n_components'] = data.get('n_components', 50)
            training_params['random_state'] = data.get('random_state', 42)
        elif model_type in ['user_based', 'item_based']:
            training_params['similarity_metric'] = data.get('similarity_metric', 'cosine')
        
        # Train the model
        result = current_app.recommender_manager.train_model(
            model_type=model_type,
            **training_params
        )
        
        return jsonify({
            'status': 'success',
            'message': f'{model_type} model trained successfully',
            'training_results': result
        })
        
    except DataValidationError as e:
        raise e
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Model training failed: {str(e)}'
        }), 500


@models_bp.route('/evaluate', methods=['GET'])
def evaluate_model():
    """
    Evaluate the currently trained model.
    
    Query Parameters:
    ----------------
        model_type (str): Specific model to evaluate (optional)
        
    Returns:
    -------
        JSON response with evaluation metrics.
    """
    try:
        model_type = request.args.get('model_type', None)
        
        # Evaluate the model
        result = current_app.recommender_manager.evaluate_model(model_type=model_type)
        
        return jsonify({
            'status': 'success',
            'evaluation_results': result
        })
        
    except ModelNotTrainedError as e:
        raise e
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Model evaluation failed: {str(e)}'
        }), 500


@models_bp.route('/optimize', methods=['POST'])
def optimize_hyperparameters():
    """
    Optimize model hyperparameters using Optuna.
    
    Request Body:
    ------------
        {
            "n_trials": 50,
            "timeout": 300
        }
        
    Returns:
    -------
        JSON response with optimization results.
    """
    try:
        data = request.get_json() or {}
        
        n_trials = data.get('n_trials', 50)
        
        # Validate parameters
        if not isinstance(n_trials, int) or n_trials < 1:
            return jsonify({
                'status': 'error',
                'message': 'n_trials must be a positive integer'
            }), 400
        
        if n_trials > 200:  # Limit for reasonable execution time
            return jsonify({
                'status': 'error',
                'message': 'n_trials cannot exceed 200'
            }), 400
        
        # Run optimization
        result = current_app.recommender_manager.optimize_model(n_trials=n_trials)
        
        return jsonify({
            'status': 'success',
            'message': 'Hyperparameter optimization completed',
            'optimization_results': result
        })
        
    except DataValidationError as e:
        raise e
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Hyperparameter optimization failed: {str(e)}'
        }), 500


@models_bp.route('/save', methods=['POST'])
def save_model():
    """
    Save the currently trained model.
    
    Request Body:
    ------------
        {
            "model_name": "my_svd_model"
        }
        
    Returns:
    -------
        JSON response with save confirmation.
    """
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name', None)
        
        # Save the model
        model_path = current_app.recommender_manager.save_model(model_name=model_name)
        
        return jsonify({
            'status': 'success',
            'message': 'Model saved successfully',
            'model_path': model_path
        })
        
    except ModelNotTrainedError as e:
        raise e
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Model save failed: {str(e)}'
        }), 500


@models_bp.route('/load', methods=['POST'])
def load_model():
    """
    Load a previously saved model.
    
    Request Body:
    ------------
        {
            "model_path": "/path/to/model.pkl"
        }
        
    Returns:
    -------
        JSON response with load confirmation.
    """
    try:
        data = request.get_json()
        
        if not data or 'model_path' not in data:
            return jsonify({
                'status': 'error',
                'message': 'model_path is required'
            }), 400
        
        model_path = data['model_path']
        
        # Load the model
        result = current_app.recommender_manager.load_model(model_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'load_results': result
        })
        
    except DataValidationError as e:
        raise e
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Model load failed: {str(e)}'
        }), 500


@models_bp.route('/info')
def get_model_info():
    """
    Get information about the current model state.
    
    Returns:
    -------
        JSON response with model information.
    """
    try:
        status = current_app.recommender_manager.get_system_status()
        
        model_info = {
            'model_trained': status.get('model_trained', False),
            'current_model_type': status.get('current_model_type'),
            'data_loaded': status.get('data_loaded', False)
        }
        
        # Add detailed model info if available
        if status.get('model_info'):
            model_info['model_details'] = status['model_info']
        
        return jsonify({
            'status': 'success',
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model info: {str(e)}'
        }), 500


@models_bp.route('/types')
def get_supported_models():
    """
    Get list of supported model types and their descriptions.
    
    Returns:
    -------
        JSON response with supported model types.
    """
    models_info = {
        'svd': {
            'name': 'Singular Value Decomposition',
            'description': 'Matrix factorization using SVD for collaborative filtering',
            'parameters': {
                'n_components': 'Number of latent factors (default: 50)',
                'random_state': 'Random seed for reproducibility (default: 42)'
            },
            'pros': ['Fast training', 'Good for sparse data', 'Scalable'],
            'cons': ['Less interpretable', 'Requires hyperparameter tuning']
        },
        'user_based': {
            'name': 'User-Based Collaborative Filtering',
            'description': 'Recommendations based on similar users preferences',
            'parameters': {
                'similarity_metric': 'Similarity measure: cosine or pearson (default: cosine)'
            },
            'pros': ['Interpretable', 'Good for explaining recommendations'],
            'cons': ['Can be slow with many users', 'Cold start problem']
        },
        'item_based': {
            'name': 'Item-Based Collaborative Filtering',
            'description': 'Recommendations based on item similarities',
            'parameters': {
                'similarity_metric': 'Similarity measure: cosine or pearson (default: cosine)'
            },
            'pros': ['More stable than user-based', 'Pre-computation possible'],
            'cons': ['Less diverse recommendations', 'Popularity bias']
        }
    }
    
    return jsonify({
        'status': 'success',
        'supported_models': models_info,
        'default_model': 'svd'
    })


@models_bp.route('/experiments')
def get_experiment_history():
    """
    Get history of model training experiments.
    
    Query Parameters:
    ----------------
        limit (int): Maximum number of experiments to return (default: 10)
        
    Returns:
    -------
        JSON response with experiment history.
    """
    try:
        limit = min(int(request.args.get('limit', 10)), 100)
        
        # Get experiment history from model writer
        experiments = current_app.recommender_manager.model_writer.get_experiment_history()
        
        # Limit results
        limited_experiments = experiments[:limit]
        
        return jsonify({
            'status': 'success',
            'experiments': limited_experiments,
            'total_experiments': len(experiments),
            'showing': len(limited_experiments)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get experiment history: {str(e)}'
        }), 500 