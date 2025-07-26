import os
import json
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from flask import jsonify

from report_objects.errors.data_validation_error import DataValidationError


class ModelWriter:
    """
    ModelWriter class.

    ModelWriter class to instantiate writer objects, which can then be used
    to save trained models, log experiments, export results, and handle Flask API responses
    for the movie recommendation system.
    """

    def __init__(self, output_path: str = "outputs/"):
        """
        ModelWriter constructor.

        Initialize a ModelWriter object by setting the fields to the arguments passed
        to the constructor.

        Args:
        ----
            output_path (str): Base path for saving models, logs, and results.

        """
        self.output_path = output_path
        self.models_path = os.path.join(output_path, "models")
        self.logs_path = os.path.join(output_path, "logs")
        self.results_path = os.path.join(output_path, "results")
        
        # Create directories if they don't exist
        for path in [self.models_path, self.logs_path, self.results_path]:
            os.makedirs(path, exist_ok=True)

    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Save a trained recommendation model to disk.

        Args:
        ----
            model (Any): The trained model object to save.
            model_name (str): Name identifier for the model.
            metadata (Optional[Dict]): Additional metadata about the model.

        Returns:
        -------
            str: Path where the model was saved.

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(self.models_path, model_filename)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata if provided
        if metadata:
            metadata_filename = f"{model_name}_{timestamp}_metadata.json"
            metadata_path = os.path.join(self.models_path, metadata_filename)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        return model_path

    def load_model(self, model_path: str) -> Any:
        """
        Load a saved model from disk.

        Args:
        ----
            model_path (str): Path to the saved model file.

        Returns:
        -------
            Any: The loaded model object.

        Raises:
        ------
            DataValidationError: If model file doesn't exist or cannot be loaded.

        """
        if not os.path.exists(model_path):
            raise DataValidationError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise DataValidationError(f"Failed to load model: {str(e)}")

    def log_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """
        Log experiment results and parameters.

        Args:
        ----
            experiment_data (Dict[str, Any]): Dictionary containing experiment parameters and results.

        Returns:
        -------
            str: Path where the experiment log was saved.

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"experiment_{timestamp}.json"
        log_path = os.path.join(self.logs_path, log_filename)
        
        # Add timestamp to experiment data
        experiment_data['timestamp'] = timestamp
        experiment_data['datetime'] = datetime.now().isoformat()
        
        with open(log_path, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        return log_path

    def save_recommendations(self, user_id: int, recommendations: List[Dict], 
                           model_name: str) -> str:
        """
        Save user recommendations to a file.

        Args:
        ----
            user_id (int): ID of the user receiving recommendations.
            recommendations (List[Dict]): List of recommended movies with metadata.
            model_name (str): Name of the model that generated recommendations.

        Returns:
        -------
            str: Path where recommendations were saved.

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recommendations_filename = f"recommendations_user_{user_id}_{timestamp}.json"
        recommendations_path = os.path.join(self.results_path, recommendations_filename)
        
        data = {
            'user_id': user_id,
            'model_name': model_name,
            'timestamp': timestamp,
            'recommendations': recommendations
        }
        
        with open(recommendations_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return recommendations_path

    def export_evaluation_results(self, evaluation_results: Dict[str, float], 
                                model_name: str) -> str:
        """
        Export model evaluation metrics to a file.

        Args:
        ----
            evaluation_results (Dict[str, float]): Dictionary of evaluation metrics.
            model_name (str): Name of the evaluated model.

        Returns:
        -------
            str: Path where evaluation results were saved.

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_filename = f"evaluation_{model_name}_{timestamp}.json"
        eval_path = os.path.join(self.results_path, eval_filename)
        
        data = {
            'model_name': model_name,
            'timestamp': timestamp,
            'evaluation_metrics': evaluation_results
        }
        
        with open(eval_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return eval_path

    def create_flask_response(self, user_id: int, recommendations: List[Dict], 
                            status: str = "success", message: str = "") -> Dict:
        """
        Create a standardized Flask API response for recommendations.

        Args:
        ----
            user_id (int): ID of the user receiving recommendations.
            recommendations (List[Dict]): List of recommended movies.
            status (str): Status of the response ("success" or "error").
            message (str): Additional message for the response.

        Returns:
        -------
            Dict: Standardized response dictionary for Flask API.

        """
        response = {
            'status': status,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'num_recommendations': len(recommendations),
            'recommendations': recommendations
        }
        
        if message:
            response['message'] = message
        
        return response

    def create_error_response(self, error_message: str, error_type: str = "error") -> Dict:
        """
        Create a standardized error response for Flask API.

        Args:
        ----
            error_message (str): Description of the error.
            error_type (str): Type of error occurred.

        Returns:
        -------
            Dict: Standardized error response dictionary.

        """
        return {
            'status': 'error',
            'error_type': error_type,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }

    def save_user_feedback(self, user_id: int, movie_id: int, rating: float, 
                          feedback_type: str = "explicit") -> str:
        """
        Save user feedback for future model training.

        Args:
        ----
            user_id (int): ID of the user providing feedback.
            movie_id (int): ID of the movie being rated.
            rating (float): User's rating for the movie.
            feedback_type (str): Type of feedback ("explicit", "implicit").

        Returns:
        -------
            str: Path where feedback was saved.

        """
        feedback_filename = "user_feedback.csv"
        feedback_path = os.path.join(self.logs_path, feedback_filename)
        
        feedback_data = {
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'feedback_type': feedback_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Append to existing feedback file or create new one
        feedback_df = pd.DataFrame([feedback_data])
        
        if os.path.exists(feedback_path):
            feedback_df.to_csv(feedback_path, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_path, index=False)
        
        return feedback_path

    def get_experiment_history(self) -> List[Dict]:
        """
        Retrieve experiment history from log files.

        Returns:
        -------
            List[Dict]: List of all logged experiments.

        """
        experiments = []
        
        for filename in os.listdir(self.logs_path):
            if filename.startswith("experiment_") and filename.endswith(".json"):
                file_path = os.path.join(self.logs_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        experiment = json.load(f)
                        experiments.append(experiment)
                except Exception:
                    continue  # Skip corrupted files
        
        # Sort by timestamp
        experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return experiments

    def cleanup_old_files(self, days_to_keep: int = 30) -> int:
        """
        Clean up old log and result files.

        Args:
        ----
            days_to_keep (int): Number of days of files to keep.

        Returns:
        -------
            int: Number of files deleted.

        """
        import time
        
        deleted_count = 0
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for directory in [self.logs_path, self.results_path]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.getctime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception:
                        continue
        
        return deleted_count 