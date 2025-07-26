import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from report_objects.data_reader import DataReader
from report_objects.model_writer import ModelWriter
from report_objects.recommender_builder import RecommenderBuilder
from report_objects.errors.model_not_trained_error import ModelNotTrainedError
from report_objects.errors.user_not_found_error import UserNotFoundError
from report_objects.errors.data_validation_error import DataValidationError
from report_objects.errors.api_connection_error import APIConnectionError


class RecommenderManager:
    """
    RecommenderManager class.

    RecommenderManager class to instantiate manager objects. These can then be used to orchestrate
    the entire movie recommendation pipeline, coordinating data loading, model training, 
    evaluation, and recommendation generation.
    """

    def __init__(
        self,
        data_path: str = "data/",
        output_path: str = "outputs/",
        tmdb_api_key: Optional[str] = None
    ):
        """
        RecommenderManager constructor.

        Initialize a RecommenderManager object by setting up the core components.

        Args:
        ----
            data_path (str): Path to the directory containing MovieLens data files.
            output_path (str): Path for saving models, logs, and results.
            tmdb_api_key (Optional[str]): API key for The Movie Database (TMDB) API.

        """
        self.data_reader = DataReader(data_path=data_path, tmdb_api_key=tmdb_api_key)
        self.model_writer = ModelWriter(output_path=output_path)
        self.recommender_builder = None
        
        self.data_loaded = False
        self.model_trained = False
        self.current_model_type = None
        
        # Store loaded data
        self.ratings_data = None
        self.movies_data = None
        self.users_data = None
        self.train_data = None
        self.test_data = None
        self.user_item_matrix = None

    def load_and_prepare_data(self, dataset_size: str = "100k", test_size: float = 0.2) -> Dict[str, Any]:
        """
        Load and prepare MovieLens data for training.

        Args:
        ----
            dataset_size (str): Size of MovieLens dataset ("100k", "1m", "10m", "20m").
            test_size (float): Proportion of data to use for testing.

        Returns:
        -------
            Dict[str, Any]: Data loading and preparation results.

        Raises:
        ------
            DataValidationError: If data loading or preparation fails.

        """
        try:
            # Load data
            data_dict = self.data_reader.load_movielens_data(dataset_size=dataset_size)
            self.ratings_data = data_dict['ratings']
            self.movies_data = data_dict['movies']
            self.users_data = data_dict['users']
            
            # Split data
            self.train_data, self.test_data = self.data_reader.split_data(test_size=test_size)
            
            # Create user-item matrix from training data
            train_reader = DataReader()
            train_reader.ratings = self.train_data
            self.user_item_matrix = train_reader.create_user_item_matrix()
            
            # Initialize recommender builder with the matrix
            self.recommender_builder = RecommenderBuilder(self.user_item_matrix)
            
            self.data_loaded = True
            
            # Get data statistics
            stats = self.data_reader.get_data_stats()
            
            return {
                'status': 'success',
                'dataset_size': dataset_size,
                'data_stats': stats,
                'train_size': len(self.train_data),
                'test_size': len(self.test_data),
                'matrix_shape': self.user_item_matrix.shape
            }
            
        except Exception as e:
            raise DataValidationError(f"Failed to load and prepare data: {str(e)}")

    def train_model(self, model_type: str = 'svd', **kwargs) -> Dict[str, Any]:
        """
        Train a recommendation model.

        Args:
        ----
            model_type (str): Type of model to train ('svd', 'user_based', 'item_based').
            **kwargs: Additional arguments for model training.

        Returns:
        -------
            Dict[str, Any]: Training results and model information.

        Raises:
        ------
            DataValidationError: If data is not loaded.
            ModelNotTrainedError: If training fails.

        """
        if not self.data_loaded or self.recommender_builder is None:
            raise DataValidationError("Data must be loaded before training models")

        try:
            if model_type == 'svd':
                n_components = kwargs.get('n_components', 50)
                random_state = kwargs.get('random_state', 42)
                
                training_results = self.recommender_builder.train_matrix_factorization(
                    n_components=n_components,
                    random_state=random_state
                )
                
            elif model_type == 'user_based':
                metric = kwargs.get('similarity_metric', 'cosine')
                self.recommender_builder.compute_user_similarity(metric=metric)
                training_results = {
                    'model_type': 'user_based',
                    'similarity_metric': metric,
                    'similarity_matrix_shape': self.recommender_builder.user_similarity_matrix.shape
                }
                
            elif model_type == 'item_based':
                metric = kwargs.get('similarity_metric', 'cosine')
                self.recommender_builder.compute_item_similarity(metric=metric)
                training_results = {
                    'model_type': 'item_based',
                    'similarity_metric': metric,
                    'similarity_matrix_shape': self.recommender_builder.item_similarity_matrix.shape
                }
                
            else:
                raise DataValidationError(f"Unsupported model type: {model_type}")

            self.model_trained = True
            self.current_model_type = model_type
            
            # Log the experiment
            experiment_data = {
                'model_type': model_type,
                'training_params': kwargs,
                'training_results': training_results,
                'data_stats': self.data_reader.get_data_stats()
            }
            
            log_path = self.model_writer.log_experiment(experiment_data)
            training_results['experiment_log'] = log_path
            
            return training_results
            
        except Exception as e:
            raise ModelNotTrainedError(f"Model training failed: {str(e)}")

    def evaluate_model(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Args:
        ----
            model_type (Optional[str]): Model type to evaluate. Uses current model if None.

        Returns:
        -------
            Dict[str, Any]: Evaluation results.

        Raises:
        ------
            ModelNotTrainedError: If no model is trained.

        """
        if not self.model_trained or self.recommender_builder is None:
            raise ModelNotTrainedError("Model must be trained before evaluation")

        eval_model_type = model_type or self.current_model_type
        
        try:
            evaluation_results = self.recommender_builder.evaluate_model(
                self.test_data, 
                method=eval_model_type
            )
            
            # Save evaluation results
            eval_path = self.model_writer.export_evaluation_results(
                evaluation_results, 
                eval_model_type
            )
            
            evaluation_results['evaluation_log'] = eval_path
            return evaluation_results
            
        except Exception as e:
            raise ModelNotTrainedError(f"Model evaluation failed: {str(e)}")

    def get_recommendations(self, user_id: int, n_recommendations: int = 10,
                          include_metadata: bool = True, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get movie recommendations for a user.

        Args:
        ----
            user_id (int): ID of the user to get recommendations for.
            n_recommendations (int): Number of recommendations to return.
            include_metadata (bool): Whether to include movie metadata from TMDB.
            model_type (Optional[str]): Model type to use. Uses current model if None.

        Returns:
        -------
            Dict[str, Any]: Recommendations with metadata.

        Raises:
        ------
            ModelNotTrainedError: If no model is trained.
            UserNotFoundError: If user is not found.

        """
        if not self.model_trained or self.recommender_builder is None:
            raise ModelNotTrainedError("Model must be trained before generating recommendations")

        rec_model_type = model_type or self.current_model_type
        
        try:
            # Get basic recommendations
            recommendations = self.recommender_builder.get_user_recommendations(
                user_id=user_id,
                n_recommendations=n_recommendations,
                method=rec_model_type
            )
            
            # Enhance with movie titles and metadata
            enhanced_recommendations = []
            for rec in recommendations:
                movie_id = rec['movie_id']
                
                # Get movie title
                movie_info = self.movies_data[self.movies_data['movie_id'] == movie_id]
                if not movie_info.empty:
                    title = movie_info.iloc[0]['title']
                    genres = movie_info.iloc[0].get('genres', 'Unknown')
                    
                    enhanced_rec = {
                        'movie_id': movie_id,
                        'title': title,
                        'genres': genres,
                        'predicted_rating': rec['predicted_rating']
                    }
                    
                    # Add TMDB metadata if requested and API key is available
                    if include_metadata and self.data_reader.tmdb_api_key:
                        try:
                            tmdb_data = self.data_reader.get_movie_metadata(title)
                            if tmdb_data:
                                enhanced_rec.update({
                                    'poster_url': tmdb_data.get('poster_url'),
                                    'overview': tmdb_data.get('overview'),
                                    'release_date': tmdb_data.get('release_date'),
                                    'vote_average': tmdb_data.get('vote_average')
                                })
                        except APIConnectionError:
                            pass  # Continue without TMDB data
                    
                    enhanced_recommendations.append(enhanced_rec)
            
            # Save recommendations
            rec_path = self.model_writer.save_recommendations(
                user_id=user_id,
                recommendations=enhanced_recommendations,
                model_name=rec_model_type
            )
            
            # Create response
            response = self.model_writer.create_flask_response(
                user_id=user_id,
                recommendations=enhanced_recommendations,
                message=f"Generated {len(enhanced_recommendations)} recommendations using {rec_model_type} model"
            )
            
            response['recommendations_log'] = rec_path
            return response
            
        except UserNotFoundError:
            raise
        except Exception as e:
            error_response = self.model_writer.create_error_response(
                error_message=f"Failed to generate recommendations: {str(e)}",
                error_type="recommendation_error"
            )
            return error_response

    def optimize_model(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using Optuna.

        Args:
        ----
            n_trials (int): Number of optimization trials.

        Returns:
        -------
            Dict[str, Any]: Optimization results.

        Raises:
        ------
            DataValidationError: If data is not loaded.

        """
        if not self.data_loaded or self.recommender_builder is None:
            raise DataValidationError("Data must be loaded before optimization")

        try:
            optimization_results = self.recommender_builder.optimize_hyperparameters(
                test_data=self.test_data,
                n_trials=n_trials
            )
            
            self.model_trained = True
            self.current_model_type = 'svd'
            
            # Log optimization experiment
            experiment_data = {
                'experiment_type': 'hyperparameter_optimization',
                'optimization_results': optimization_results,
                'n_trials': n_trials
            }
            
            log_path = self.model_writer.log_experiment(experiment_data)
            optimization_results['experiment_log'] = log_path
            
            return optimization_results
            
        except Exception as e:
            raise ModelNotTrainedError(f"Model optimization failed: {str(e)}")

    def save_model(self, model_name: Optional[str] = None) -> str:
        """
        Save the current trained model.

        Args:
        ----
            model_name (Optional[str]): Name for the saved model. Auto-generated if None.

        Returns:
        -------
            str: Path where the model was saved.

        Raises:
        ------
            ModelNotTrainedError: If no model is trained.

        """
        if not self.model_trained or self.recommender_builder is None:
            raise ModelNotTrainedError("No trained model to save")

        if model_name is None:
            model_name = f"recommender_{self.current_model_type}"
        
        metadata = {
            'model_type': self.current_model_type,
            'model_info': self.recommender_builder.get_model_info(),
            'data_stats': self.data_reader.get_data_stats()
        }
        
        model_path = self.model_writer.save_model(
            model=self.recommender_builder,
            model_name=model_name,
            metadata=metadata
        )
        
        return model_path

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a previously saved model.

        Args:
        ----
            model_path (str): Path to the saved model file.

        Returns:
        -------
            Dict[str, Any]: Model loading results.

        """
        try:
            self.recommender_builder = self.model_writer.load_model(model_path)
            
            model_info = self.recommender_builder.get_model_info()
            self.model_trained = model_info['is_trained']
            
            return {
                'status': 'success',
                'model_path': model_path,
                'model_info': model_info
            }
            
        except Exception as e:
            raise DataValidationError(f"Failed to load model: {str(e)}")

    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """
        Get user profile information including rating history.

        Args:
        ----
            user_id (int): ID of the user.

        Returns:
        -------
            Dict[str, Any]: User profile information.

        Raises:
        ------
            DataValidationError: If data is not loaded.
            UserNotFoundError: If user is not found.

        """
        if not self.data_loaded:
            raise DataValidationError("Data must be loaded to get user profiles")

        try:
            user_ratings = self.data_reader.get_user_ratings(user_id)
            
            if user_ratings.empty:
                raise UserNotFoundError(f"User {user_id} not found in the dataset")
            
            # Calculate user statistics
            avg_rating = user_ratings['rating'].mean()
            num_ratings = len(user_ratings)
            favorite_genres = user_ratings['genres'].value_counts().head(3).to_dict()
            
            profile = {
                'user_id': user_id,
                'num_ratings': num_ratings,
                'avg_rating': avg_rating,
                'favorite_genres': favorite_genres,
                'recent_ratings': user_ratings.sort_values('timestamp', ascending=False).head(5).to_dict('records')
            }
            
            return profile
            
        except UserNotFoundError:
            raise
        except Exception as e:
            raise DataValidationError(f"Failed to get user profile: {str(e)}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics.

        Returns:
        -------
            Dict[str, Any]: System status information.

        """
        status = {
            'data_loaded': self.data_loaded,
            'model_trained': self.model_trained,
            'current_model_type': self.current_model_type,
            'tmdb_api_available': self.data_reader.tmdb_api_key is not None
        }
        
        if self.data_loaded:
            status['data_stats'] = self.data_reader.get_data_stats()
            
        if self.model_trained and self.recommender_builder:
            status['model_info'] = self.recommender_builder.get_model_info()
            
        # Get experiment history
        status['experiment_history'] = self.model_writer.get_experiment_history()[:5]  # Last 5 experiments
        
        return status

    def record_user_feedback(self, user_id: int, movie_id: int, rating: float) -> str:
        """
        Record user feedback for future model improvement.

        Args:
        ----
            user_id (int): ID of the user providing feedback.
            movie_id (int): ID of the movie being rated.
            rating (float): User's rating for the movie.

        Returns:
        -------
            str: Path where feedback was saved.

        """
        feedback_path = self.model_writer.save_user_feedback(
            user_id=user_id,
            movie_id=movie_id,
            rating=rating,
            feedback_type="explicit"
        )
        
        return feedback_path 