import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import optuna

from report_objects.errors.model_not_trained_error import ModelNotTrainedError
from report_objects.errors.user_not_found_error import UserNotFoundError
from report_objects.errors.data_validation_error import DataValidationError


class RecommenderBuilder:
    """
    RecommenderBuilder class.

    RecommenderBuilder class to instantiate recommender objects. These can then be used to build
    collaborative filtering models, perform similarity calculations, matrix factorization,
    and generate movie recommendations.
    """

    def __init__(self, user_item_matrix: pd.DataFrame):
        """
        RecommenderBuilder constructor.

        Initialize a RecommenderBuilder object by setting the fields to the arguments passed
        to the constructor.

        Args:
        ----
            user_item_matrix (pd.DataFrame): User-item matrix with users as rows and movies as columns.

        """
        self.user_item_matrix = user_item_matrix
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        self.is_trained = False

    def compute_user_similarity(self, metric: str = 'cosine') -> np.ndarray:
        """
        Compute user-user similarity matrix.

        Args:
        ----
            metric (str): Similarity metric to use ('cosine', 'pearson').

        Returns:
        -------
            np.ndarray: User similarity matrix.

        """
        if metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif metric == 'pearson':
            self.user_similarity_matrix = np.corrcoef(self.user_item_matrix)
            # Replace NaN values with 0
            self.user_similarity_matrix = np.nan_to_num(self.user_similarity_matrix)
        else:
            raise DataValidationError(f"Unsupported similarity metric: {metric}")

        return self.user_similarity_matrix

    def compute_item_similarity(self, metric: str = 'cosine') -> np.ndarray:
        """
        Compute item-item similarity matrix.

        Args:
        ----
            metric (str): Similarity metric to use ('cosine', 'pearson').

        Returns:
        -------
            np.ndarray: Item similarity matrix.

        """
        if metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        elif metric == 'pearson':
            self.item_similarity_matrix = np.corrcoef(self.user_item_matrix.T)
            # Replace NaN values with 0
            self.item_similarity_matrix = np.nan_to_num(self.item_similarity_matrix)
        else:
            raise DataValidationError(f"Unsupported similarity metric: {metric}")

        return self.item_similarity_matrix

    def train_matrix_factorization(self, n_components: int = 50, random_state: int = 42) -> Dict[str, Any]:
        """
        Train matrix factorization model using SVD.

        Args:
        ----
            n_components (int): Number of latent factors.
            random_state (int): Random seed for reproducibility.

        Returns:
        -------
            Dict[str, Any]: Training results and model information.

        """
        # Convert to sparse matrix for efficiency
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Initialize and train SVD model
        self.svd_model = TruncatedSVD(
            n_components=n_components,
            random_state=random_state
        )
        
        # Fit the model and get user factors
        self.user_factors = self.svd_model.fit_transform(sparse_matrix)
        
        # Get item factors
        self.item_factors = self.svd_model.components_.T
        
        self.is_trained = True
        
        # Calculate explained variance ratio
        explained_variance_ratio = self.svd_model.explained_variance_ratio_.sum()
        
        return {
            'model_type': 'SVD',
            'n_components': n_components,
            'explained_variance_ratio': explained_variance_ratio,
            'user_factors_shape': self.user_factors.shape,
            'item_factors_shape': self.item_factors.shape
        }

    def predict_rating(self, user_idx: int, item_idx: int, method: str = 'svd') -> float:
        """
        Predict rating for a specific user-item pair.

        Args:
        ----
            user_idx (int): Index of the user in the user-item matrix.
            item_idx (int): Index of the item in the user-item matrix.
            method (str): Prediction method ('svd', 'user_based', 'item_based').

        Returns:
        -------
            float: Predicted rating.

        Raises:
        ------
            ModelNotTrainedError: If the specified method hasn't been trained.

        """
        if method == 'svd':
            if not self.is_trained or self.user_factors is None:
                raise ModelNotTrainedError("SVD model must be trained before making predictions")
            
            predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            return predicted_rating
            
        elif method == 'user_based':
            if self.user_similarity_matrix is None:
                raise ModelNotTrainedError("User similarity matrix must be computed before making predictions")
            
            return self._predict_user_based(user_idx, item_idx)
            
        elif method == 'item_based':
            if self.item_similarity_matrix is None:
                raise ModelNotTrainedError("Item similarity matrix must be computed before making predictions")
            
            return self._predict_item_based(user_idx, item_idx)
            
        else:
            raise DataValidationError(f"Unsupported prediction method: {method}")

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10,
                               method: str = 'svd', exclude_rated: bool = True) -> List[Dict]:
        """
        Get movie recommendations for a specific user.

        Args:
        ----
            user_id (int): ID of the user (from original dataset).
            n_recommendations (int): Number of recommendations to return.
            method (str): Recommendation method ('svd', 'user_based', 'item_based').
            exclude_rated (bool): Whether to exclude movies the user has already rated.

        Returns:
        -------
            List[Dict]: List of recommended movies with predicted ratings.

        Raises:
        ------
            UserNotFoundError: If user is not found in the dataset.

        """
        # Check if user exists in the matrix
        if user_id not in self.user_item_matrix.index:
            raise UserNotFoundError(f"User {user_id} not found in the dataset")

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Get predictions for all movies
        predictions = []
        for item_idx, movie_id in enumerate(self.user_item_matrix.columns):
            # Skip if user has already rated this movie and exclude_rated is True
            if exclude_rated and user_ratings.iloc[item_idx] > 0:
                continue
                
            try:
                predicted_rating = self.predict_rating(user_idx, item_idx, method)
                predictions.append({
                    'movie_id': movie_id,
                    'predicted_rating': predicted_rating
                })
            except ModelNotTrainedError:
                raise
            except Exception:
                continue  # Skip movies that can't be predicted

        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:n_recommendations]

    def evaluate_model(self, test_data: pd.DataFrame, method: str = 'svd') -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
        ----
            test_data (pd.DataFrame): Test dataset with columns ['user_id', 'movie_id', 'rating'].
            method (str): Evaluation method ('svd', 'user_based', 'item_based').

        Returns:
        -------
            Dict[str, float]: Dictionary containing evaluation metrics.

        """
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            actual_rating = row['rating']
            
            try:
                # Get indices
                user_idx = self.user_item_matrix.index.get_loc(user_id)
                item_idx = self.user_item_matrix.columns.get_loc(movie_id)
                
                # Predict rating
                predicted_rating = self.predict_rating(user_idx, item_idx, method)
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
                
            except (KeyError, ValueError):
                continue  # Skip if user or movie not in training data

        if not predictions:
            return {'error': 'No valid predictions could be made'}

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'num_predictions': len(predictions)
        }

    def optimize_hyperparameters(self, test_data: pd.DataFrame, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
        ----
            test_data (pd.DataFrame): Test dataset for evaluation.
            n_trials (int): Number of optimization trials.

        Returns:
        -------
            Dict[str, Any]: Best hyperparameters and optimization results.

        """
        def objective(trial):
            # Suggest hyperparameters
            n_components = trial.suggest_int('n_components', 10, 100)
            
            # Train model with suggested parameters
            self.train_matrix_factorization(n_components=n_components)
            
            # Evaluate model
            metrics = self.evaluate_model(test_data, method='svd')
            
            # Return RMSE (we want to minimize this)
            return metrics.get('rmse', float('inf'))

        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters and retrain with them
        best_params = study.best_params
        self.train_matrix_factorization(n_components=best_params['n_components'])
        
        return {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': n_trials
        }

    def _predict_user_based(self, user_idx: int, item_idx: int, k: int = 50) -> float:
        """
        Predict rating using user-based collaborative filtering.

        Args:
        ----
            user_idx (int): Index of the user.
            item_idx (int): Index of the item.
            k (int): Number of similar users to consider.

        Returns:
        -------
            float: Predicted rating.

        """
        # Get user similarities
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Get ratings for the target item
        item_ratings = self.user_item_matrix.iloc[:, item_idx]
        
        # Find users who have rated this item
        rated_users = item_ratings > 0
        
        if not rated_users.any():
            return self.user_item_matrix.iloc[user_idx].mean()  # User's average rating
        
        # Get similarities and ratings for users who rated this item
        similarities = user_similarities[rated_users]
        ratings = item_ratings[rated_users]
        
        # Sort by similarity and take top k
        top_k_indices = similarities.argsort()[-k:][::-1]
        top_similarities = similarities.iloc[top_k_indices]
        top_ratings = ratings.iloc[top_k_indices]
        
        # Calculate weighted average
        if top_similarities.sum() == 0:
            return self.user_item_matrix.iloc[user_idx].mean()
        
        predicted_rating = np.sum(top_similarities * top_ratings) / np.sum(np.abs(top_similarities))
        return predicted_rating

    def _predict_item_based(self, user_idx: int, item_idx: int, k: int = 50) -> float:
        """
        Predict rating using item-based collaborative filtering.

        Args:
        ----
            user_idx (int): Index of the user.
            item_idx (int): Index of the item.
            k (int): Number of similar items to consider.

        Returns:
        -------
            float: Predicted rating.

        """
        # Get item similarities
        item_similarities = self.item_similarity_matrix[item_idx]
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Find items that the user has rated
        rated_items = user_ratings > 0
        
        if not rated_items.any():
            return self.user_item_matrix.iloc[:, item_idx].mean()  # Item's average rating
        
        # Get similarities and ratings for items the user has rated
        similarities = item_similarities[rated_items]
        ratings = user_ratings[rated_items]
        
        # Sort by similarity and take top k
        top_k_indices = similarities.argsort()[-k:][::-1]
        top_similarities = similarities.iloc[top_k_indices]
        top_ratings = ratings.iloc[top_k_indices]
        
        # Calculate weighted average
        if top_similarities.sum() == 0:
            return self.user_item_matrix.iloc[:, item_idx].mean()
        
        predicted_rating = np.sum(top_similarities * top_ratings) / np.sum(np.abs(top_similarities))
        return predicted_rating

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model state.

        Returns:
        -------
            Dict[str, Any]: Model information and statistics.

        """
        info = {
            'is_trained': self.is_trained,
            'matrix_shape': self.user_item_matrix.shape,
            'num_users': len(self.user_item_matrix.index),
            'num_items': len(self.user_item_matrix.columns),
            'sparsity': (self.user_item_matrix == 0).sum().sum() / self.user_item_matrix.size,
            'user_similarity_computed': self.user_similarity_matrix is not None,
            'item_similarity_computed': self.item_similarity_matrix is not None
        }
        
        if self.is_trained and self.svd_model is not None:
            info['svd_components'] = self.svd_model.n_components
            info['explained_variance_ratio'] = self.svd_model.explained_variance_ratio_.sum()
        
        return info 