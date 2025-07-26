import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split

from report_objects.errors.data_validation_error import DataValidationError
from report_objects.errors.api_connection_error import APIConnectionError


class DataReader:
    """
    DataReader class.

    DataReader class that defines the state and behavior of data_reader objects. Instantiated data_readers can
    then be used to load MovieLens datasets, fetch movie metadata from TMDB API, and process user input data
    for the movie recommendation system.
    """

    def __init__(self, data_path: str = "data/", tmdb_api_key: Optional[str] = None):
        """
        DataReader constructor.

        Initialize a DataReader object by setting the fields to the arguments passed to the constructor.

        Args:
        ----
            data_path (str): Path to the directory containing MovieLens data files.
            tmdb_api_key (Optional[str]): API key for The Movie Database (TMDB) API.

        """
        self.data_path = data_path
        self.tmdb_api_key = tmdb_api_key
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.ratings = None
        self.movies = None
        self.users = None

    def load_movielens_data(self, dataset_size: str = "100k") -> Dict[str, pd.DataFrame]:
        """
        Load MovieLens dataset from files.

        Load the MovieLens dataset files (ratings, movies, users) and return them as DataFrames.

        Args:
        ----
            dataset_size (str): Size of the MovieLens dataset to load ("100k", "1m", "10m", "20m").

        Returns:
        -------
            Dict[str, pd.DataFrame]: Dictionary containing loaded DataFrames for 'ratings', 'movies', and 'users'.

        Raises:
        ------
            DataValidationError: If data files are missing or have invalid format.

        """
        try:
            # Load ratings data - handle different dataset formats
            ratings_file = None
            possible_ratings_files = [
                os.path.join(self.data_path, f"ml-{dataset_size}", "ratings.dat"),
                os.path.join(self.data_path, f"ml-{dataset_size}", "ratings.csv"),
                os.path.join(self.data_path, f"ml-{dataset_size}", "u.data")  # MovieLens 100k format
            ]
            
            for file_path in possible_ratings_files:
                if os.path.exists(file_path):
                    ratings_file = file_path
                    break
            
            if not ratings_file:
                raise FileNotFoundError(f"No ratings file found in ml-{dataset_size} directory")
            
            # Load ratings based on file format
            if ratings_file.endswith('.dat'):
                self.ratings = pd.read_csv(
                    ratings_file,
                    sep='::',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'],
                    engine='python'
                )
            elif ratings_file.endswith('u.data'):
                # MovieLens 100k specific format
                self.ratings = pd.read_csv(
                    ratings_file,
                    sep='\t',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'],
                    engine='python'
                )
            else:
                # CSV format
                self.ratings = pd.read_csv(ratings_file)

            # Load movies data - handle different dataset formats
            movies_file = None
            possible_movies_files = [
                os.path.join(self.data_path, f"ml-{dataset_size}", "movies.dat"),
                os.path.join(self.data_path, f"ml-{dataset_size}", "movies.csv"),
                os.path.join(self.data_path, f"ml-{dataset_size}", "u.item")  # MovieLens 100k format
            ]
            
            for file_path in possible_movies_files:
                if os.path.exists(file_path):
                    movies_file = file_path
                    break
            
            if not movies_file:
                raise FileNotFoundError(f"No movies file found in ml-{dataset_size} directory")
            
            # Load movies based on file format
            if movies_file.endswith('.dat'):
                self.movies = pd.read_csv(
                    movies_file,
                    sep='::',
                    names=['movie_id', 'title', 'genres'],
                    engine='python'
                )
            elif movies_file.endswith('u.item'):
                # MovieLens 100k specific format
                self.movies = pd.read_csv(
                    movies_file,
                    sep='|',
                    names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
                          [f'genre_{i}' for i in range(19)],  # 19 genre columns
                    engine='python',
                    encoding='latin1'
                )
                # Convert genre columns to pipe-separated string
                genre_cols = [f'genre_{i}' for i in range(19)]
                genre_names = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
                
                def create_genre_string(row):
                    genres = []
                    for i, col in enumerate(genre_cols):
                        if row[col] == 1:
                            genres.append(genre_names[i])
                    return '|'.join(genres) if genres else 'unknown'
                
                self.movies['genres'] = self.movies[genre_cols].apply(create_genre_string, axis=1)
                # Keep only needed columns
                self.movies = self.movies[['movie_id', 'title', 'genres']]
            else:
                # CSV format
                self.movies = pd.read_csv(movies_file)

            # Load users data (if available)
            users_file = None
            possible_users_files = [
                os.path.join(self.data_path, f"ml-{dataset_size}", "users.dat"),
                os.path.join(self.data_path, f"ml-{dataset_size}", "users.csv"),
                os.path.join(self.data_path, f"ml-{dataset_size}", "u.user")  # MovieLens 100k format
            ]
            
            for file_path in possible_users_files:
                if os.path.exists(file_path):
                    users_file = file_path
                    break
            
            if users_file:
                if users_file.endswith('.dat'):
                    self.users = pd.read_csv(
                        users_file,
                        sep='::',
                        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                        engine='python'
                    )
                elif users_file.endswith('u.user'):
                    # MovieLens 100k specific format
                    self.users = pd.read_csv(
                        users_file,
                        sep='|',
                        names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                        engine='python'
                    )
                else:
                    # CSV format
                    self.users = pd.read_csv(users_file)

            # Validate required columns
            self._validate_data()

            return {
                'ratings': self.ratings,
                'movies': self.movies,
                'users': self.users
            }

        except Exception as e:
            raise DataValidationError(f"Failed to load MovieLens data: {str(e)}")

    def create_user_item_matrix(self) -> pd.DataFrame:
        """
        Create user-item matrix from ratings data.

        Transform the ratings DataFrame into a user-item matrix where rows are users,
        columns are movies, and values are ratings.

        Returns:
        -------
            pd.DataFrame: User-item matrix with users as rows and movies as columns.

        Raises:
        ------
            DataValidationError: If ratings data is not loaded.

        """
        if self.ratings is None:
            raise DataValidationError("Ratings data must be loaded before creating user-item matrix")

        user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )

        return user_item_matrix

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split ratings data into training and testing sets.

        Args:
        ----
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.

        Returns:
        -------
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.

        Raises:
        ------
            DataValidationError: If ratings data is not loaded.

        """
        if self.ratings is None:
            raise DataValidationError("Ratings data must be loaded before splitting")

        train_data, test_data = train_test_split(
            self.ratings,
            test_size=test_size,
            random_state=random_state,
            stratify=self.ratings['user_id']
        )

        return train_data, test_data

    def get_movie_metadata(self, movie_title: str) -> Optional[Dict]:
        """
        Fetch movie metadata from TMDB API.

        Args:
        ----
            movie_title (str): Title of the movie to search for.

        Returns:
        -------
            Optional[Dict]: Movie metadata including poster URL, genres, overview, etc.

        Raises:
        ------
            APIConnectionError: If TMDB API request fails.

        """
        if not self.tmdb_api_key:
            return None

        try:
            # Search for movie
            search_url = f"{self.tmdb_base_url}/search/movie"
            params = {
                'api_key': self.tmdb_api_key,
                'query': movie_title
            }

            response = requests.get(search_url, params=params)
            response.raise_for_status()

            data = response.json()
            if data['results']:
                movie = data['results'][0]  # Get first result
                
                # Build poster URL if available
                poster_path = movie.get('poster_path')
                if poster_path:
                    movie['poster_url'] = f"https://image.tmdb.org/t/p/w500{poster_path}"
                
                return movie

            return None

        except requests.RequestException as e:
            raise APIConnectionError(f"Failed to fetch movie metadata from TMDB: {str(e)}")

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """
        Get all ratings for a specific user.

        Args:
        ----
            user_id (int): ID of the user.

        Returns:
        -------
            pd.DataFrame: DataFrame containing user's ratings with movie information.

        Raises:
        ------
            DataValidationError: If data is not loaded.

        """
        if self.ratings is None or self.movies is None:
            raise DataValidationError("Ratings and movies data must be loaded")

        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        user_ratings_with_movies = user_ratings.merge(
            self.movies,
            on='movie_id',
            how='left'
        )

        return user_ratings_with_movies

    def _validate_data(self) -> None:
        """
        Validate loaded data for required columns and basic integrity.

        Raises:
        ------
            DataValidationError: If data validation fails.

        """
        # Validate ratings data
        if self.ratings is not None:
            required_rating_columns = ['user_id', 'movie_id', 'rating']
            missing_columns = [col for col in required_rating_columns if col not in self.ratings.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns in ratings data: {missing_columns}")

        # Validate movies data
        if self.movies is not None:
            required_movie_columns = ['movie_id', 'title']
            missing_columns = [col for col in required_movie_columns if col not in self.movies.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns in movies data: {missing_columns}")

    def get_data_stats(self) -> Dict[str, any]:
        """
        Get basic statistics about the loaded dataset.

        Returns:
        -------
            Dict[str, any]: Dictionary containing dataset statistics.

        """
        stats = {}
        
        if self.ratings is not None:
            # Convert numpy types to Python types for JSON serialization
            stats['num_ratings'] = int(len(self.ratings))
            stats['num_users'] = int(self.ratings['user_id'].nunique())
            stats['num_movies'] = int(self.ratings['movie_id'].nunique())
            stats['avg_rating'] = float(self.ratings['rating'].mean())
            stats['rating_scale'] = (float(self.ratings['rating'].min()), float(self.ratings['rating'].max()))
            
        if self.movies is not None:
            stats['total_movies'] = int(len(self.movies))
            
        return stats 