#!/usr/bin/env python3
"""
Data Setup Helper for ML Movie Recommender System.

This script downloads and prepares MovieLens datasets for use with the recommendation system.
It handles downloading, extracting, and validating the data files.
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from config import Config


class DataSetup:
    """Helper class for downloading and setting up MovieLens datasets."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize DataSetup with specified data path.
        
        Args:
        ----
            data_path (Optional[str]): Path to store data files. Uses config default if None.
        """
        self.data_path = Path(data_path or Config.DATA_PATH)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, dataset_size: str = '100k', force_download: bool = False) -> bool:
        """
        Download MovieLens dataset.
        
        Args:
        ----
            dataset_size (str): Size of dataset to download ('100k', '1m', '10m', '20m', '25m').
            force_download (bool): Whether to re-download if dataset already exists.
            
        Returns:
        -------
            bool: True if download was successful, False otherwise.
        """
        if dataset_size not in Config.SUPPORTED_DATASETS:
            print(f"❌ Unsupported dataset size: {dataset_size}")
            print(f"✅ Supported sizes: {', '.join(Config.SUPPORTED_DATASETS)}")
            return False
        
        dataset_dir = self.data_path / f"ml-{dataset_size}"
        zip_path = self.data_path / f"ml-{dataset_size}.zip"
        
        # Check if dataset already exists
        if dataset_dir.exists() and not force_download:
            print(f"✅ Dataset ml-{dataset_size} already exists. Use force_download=True to re-download.")
            return True
        
        # Get download URL
        url = Config.get_movielens_url(dataset_size)
        print(f"📥 Downloading MovieLens {dataset_size} dataset from {url}")
        
        try:
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Simple progress indicator
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r📊 Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ Download completed: {zip_path}")
            
            # Extract the dataset
            print(f"📂 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)
            
            print(f"✅ Extraction completed: {dataset_dir}")
            
            # Clean up zip file
            zip_path.unlink()
            print(f"🗑️  Cleaned up zip file")
            
            # Validate the extracted data
            if self.validate_dataset(dataset_size):
                print(f"✅ Dataset validation passed!")
                return True
            else:
                print(f"❌ Dataset validation failed!")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            return False
        except zipfile.BadZipFile as e:
            print(f"❌ Extraction failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def validate_dataset(self, dataset_size: str) -> bool:
        """
        Validate that the dataset has been properly downloaded and extracted.
        
        Args:
        ----
            dataset_size (str): Size of dataset to validate.
            
        Returns:
        -------
            bool: True if validation passes, False otherwise.
        """
        dataset_dir = self.data_path / f"ml-{dataset_size}"
        
        if not dataset_dir.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return False
        
        # Check for required files (varies by dataset)
        required_files = self._get_required_files(dataset_size)
        missing_files = []
        
        for file_name in required_files:
            file_path = dataset_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            return False
        
        # Try to load and validate data structure
        try:
            self._validate_data_structure(dataset_size)
            return True
        except Exception as e:
            print(f"❌ Data structure validation failed: {e}")
            return False
    
    def _get_required_files(self, dataset_size: str) -> List[str]:
        """Get list of required files for a dataset size."""
        if dataset_size == '100k':
            return ['u.data', 'u.item', 'u.user']
        else:
            return ['ratings.dat', 'movies.dat', 'users.dat'] if dataset_size == '1m' else ['ratings.csv', 'movies.csv']
    
    def _validate_data_structure(self, dataset_size: str) -> None:
        """Validate the structure of the dataset files."""
        dataset_dir = self.data_path / f"ml-{dataset_size}"
        
        if dataset_size == '100k':
            # Validate 100k dataset
            ratings_path = dataset_dir / 'u.data'
            ratings_df = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
            
            if ratings_df.empty:
                raise ValueError("Ratings file is empty")
            
            required_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
            if not all(col in ratings_df.columns for col in required_columns):
                raise ValueError(f"Missing columns in ratings file. Expected: {required_columns}")
        
        elif dataset_size == '1m':
            # Validate 1M dataset
            ratings_path = dataset_dir / 'ratings.dat'
            ratings_df = pd.read_csv(ratings_path, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
            
            if ratings_df.empty:
                raise ValueError("Ratings file is empty")
        
        else:
            # Validate larger datasets (CSV format)
            ratings_path = dataset_dir / 'ratings.csv'
            ratings_df = pd.read_csv(ratings_path)
            
            if ratings_df.empty:
                raise ValueError("Ratings file is empty")
    
    def get_dataset_info(self, dataset_size: str) -> Dict:
        """
        Get information about a dataset.
        
        Args:
        ----
            dataset_size (str): Size of dataset to analyze.
            
        Returns:
        -------
            Dict: Dataset information including statistics.
        """
        if not self.validate_dataset(dataset_size):
            return {'error': 'Dataset not found or invalid'}
        
        dataset_dir = self.data_path / f"ml-{dataset_size}"
        
        try:
            # Load ratings data
            if dataset_size == '100k':
                ratings_path = dataset_dir / 'u.data'
                ratings_df = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
            elif dataset_size == '1m':
                ratings_path = dataset_dir / 'ratings.dat'
                ratings_df = pd.read_csv(ratings_path, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
            else:
                ratings_path = dataset_dir / 'ratings.csv'
                ratings_df = pd.read_csv(ratings_path)
            
            info = {
                'dataset_size': dataset_size,
                'num_ratings': len(ratings_df),
                'num_users': ratings_df['user_id'].nunique(),
                'num_movies': ratings_df['movie_id'].nunique(),
                'rating_scale': (ratings_df['rating'].min(), ratings_df['rating'].max()),
                'avg_rating': ratings_df['rating'].mean(),
                'sparsity': 1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['movie_id'].nunique())),
                'data_path': str(dataset_dir)
            }
            
            return info
            
        except Exception as e:
            return {'error': f'Failed to analyze dataset: {e}'}
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets in the data directory."""
        available = []
        for dataset_size in Config.SUPPORTED_DATASETS:
            if self.validate_dataset(dataset_size):
                available.append(dataset_size)
        return available
    
    def cleanup_dataset(self, dataset_size: str) -> bool:
        """
        Remove a dataset from the data directory.
        
        Args:
        ----
            dataset_size (str): Size of dataset to remove.
            
        Returns:
        -------
            bool: True if cleanup was successful, False otherwise.
        """
        dataset_dir = self.data_path / f"ml-{dataset_size}"
        
        if not dataset_dir.exists():
            print(f"ℹ️  Dataset ml-{dataset_size} does not exist")
            return True
        
        try:
            import shutil
            shutil.rmtree(dataset_dir)
            print(f"🗑️  Removed dataset: {dataset_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to remove dataset: {e}")
            return False


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MovieLens Dataset Setup Helper")
    parser.add_argument('--dataset', '-d', default='100k', 
                       choices=Config.SUPPORTED_DATASETS,
                       help='Dataset size to download (default: 100k)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force re-download even if dataset exists')
    parser.add_argument('--info', '-i', action='store_true',
                       help='Show information about existing datasets')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available datasets')
    parser.add_argument('--cleanup', '-c', 
                       help='Remove specified dataset')
    
    args = parser.parse_args()
    
    setup = DataSetup()
    
    if args.list:
        available = setup.list_available_datasets()
        print(f"📋 Available datasets: {', '.join(available) if available else 'None'}")
        return
    
    if args.info:
        available = setup.list_available_datasets()
        for dataset_size in available:
            info = setup.get_dataset_info(dataset_size)
            print(f"\n📊 Dataset: {dataset_size}")
            for key, value in info.items():
                if key != 'dataset_size':
                    print(f"   {key}: {value}")
        return
    
    if args.cleanup:
        setup.cleanup_dataset(args.cleanup)
        return
    
    # Download dataset
    print(f"🚀 Starting setup for MovieLens {args.dataset} dataset")
    if setup.download_dataset(args.dataset, args.force):
        info = setup.get_dataset_info(args.dataset)
        print(f"\n📊 Dataset Information:")
        for key, value in info.items():
            if key != 'error':
                print(f"   {key}: {value}")
        print(f"\n✅ Setup completed successfully!")
    else:
        print(f"\n❌ Setup failed!")


if __name__ == "__main__":
    main() 