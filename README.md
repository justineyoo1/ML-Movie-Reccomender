# Machine Learning Movie Recommender

A movie recommendation system featuring **movie posters and detailed summaries** side-by-side. Built with advanced machine learning algorithms and powered by real movie data from TMDB.

![Rating Interface](images/rating.png)

![Recommendations](images/reccomendation.png)
*Get personalized recommendations with movie posters, ratings, and click-to-view details*

## Features

### **Machine Learning**
- **Collaborative Filtering** - Finds users with similar taste
- **Matrix Factorization (SVD)** - Discovers hidden preference patterns  
- **Content-Based Analysis** - Matches your favorite genres
- **Hybrid Approach** - Combines multiple algorithms for accuracy

### **TMDB Integration**
- **Original resolution posters** - Cinema-quality images
- **Complete movie data** - Plot summaries, ratings, cast information
- **Clickable movies** - Direct links to TMDB for full details
- **Real-time data** - Always up-to-date movie information

## Quick Start

**One-Click Start (Recommended)**
```bash
python start.py
```

**Manual Start**
```bash
# Terminal 1: Backend API (Port 8000)
python app.py

# Terminal 2: Frontend Interface (Port 3000)
python frontend_app.py

# Then open: http://127.0.0.1:3000
```

## How to Use

1. **Visit** http://127.0.0.1:3000
2. **Rate 5 movies** using the star rating system
3. **Read movie summaries** to make informed ratings
4. **Click movie posters** to view full details on TMDB
5. **Get instant recommendations** tailored to your taste
6. **Discover** your next favorite movie!

## Interface Walkthrough

### **Rating Phase**
- **Large movie poster** on the left (300px width, full height)
- **Movie information** on the right including:
  - Title and release year
  - Genre tags
  - **Complete plot summary** from TMDB
  - 5-star rating system
- **Two action buttons**: "Haven't Watched" or "Rate This Movie"
- **Progress tracking** shows how many more ratings needed

### **Recommendation Phase**
- **Grid of recommended movies** with posters and metadata
- **Confidence scores** showing predicted rating for you
- **Click any movie** to view full TMDB page
- **Genre-based reasoning** explaining why each movie was recommended

## Technical Stack

- **Backend**: Python Flask API with ML algorithms
- **Frontend**: Modern HTML5/CSS3/JavaScript with dark theme
- **ML Libraries**: scikit-learn, NumPy, pandas, SciPy, Optuna
- **Data Source**: MovieLens 100k dataset + TMDB API
- **Database**: 943 users, 1,682 movies, 100,000 ratings

## Project Structure

```
ml-movie-recommender/
├── start.py              # One-click startup script
├── frontend_app.py       # Frontend server (port 3000)
├── app.py                # Backend API server (port 8000)
├── templates/simple.html # Dark-themed movie interface
├── static/js/tmdb.js     # TMDB integration client
├── report_objects/       # ML recommendation engine
├── routes/              # API endpoints
│   ├── recommendations.py  # ML recommendation logic
│   ├── tmdb.py             # Movie data and posters
│   ├── models.py           # Model management
│   └── data.py             # Dataset operations
├── data/                 # MovieLens dataset
├── images/              # Screenshots for README
├── config.py             # Configuration management
└── requirements-core.txt # Python dependencies
```

## Setup & Configuration

### **Dependencies**
```bash
pip install -r requirements-core.txt
```

### **TMDB API Key (Required for movie posters)**
Create a `.env` file:
```bash
TMDB_API_KEY=your_api_key_here
TMDB_ACCESS_TOKEN=your_access_token_here
TMDB_BASE_URL=https://api.themoviedb.org/3
TMDB_IMAGE_BASE_URL=https://image.tmdb.org/t/p/original
```

Get your API key at: https://www.themoviedb.org/settings/api

## Machine Learning Algorithms

The system uses three sophisticated ML approaches:

### **1. Matrix Factorization (SVD)**
```
R ≈ U × Σ × V^T
```
Decomposes the user-movie rating matrix to find latent factors

### **2. User-Based Collaborative Filtering**
```
r̂ui = Σ(sim(u,k) × rki) / Σ|sim(u,k)|
```
Finds users with similar preferences and recommends their favorite movies

### **3. Content-Based Filtering**
Analyzes movie genres, cast, and metadata to find similar films

All algorithms are optimized using **Optuna** for hyperparameter tuning.



---

**Ready to discover your next favorite movie?**

```bash
python start.py
```

**Visit: http://127.0.0.1:3000**

*A  machine learning project that actually works.* 
