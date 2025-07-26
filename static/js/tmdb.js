/**
 * TMDB (The Movie Database) JavaScript utilities
 * Handles movie poster fetching and TMDB links
 */

class TMDBClient {
    constructor() {
        this.imageBaseUrl = 'https://image.tmdb.org/t/p/original'; // Highest resolution
        this.tmdbBaseUrl = 'https://www.themoviedb.org/movie';
        this.cache = new Map(); // Cache movie data
    }

    /**
     * Get movie data from TMDB (poster, overview, etc.)
     * @param {string} title - Movie title
     * @param {string} year - Movie year
     * @returns {Promise<Object|null>} Movie data or null if not found
     */
    async getMovieData(title, year) {
        const cacheKey = `${title}_${year}`;
        
        // Check cache first
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            // Clean up title for search
            const cleanTitle = this.cleanMovieTitle(title);
            
            const response = await fetch(`http://127.0.0.1:8000/api/tmdb/search?title=${encodeURIComponent(cleanTitle)}&year=${year}`);
            
            if (!response.ok) {
                this.cache.set(cacheKey, null);
                return null;
            }

            const data = await response.json();
            
            if (data.poster_url) {
                this.cache.set(cacheKey, data);
                return data;
            }

            this.cache.set(cacheKey, null);
            return null;

        } catch (error) {
            console.warn(`Failed to fetch movie data for ${title}:`, error);
            this.cache.set(cacheKey, null);
            return null;
        }
    }

    /**
     * Get movie poster URL from TMDB (legacy method for compatibility)
     * @param {string} title - Movie title
     * @param {string} year - Movie year
     * @returns {Promise<string|null>} Poster URL or null if not found
     */
    async getMoviePoster(title, year) {
        const movieData = await this.getMovieData(title, year);
        return movieData ? movieData.poster_url : null;
    }

    /**
     * Get TMDB movie page URL
     * @param {string} title - Movie title
     * @param {string} year - Movie year
     * @returns {Promise<string|null>} TMDB URL or null if not found
     */
    async getMovieUrl(title, year) {
        const cacheKey = `url_${title}_${year}`;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            const cleanTitle = this.cleanMovieTitle(title);
            
            const response = await fetch(`http://127.0.0.1:8000/api/tmdb/search?title=${encodeURIComponent(cleanTitle)}&year=${year}`);
            
            if (!response.ok) {
                this.cache.set(cacheKey, null);
                return null;
            }

            const data = await response.json();
            
            if (data.tmdb_id) {
                const url = `${this.tmdbBaseUrl}/${data.tmdb_id}`;
                this.cache.set(cacheKey, url);
                return url;
            }

            this.cache.set(cacheKey, null);
            return null;

        } catch (error) {
            console.warn(`Failed to fetch TMDB URL for ${title}:`, error);
            this.cache.set(cacheKey, null);
            return null;
        }
    }

    /**
     * Clean movie title for better TMDB search results
     * @param {string} title - Raw movie title
     * @returns {string} Cleaned title
     */
    cleanMovieTitle(title) {
        return title
            .replace(/\(\d{4}\)/g, '') // Remove year from title
            .replace(/,\s*The$/i, '') // Remove trailing "The"
            .replace(/^The\s+/i, '') // Remove leading "The"
            .trim();
    }

    /**
     * Extract year from movie title
     * @param {string} title - Movie title with year
     * @returns {string|null} Year or null if not found
     */
    extractYear(title) {
        const match = title.match(/\((\d{4})\)/);
        return match ? match[1] : null;
    }

    /**
     * Load movie poster into an image element
     * @param {HTMLImageElement} imgElement - Image element to load poster into
     * @param {string} title - Movie title
     * @param {string} year - Movie year
     * @param {string} fallbackEmoji - Fallback emoji if poster not found
     */
    async loadMoviePoster(imgElement, title, year, fallbackEmoji = '🎬') {
        // Set initial state
        imgElement.style.opacity = '0.5';
        imgElement.alt = `${title} poster`;

        try {
            const posterUrl = await this.getMoviePoster(title, year);
            
            if (posterUrl) {
                // Create a new image to test loading
                const testImg = new Image();
                testImg.onload = () => {
                    imgElement.src = posterUrl;
                    imgElement.style.opacity = '1';
                    imgElement.classList.add('has-poster');
                };
                testImg.onerror = () => {
                    this.setFallbackPoster(imgElement, fallbackEmoji);
                };
                testImg.src = posterUrl;
            } else {
                this.setFallbackPoster(imgElement, fallbackEmoji);
            }
        } catch (error) {
            console.warn(`Error loading poster for ${title}:`, error);
            this.setFallbackPoster(imgElement, fallbackEmoji);
        }
    }

    /**
     * Set fallback poster when TMDB poster is not available
     * @param {HTMLImageElement} imgElement - Image element
     * @param {string} emoji - Fallback emoji
     */
    setFallbackPoster(imgElement, emoji) {
        // Create a canvas with the emoji
        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 300;
        const ctx = canvas.getContext('2d');
        
        // Set background
        ctx.fillStyle = '#e5e7eb';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw emoji
        ctx.font = '80px serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#6b7280';
        ctx.fillText(emoji, canvas.width / 2, canvas.height / 2);
        
        // Set as image source
        imgElement.src = canvas.toDataURL();
        imgElement.style.opacity = '1';
        imgElement.classList.add('fallback-poster');
    }

    /**
     * Make a movie element clickable to open TMDB page
     * @param {HTMLElement} element - Element to make clickable
     * @param {string} title - Movie title
     * @param {string} year - Movie year
     */
    async makeMovieClickable(element, title, year) {
        try {
            const url = await this.getMovieUrl(title, year);
            
            if (url) {
                element.style.cursor = 'pointer';
                element.title = `Click to view ${title} on TMDB`;
                
                element.addEventListener('click', (e) => {
                    // Don't interfere with rating functionality
                    if (e.target.classList.contains('star') || e.target.closest('.rating-stars')) {
                        return;
                    }
                    
                    window.open(url, '_blank');
                });

                // Add visual feedback
                element.addEventListener('mouseenter', () => {
                    if (!element.classList.contains('rating-stars')) {
                        element.style.transform = 'scale(1.02)';
                    }
                });

                element.addEventListener('mouseleave', () => {
                    if (!element.classList.contains('rating-stars')) {
                        element.style.transform = 'scale(1)';
                    }
                });
            }
        } catch (error) {
            console.warn(`Failed to make ${title} clickable:`, error);
        }
    }

    /**
     * Preload posters for multiple movies
     * @param {Array} movies - Array of movie objects with title and year
     */
    async preloadPosters(movies) {
        const promises = movies.map(movie => {
            const year = this.extractYear(movie.title) || movie.year;
            return this.getMoviePoster(movie.title, year);
        });

        try {
            await Promise.allSettled(promises);
            console.log('Movie posters preloaded');
        } catch (error) {
            console.warn('Error preloading posters:', error);
        }
    }
}

// Create global TMDB client instance
window.tmdbClient = new TMDBClient();

console.log('TMDB client initialized successfully!'); 