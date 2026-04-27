# Movie Rating Prediction

This project builds a regression model to predict movie ratings from historical data and metadata.

## Objective
Predict a movie's average rating using features such as:
- Genres
- Director
- Main actors
- Runtime
- Popularity
- Release year
- Vote count

## Dataset Files Used
- `ratings_small.csv` (target signal)
- `links_small.csv` (MovieLens to TMDB ID mapping)
- `movies_metadata.csv` (movie attributes and genres)
- `credits.csv` (cast and crew for actors/director)

## Approach
1. Aggregate historical ratings per movie into:
   - `mean_rating` (prediction target)
   - `rating_count` (used for filtering noisy samples)
2. Join ratings with metadata and credits.
3. Parse and engineer text features:
   - `genre_*`
   - `director_*`
   - `actor_*` (top 3 billed cast)
4. Build a combined feature space:
   - TF-IDF on engineered text tokens
   - Standardized numeric columns (`runtime`, `popularity`, `vote_count`, `release_year`)
5. Train and compare regression models:
   - Ridge Regression
   - ElasticNet Regression
6. Evaluate with:
   - MAE
   - RMSE
   - R^2

## Run
From this folder, run:

```powershell
& ".venv/Scripts/python.exe" "movie_rating_prediction.py"
```

Install dependencies first:

```powershell
& ".venv/Scripts/python.exe" -m pip install -r requirements.txt
```

Optional arguments:

```powershell
& ".venv/Scripts/python.exe" "movie_rating_prediction.py" --min-votes 20 --ratings "ratings_small.csv" --links "links_small.csv" --metadata "movies_metadata.csv" --credits "credits.csv"
```

## Expected Output
- Dataset shape after preprocessing
- Number of movies used
- Average target rating
- Model comparison on test split (MAE, RMSE, R^2)
- Sample predictions vs actual ratings

## Notes
- Default uses `ratings_small.csv` and `links_small.csv` for faster iteration.
- You can switch to the full ratings files by passing `--ratings ratings.csv --links links.csv` (this is slower and more memory-intensive).
