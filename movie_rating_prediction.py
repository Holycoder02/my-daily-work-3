import argparse
import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: float


def safe_literal_eval(value):
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except (ValueError, SyntaxError):
        return []


def extract_genres(genres_raw):
    genres = safe_literal_eval(genres_raw)
    names = [g.get("name", "") for g in genres if isinstance(g, dict)]
    return " ".join([f"genre_{name.strip().lower().replace(' ', '_')}" for name in names if name])


def extract_director(crew_raw):
    crew = safe_literal_eval(crew_raw)
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            name = member.get("name", "").strip().lower().replace(" ", "_")
            if name:
                return f"director_{name}"
    return "director_unknown"


def extract_top_cast(cast_raw, top_n=3):
    cast = safe_literal_eval(cast_raw)
    actors = []
    for member in cast:
        if isinstance(member, dict):
            name = member.get("name", "").strip().lower().replace(" ", "_")
            if name:
                actors.append(name)
        if len(actors) >= top_n:
            break
    if not actors:
        return "actor_unknown"
    return " ".join([f"actor_{name}" for name in actors])


def build_feature_dataset(ratings_path, links_path, metadata_path, credits_path, min_votes):
    ratings = pd.read_csv(ratings_path, usecols=["movieId", "rating"])
    links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])
    metadata = pd.read_csv(
        metadata_path,
        usecols=["id", "title", "genres", "runtime", "popularity", "release_date", "vote_count"],
        low_memory=False,
    )
    credits = pd.read_csv(credits_path, usecols=["id", "cast", "crew"])

    ratings_agg = ratings.groupby("movieId").agg(mean_rating=("rating", "mean"), rating_count=("rating", "count")).reset_index()
    ratings_agg = ratings_agg[ratings_agg["rating_count"] >= min_votes]

    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    metadata["id"] = pd.to_numeric(metadata["id"], errors="coerce")
    metadata = metadata.dropna(subset=["id"])
    metadata["id"] = metadata["id"].astype(int)

    credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
    credits = credits.dropna(subset=["id"])
    credits["id"] = credits["id"].astype(int)

    merged = ratings_agg.merge(links, on="movieId", how="inner")
    merged = merged.merge(metadata, left_on="tmdbId", right_on="id", how="inner")
    merged = merged.merge(credits[["id", "cast", "crew"]], on="id", how="left", suffixes=("", "_credits"))

    merged["genre_features"] = merged["genres"].apply(extract_genres)
    merged["director_feature"] = merged["crew"].apply(extract_director)
    merged["cast_features"] = merged["cast"].apply(extract_top_cast)
    merged["combined_features"] = (
        merged["genre_features"].fillna("")
        + " "
        + merged["director_feature"].fillna("director_unknown")
        + " "
        + merged["cast_features"].fillna("actor_unknown")
    ).str.strip()

    merged["runtime"] = pd.to_numeric(merged["runtime"], errors="coerce")
    merged["popularity"] = pd.to_numeric(merged["popularity"], errors="coerce")
    merged["vote_count"] = pd.to_numeric(merged["vote_count"], errors="coerce")
    merged["release_year"] = pd.to_datetime(merged["release_date"], errors="coerce").dt.year

    model_data = merged[
        [
            "movieId",
            "title",
            "combined_features",
            "runtime",
            "popularity",
            "vote_count",
            "release_year",
            "mean_rating",
            "rating_count",
        ]
    ].dropna(subset=["mean_rating", "combined_features"])

    return model_data


def evaluate_models(df):
    feature_cols = ["combined_features", "runtime", "popularity", "vote_count", "release_year"]
    target_col = "mean_rating"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = ["runtime", "popularity", "vote_count", "release_year"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000),
                "combined_features",
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
        ]
    )

    model_candidates = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.0008, l1_ratio=0.2, max_iter=5000),
    }

    results = []
    best_model = None
    best_rmse = np.inf

    for model_name, model in model_candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)
        results.append(ModelResult(model_name, mae, rmse, r2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

    return results, best_model, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Movie rating prediction using metadata and cast/crew features")
    parser.add_argument("--ratings", default="ratings_small.csv", help="Path to ratings CSV")
    parser.add_argument("--links", default="links_small.csv", help="Path to links CSV")
    parser.add_argument("--metadata", default="movies_metadata.csv", help="Path to movies metadata CSV")
    parser.add_argument("--credits", default="credits.csv", help="Path to credits CSV")
    parser.add_argument("--min-votes", type=int, default=20, help="Minimum number of ratings per movie")
    args = parser.parse_args()

    df = build_feature_dataset(
        ratings_path=args.ratings,
        links_path=args.links,
        metadata_path=args.metadata,
        credits_path=args.credits,
        min_votes=args.min_votes,
    )

    print(f"Prepared dataset shape: {df.shape}")
    print(f"Movies included after filters: {df['movieId'].nunique()}")
    print(f"Average target rating: {df['mean_rating'].mean():.3f}")

    results, best_model, X_test, y_test = evaluate_models(df)
    print("\nModel performance on test split:")
    for r in sorted(results, key=lambda x: x.rmse):
        print(f"- {r.name}: MAE={r.mae:.4f}, RMSE={r.rmse:.4f}, R2={r.r2:.4f}")

    # Display a small sample of predictions for quick qualitative inspection.
    sample = X_test.head(5).copy()
    sample_preds = best_model.predict(sample)
    print("\nSample predictions:")
    for idx, pred in zip(sample.index, sample_preds):
        print(f"- idx={idx}: predicted={pred:.3f}, actual={y_test.loc[idx]:.3f}")


if __name__ == "__main__":
    main()
