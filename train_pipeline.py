# src/train_pipeline.py
import argparse
import joblib
import os

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils import robust_read_csv, DEFAULT_FEATURES, align_and_fill

def build_pipeline(model_name='xgb', random_state=42):
    if model_name == 'xgb':
        clf = XGBClassifier(
            n_estimators=200,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )
    else:
        clf = LGBMClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )

    numeric_features = DEFAULT_FEATURES  # already numeric
    preproc = ColumnTransformer([
        ('num', StandardScaler(), numeric_features)
    ], remainder='drop')

    # Build pipeline
    pipeline = Pipeline([
        ('preproc', preproc),
        ('clf', clf)
    ])
    return pipeline

def train(csv_path, out_path, model_name='xgb', do_grid=False, test_size=0.2, random_state=42):
    print("Loading:", csv_path)
    df = robust_read_csv(csv_path)
    # Align and extract features
    X = align_and_fill(df, required_features=DEFAULT_FEATURES)
    if 'Class' in df.columns:
        y = df['Class'].astype(int)
    else:
        raise ValueError("Training CSV must contain target column 'Class'")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Apply SMOTE on training set to help imbalance
    print("Applying SMOTE to training set...")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    pipeline = build_pipeline(model_name=model_name, random_state=random_state)

    if do_grid:
        print("Running GridSearch (small grid)...")
        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [4, 6] if model_name == 'xgb' else [ -1, 10 ]
        }
        search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
        search.fit(X_res, y_res)
        best = search.best_estimator_
        print("Best params:", search.best_params_)
        pipeline = best
    else:
        pipeline.fit(X_res, y_res)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, "predict_proba") else None
    print("=== Classification report on test set ===")
    print(classification_report(y_test, y_pred, digits=4))
    if y_proba is not None:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # Save artifact
    artifact = {
        'pipeline': pipeline,
        'features': DEFAULT_FEATURES,
        'model_name': model_name
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump(artifact, out_path)
    print("Saved artifact to:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to training CSV (must include Class column)")
    parser.add_argument("--out", default="../models/model_artifact.joblib", help="output artifact path")
    parser.add_argument("--model", choices=['xgb','lgbm'], default='xgb', help="model choice")
    parser.add_argument("--grid", action='store_true', help="run a small gridsearch")
    args = parser.parse_args()
    train(args.csv, args.out, model_name=args.model, do_grid=args.grid)
