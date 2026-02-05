from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_linear_probe(X_train, y_train, X_test, y_test, C=1.0, max_iter=1500, seed=0, scale=True):
    if scale:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver="lbfgs",
                n_jobs=-1,
                random_state=seed
            ))
        ])
    else:
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=-1,
            random_state=seed
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc
