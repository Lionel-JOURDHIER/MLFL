from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, classification_report, confusion_matrix, mean_squared_error
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
import json

target_dict = {
    "iris": "species",
    "penguins": "species",
    "titanic": "survived",
    "churn": "Exited"
}


def training_model(
        dataset:dict, 
        dataset_name:str,
        max_iter: int = 1000, 
        C: float = 1.0, 
        solver: str = "lbfgs"
        ):
    
    df_dataset = pd.DataFrame(dataset)
    target = target_dict[dataset_name]

    X = df_dataset.drop(columns=[target])
    y = df_dataset[target]

    # Transformation des données
    num_features = X.select_dtypes(include=['int', 'float']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    bool_features = X.select_dtypes(include=['bool']).columns.tolist()

    for category in cat_features:
        X = pd.get_dummies(X, columns=[category])

    for bool in bool_features:
        X = pd.get_dummies(X, columns=[bool])

    #Création du preprocessor pour préparation des données
    scaler = StandardScaler()

    # Division Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # On Scale les données numériques
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])


    # On resample les donnée pour équilibrer les résultats.
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)

    # Fit du model 
    model.fit(X_resampled, y_resampled)
    return model, X_test, y_test, X_resampled, y_resampled

def metrics_model(model, X_test, y_test):
    # Prediction sur le test set
    y_pred = model.predict(X_test)
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, digits=4)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Affichage des métriques
    conf_matrix_json = {"matrix": conf_matrix.tolist()}


    return accuracy, class_report, conf_matrix_json