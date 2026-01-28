import os
import uvicorn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException
from typing import Literal
from module.cleandataset import df_iris, df_penguins, df_titanic, df_churn
from module.scikitlearn import metrics_model, training_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_PORT = int(os.getenv('FASTAPI_PORT'))
API_URL = os.getenv('API_ROOT_URL')
API_ROOT_URL = f"http://{API_URL}:{API_PORT}"

# Initialisation of the logger
os.makedirs("../logs", exist_ok=True)
logger.add("../logs/agent_api.log", rotation="10 MB", retention="7 days", level="INFO")
logger.info("Vous etes logarisé dans agent_api.log")

# Creation of the FastAPI application
logger.info("Création de l'API")
app = FastAPI(title="API MachineLearning")

# Definition of Pydantic models
class Dataset(BaseModel):
    name: Literal["iris","penguins","titanic","churn"] = Field(min_length=1, description="The name of the dataset")

class DatasetResponse(BaseModel):
    dataset : dict

class DatasetColumnsResponse(BaseModel):
    columns : list[str]

class TrainingConfig(BaseModel):
    dataset_name : Literal["iris","penguins","titanic","churn"] = Field(min_length=1, description="The name of the dataset")
    dataset : dict
    max_iter : int = Field(description="The maximum number of iterations")
    C : float = Field(description="The regularization parameter")
    solver : Literal["lbfgs", "liblinear", "newton-cg", "sag", "saga", "newton-cholesky"] = Field(default="lbfgs", description="The optimization algorithm")

class TrainingResponse(BaseModel):
    name : str
    accuracy_score : float = Field(description="The accuracy of the model")
    classification_report_test : str = Field(description="The classification report of the model")
    classification_report_train : str = Field(description="The classification report of the model during training")
    confusion_matrix : dict

class Model(BaseModel):
    name: str = Field(min_length=1, description="The name of the model")

@app.get("/")
def root():
    '''
    Root endpoint to check if the API is running
    '''
    return {"message": "API is running"}

@app.post("/dataset")
def show_dataset(request: Dataset):
    '''
    Endpoint to show dataset information
    Args : 
        request (Dataset): The dataset object containing the size of the data points
    Returns :
        dataset (DataFrame): The requested dataset
    '''
    logger.info(f"Appel API : Selection DATASET : {request.name}")
    dataset_dict ={
        "iris": df_iris,
        "penguins": df_penguins,
        "titanic": df_titanic,
        "churn": df_churn
    }
    try : 
        if request.name : 
            dataset = dataset_dict.get(request.name)
            df_dict = dataset.to_dict()
            return {"dataset": df_dict}
        else : 
            return {"message":"Please provide a valid dataset name"}
    except Exception as e:
        logger.error(f"Error code 500 in API get /dataset : {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

@app.post("/columns")
def show_columns(request: Dataset):
    '''
    Endpoint to show dataset columns
    Args : 
        request (Dataset): The dataset object containing the size of the data points
    Returns :
        columns (list): A list of columns in the dataset
    '''
    logger.info(f"Appel API : Selection COLONNES : {request.name}")
    try : 
        dict_dataset = show_dataset(request)["dataset"]
        dataset = pd.DataFrame(dict_dataset)
        list_columns = dataset.columns.tolist()
        return {"columns": list_columns}
    except Exception as e:
        logger.error(f"Error code 500 in API get /columns : {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

@app.post("/train")
def train_model(request: TrainingConfig):
    '''
    Endpoint to train a model on the dataset
    Args : 
        request (Dataset): The dataset object containing the size of the data points
    Returns :
        response (dict): A dictionary containing the trained model and its evaluation metrics
    '''
    # Create model name and load the dataset
    maintenant = dt.datetime.now()
    horodatage = maintenant.strftime("%d/%m/%Y %H:%M:%S")
    model_name = request.dataset_name + "_" + horodatage
    logger.info(f"Appel API : Train MODEL {model_name}")
    try : 
        model, X_test, y_test, X_resampled, y_resampled = training_model(
            request.dataset,
            request.dataset_name,
            max_iter=request.max_iter,
            C=request.C,
            solver = request.solver)
        
        accuracy_test, class_report_test, conf_matrix_json_test = metrics_model(model, X_test, y_test)
        accuracy_train, class_report_train, conf_matrix_json_train = metrics_model(model, X_resampled, y_resampled)
        response = {
            "name" : model_name,
            "accuracy_score" : accuracy_test,
            "classification_report_test" : class_report_test,
            "classification_report_train" : class_report_train,
            "confusion_matrix" : conf_matrix_json_test
        }
        return response
    except Exception as e :
        logger.error(f"Erreur lors du train MODEL {model_name}")
        return {"error": str(e)}






if __name__ == "__main__" :

    # 1 on récupère le port de l'API
    try : 
        port = API_PORT
        host = API_URL
        url = API_ROOT_URL
    except ValueError :
        print("Erreur : FASTAPI_PORT invalide, utilisation du port par défaut 8000.")
        port = 8000

    logger.info(f"Démarrage du serveur FastAPI sur {API_ROOT_URL}")

    # 2. On lance uvicorn
    uvicorn.run(
        "main:app",
        reload = True,
        port = port,
        host = host,
        log_level="debug"
    )

