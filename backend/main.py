import os
import uvicorn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException
from typing import Literal
from module.cleandataset import df_iris, df_penguins, df_titanic, df_churn
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

class Data(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset : dict
    x_axis: str = Field(min_length=1, description="Data use in the x_axis of the scatterplot")
    y_axis: str = Field(min_length=1, description="Data use in the y_axis of the scatterplot")
    color: str = Field(min_length=1, description="Color used for the data points")
    size: str = Field(min_length=1, description="Size used for the data points")

class GraphResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    graph : plt.Axes

class TrainingConfig(BaseModel):
    max_iter : int = Field(description="The maximum number of iterations")
    C : float = Field(description="The regularization parameter")
    solver : Literal["lbfgs", "liblinear", "newton-cg", "sag", "saga", "newton-cholesky"] = Field(default="lbfgs", description="The optimization algorithm")
    # TODO : Forcer les L1 en fonction du solver
    l1_ratio : float = Field(min=0, max=1, description="The L1 ratio")

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

@app.get("/columns")
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
        dataset = show_dataset(request.name)["dataset"]
        list_columns = dataset.columns.tolist()
        return {"columns": list_columns}
    except Exception as e:
        logger.error(f"Error code 500 in API get /columns : {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

@app.post("/graph")
def show_graph(request: Data):
    '''
    Endpoint to show graph of dataset
    Args : 
       request (Data): Pydantic Request with dataset and columns used in making the graph
    Returns :
        graph (pyplot.figure): A plot figure representing the graph of the dataset
    '''
    logger.info(f"Appel API : Selection GRAPH")
    try : 
        dataset = request.dataset
        df_dataset = pd.DataFrame(dataset)
        x_axis = request.x_axis
        y_axis = request.y_axis
        color = request.color
        size = request.size
        graph = sns.scatterplot(x=x_axis, y=y_axis, hue=color, size=size, data=df_dataset)
        return graph.figure
    except Exception as e:
        logger.error(f"Error code 500 in API post /graph : {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

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

