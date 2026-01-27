import seaborn as sns
import pandas as pd

# Import dataset iris from seaborn
df_iris = sns.load_dataset('iris')

# Import + clean dataset penguins from seaborn
df_penguins = sns.load_dataset('penguins')
df_penguins.dropna(inplace=True)

# Import + clean dataset titanic from seaborn
df_titanic = sns.load_dataset("titanic")
df_titanic = df_titanic.drop(columns=["class","who","adult_male","deck","embark_town","alive","alone"])
df_titanic.dropna(inplace=True)
df_titanic = df_titanic.convert_dtypes()

# Import dataset Churn from csv
df_churn = pd.read_csv('data/churn.csv')
