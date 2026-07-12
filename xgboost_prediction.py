#%%
import pandas as pd
import numpy as np
import json
import os
import zipfile

#%%

# Creo el directorio y archivo kaggle.json
kaggle_dir = os.path.expanduser("~/.kaggle")
kaggle_file = os.path.join(kaggle_dir, "kaggle.json")

if not os.path.exists(kaggle_dir):
    os.makedirs(kaggle_dir)

# Copio el contenido del archivo kaggle.json
kaggle_json_content = '''
{
  "username": "sofagonzlezdelsolar",
  "key": "6530003e23165af85a76188f36288045"
}
'''
with open(kaggle_file, 'w') as f:
    f.write(kaggle_json_content)

# Cambio los permisos del archivo kaggle.json
os.chmod(kaggle_file, 0o600)

# %%
#Guardo el Dataframe Train
dftrain = pd.read_csv(r'C:\Users\sofia\Documents\Tp2 Análisis Predictivo\base_train.csv')
dfval = pd.read_csv(r'C:\Users\sofia\Documents\Tp2 Análisis Predictivo\base_val.csv')

#%%
#PREPARACIÓN DE LA BASE

#BORRO COLUMNAS VACIAS O INNECESARIAS
# Verifico si existen columnas que tienen todas las filas con valores faltantes
Columnas_vacias = dftrain.columns[dftrain.isna().all()].tolist()
dftrain = dftrain.drop(Columnas_vacias, axis=1)

#Borro las columnas que estoy segura que no sirven para la predicción
columnas_eliminar = ['host_id','calendar_last_scraped']
dftrain = dftrain.drop(columnas_eliminar, axis=1)
dftrain.columns

#%%
#PREPARACIÓN DE LA BASE

#BORRO COLUMNAS VACIAS O INNECESARIAS
# Verifico si existen columnas que tienen todas las filas con valores faltantes
Columnas_vacias = dfval.columns[dfval.isna().all()].tolist()
dfval = dfval.drop(Columnas_vacias, axis=1)

#Borro las columnas que estoy segura que no sirven para la predicción
columnas_eliminar = ['host_id','calendar_last_scraped']
dfval = dfval.drop(columnas_eliminar, axis=1)
dfval.columns


#%%
dftrain_cat = dftrain.select_dtypes(include=['object']).columns
dftrain_num = dftrain.select_dtypes(exclude=['object']).columns


#Imputo Missings

from feature_engine.imputation import CategoricalImputer
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder

# Imputación de variables categóricas

imputercat = CategoricalImputer(fill_value="Null", return_object=True, variables=dftrain_cat.tolist())
dftrain[dftrain_cat] = imputercat.fit_transform(dftrain[dftrain_cat])

imputernum = SimpleImputer(strategy='mean')
dftrain[dftrain_num] = imputernum.fit_transform(dftrain[dftrain_num])

#%%
#Imputo los outliers de beds

def imputar_outliers_por_mediana(data, variable, umbral=1.5):
    # Calcula la mediana de la variable
    mediana = np.median(data[variable])

    # Calcula el rango intercuartílico
    q1 = np.percentile(data[variable], 25)
    q3 = np.percentile(data[variable], 75)
    rango_intercuartilico = q3 - q1

    # Calcula los límites inferior y superior para detectar los outliers
    limite_inferior = q1 - umbral * rango_intercuartilico
    limite_superior = q3 + umbral * rango_intercuartilico

    # Imputa los valores atípicos por la mediana
    data.loc[(data[variable] < limite_inferior) | (data[variable] > limite_superior), variable] = mediana

    return data

# Aplicar la función para cambiar los outliers de la variable "beds" por la mediana
dftrain = imputar_outliers_por_mediana(dftrain, "beds")

#%%
dfval_cat = dfval.select_dtypes(include=['object']).columns
dfval_num = dfval.select_dtypes(exclude=['object']).columns


#Imputo Missings

from feature_engine.imputation import CategoricalImputer
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder

# Imputación de variables categóricas

imputercat = CategoricalImputer(fill_value="Null", return_object=True, variables=dfval_cat.tolist())
dfval[dfval_cat] = imputercat.fit_transform(dfval[dfval_cat])

imputernum = SimpleImputer(strategy='mean')
dfval[dfval_num] = imputernum.fit_transform(dfval[dfval_num])

#%%
#Imputo los outliers de beds

def imputar_outliers_por_mediana(data, variable, umbral=1.5):
    # Calcula la mediana de la variable
    mediana = np.median(data[variable])

    # Calcula el rango intercuartílico
    q1 = np.percentile(data[variable], 25)
    q3 = np.percentile(data[variable], 75)
    rango_intercuartilico = q3 - q1

    # Calcula los límites inferior y superior para detectar los outliers
    limite_inferior = q1 - umbral * rango_intercuartilico
    limite_superior = q3 + umbral * rango_intercuartilico

    # Imputa los valores atípicos por la mediana
    data.loc[(data[variable] < limite_inferior) | (data[variable] > limite_superior), variable] = mediana

    return data

# Aplicar la función para cambiar los outliers de la variable "beds" por la mediana
dfval = imputar_outliers_por_mediana(dfval, "beds")

#%%
from textblob import TextBlob

dftrain['sentiment_score_description'] = dftrain['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
dftrain['sentiment_score_neighborhood_overview'] = dftrain['neighborhood_overview'].apply(lambda x: TextBlob(x).sentiment.polarity)
dftrain['sentiment_score_host_about'] = dftrain['host_about'].apply(lambda x: TextBlob(x).sentiment.polarity)
dftrain['sentiment_score_name'] = dftrain['name'].apply(lambda x: TextBlob(x).sentiment.polarity)

#%%

dfval['sentiment_score_description'] = dfval['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
dfval['sentiment_score_neighborhood_overview'] = dfval['neighborhood_overview'].apply(lambda x: TextBlob(x).sentiment.polarity)
dfval['sentiment_score_host_about'] = dfval['host_about'].apply(lambda x: TextBlob(x).sentiment.polarity)
dfval['sentiment_score_name'] = dfval['name'].apply(lambda x: TextBlob(x).sentiment.polarity)

#%%

#TRANSFORMO LAS CATEGORICAS 

#Ahora reemplazo/agrego las columnas que solo me interesa saber si hay info
#Quiero saber si hay info en la columna name y el largo de el name
dftrain['name'] = np.where(dftrain['name']=='Null', 0, dftrain['name'].apply(lambda x: len(str(x))))
#Lo mismo con otras columnas
dftrain['description?'] = np.where(dftrain['description']=='Null', 0, 1)
dftrain['description'] = np.where(dftrain['description']=='Null', 0, dftrain['description'].apply(lambda x: len(str(x))))
dftrain['neighborhood_overview?'] = np.where(dftrain['neighborhood_overview']=='Null', 0, 1)
dftrain['neighborhood_overview'] = np.where(dftrain['neighborhood_overview']=='Null', 0, dftrain['neighborhood_overview'].apply(lambda x: len(str(x))))
dftrain['host_name'] = np.where(dftrain['host_name']=='Null', 0, 1)
dftrain['host_about?'] = np.where(dftrain['host_about']=='Null', 0, 1)
dftrain['host_about'] = np.where(dftrain['host_about']=='Null', 0, dftrain['host_about'].apply(lambda x: len(str(x))))


#%%
#TRANSFORMO LAS CATEGORICAS 

#Ahora reemplazo/agrego las columnas que solo me interesa saber si hay info
#Quiero saber si hay info en la columna name y el largo de el name
dfval['name'] = np.where(dfval['name']=='Null', 0, dfval['name'].apply(lambda x: len(str(x))))
#Lo mismo con otras columnas
dfval['description?'] = np.where(dfval['description']=='Null', 0, 1)
dfval['description'] = np.where(dfval['description']=='Null', 0, dfval['description'].apply(lambda x: len(str(x))))
dfval['neighborhood_overview?'] = np.where(dfval['neighborhood_overview']=='Null', 0, 1)
dfval['neighborhood_overview'] = np.where(dfval['neighborhood_overview']=='Null', 0, dfval['neighborhood_overview'].apply(lambda x: len(str(x))))
dfval['host_name'] = np.where(dfval['host_name']=='Null', 0, 1)
dfval['host_about?'] = np.where(dfval['host_about']=='Null', 0, 1)
dfval['host_about'] = np.where(dfval['host_about']=='Null', 0, dfval['host_about'].apply(lambda x: len(str(x))))



#%%
#SIGO TRANSFORMANDO
#Pongo 1 si esta excempto y 0 si no lo está
dftrain['license'] = np.where(dftrain['license'].str.contains('Exempt', case=False), 1, 0)
print(dftrain['license'].unique())
#Agrego dos variables que cuenten la cantidad de amenities que tiene el alojamiento
dftrain['amenities_count'] = dftrain['amenities'].str.count(',')+1

#%%
#SIGO TRANSFORMANDO
#Pongo 1 si esta excempto y 0 si no lo está
dfval['license'] = np.where(dfval['license'].str.contains('Exempt', case=False), 1, 0)
print(dfval['license'].unique())
#Agrego dos variables que cuenten la cantidad de amenities que tiene el alojamiento
dfval['amenities_count'] = dfval['amenities'].str.count(',')+1


#%%
#TRANSFORMO LAS COLUMNAS QUE QUIZÁS PUEDEN SERVIR EN OTRO FORMATO
#A la columna host_since, first_review y last_review  le dejo solo el dato del año y la convierto en numerica
dftrain['host_since'] = pd.to_datetime(dftrain['host_since'])
dftrain['host_since'] = dftrain['host_since'].dt.strftime('%d%m%Y').astype(int)
dftrain['first_review'] = pd.to_datetime(dftrain['first_review'])
dftrain['first_review'] = dftrain['first_review'].dt.strftime('%d%m%Y').astype(int)
dftrain['last_review'] = pd.to_datetime(dftrain['last_review'])
dftrain['last_review'] = dftrain['last_review'].dt.strftime('%d%m%Y').astype(int)


#%%
#TRANSFORMO LAS COLUMNAS QUE QUIZÁS PUEDEN SERVIR EN OTRO FORMATO
#A la columna host_since, first_review y last_review  le dejo solo el dato del año y la convierto en numerica

dfval['host_since'] = pd.to_datetime(dfval['host_since'])
dfval['host_since'] = dfval['host_since'].dt.strftime('%d%m%Y').astype(int)
dfval['first_review'] = pd.to_datetime(dfval['first_review'])
dfval['first_review'] = dfval['first_review'].dt.strftime('%d%m%Y').astype(int)
dfval['last_review'] = pd.to_datetime(dfval['last_review'])
dfval['last_review'] = dfval['last_review'].dt.strftime('%d%m%Y').astype(int)

#%%
#TRANSFORMO LAS VARIABLES DE PORCENTAJE A NUMÉRICAS E IMPUTO MISSINGS CON LA MEDIA

print(dftrain['host_response_rate'].dtype)
dftrain['host_response_rate'] = dftrain['host_response_rate'].str.rstrip('%')
dftrain['host_response_rate'].replace('Null',95,inplace=True)
dftrain['host_response_rate'] = dftrain['host_response_rate'].astype(int)
dftrain['host_acceptance_rate'] = dftrain['host_acceptance_rate'].str.rstrip('%')
dftrain['host_acceptance_rate'].replace('Null',80,inplace=True)
dftrain['host_acceptance_rate'] = dftrain['host_acceptance_rate'].astype(int)

#%%
#TRANSFORMO LAS VARIABLES DE PORCENTAJE A NUMÉRICAS E IMPUTO MISSINGS CON LA MEDIA

print(dfval['host_response_rate'].dtype)
dfval['host_response_rate'] = dfval['host_response_rate'].str.rstrip('%')
dfval['host_response_rate'].replace('Null',95,inplace=True)
dfval['host_response_rate'] = dfval['host_response_rate'].astype(int)
dfval['host_acceptance_rate'] = dfval['host_acceptance_rate'].str.rstrip('%')
dfval['host_acceptance_rate'].replace('Null',80,inplace=True)
dfval['host_acceptance_rate'] = dfval['host_acceptance_rate'].astype(int)

#%%
#TRANSFORMO LA VARIABLE DE PRECIO A NUMÉRICA

dftrain['price'] = dftrain['price'].str.replace('$', '')
dftrain['price'] = dftrain['price'].str.replace(',', '')
dftrain['price'] = dftrain['price'].astype(float)

#%%
#TRANSFORMO LA VARIABLE DE PRECIO A NUMÉRICA

dfval['price'] = dfval['price'].str.replace('$', '')
dfval['price'] = dfval['price'].str.replace(',', '')
dfval['price'] = dfval['price'].astype(float)


#%%
Dummys_Planeadas = ['host_is_superhost', 'source', 'host_has_profile_pic',
'host_identity_verified','has_availability']

#%%
#AGREGO LAS DUMMYS 
#Creo variable dummy de super host
dummy_SuperHost = pd.get_dummies(dftrain['host_is_superhost'], prefix='dummy_SH').astype(int)
dftrain = pd.concat([dftrain, dummy_SuperHost], axis=1)
dftrain = dftrain.drop('host_is_superhost', axis=1)


#Dummys para source
dummy_source = pd.get_dummies(dftrain['source'], prefix='dummy').astype(int)
dftrain = pd.concat([dftrain, dummy_source], axis=1)
dftrain = dftrain.drop('source', axis=1)


#Dummys para host_identity_verified
dummy_host_identity_verified = pd.get_dummies(dftrain['host_identity_verified'], prefix='dummyverfied').astype(int)
dftrain = pd.concat([dftrain, dummy_host_identity_verified], axis=1)
dftrain = dftrain.drop('host_identity_verified', axis=1)


#Dummys para host_has_profile_pic
dummy_host_has_profile_pic = pd.get_dummies(dftrain['host_has_profile_pic'], prefix='dummyprofilepic').astype(int)
dftrain = pd.concat([dftrain, dummy_host_has_profile_pic], axis=1)
dftrain = dftrain.drop('host_has_profile_pic', axis=1)

#Dummys para has_availability
dummy_has_availability= pd.get_dummies(dftrain['has_availability'], prefix='dummyavailability').astype(int)
dftrain = pd.concat([dftrain, dummy_has_availability], axis=1)
dftrain = dftrain.drop('has_availability', axis=1)

#Dummys para instant_bookable
dummy_instant_bookable= pd.get_dummies(dftrain['instant_bookable'], prefix='dummybookable').astype(int)
dftrain = pd.concat([dftrain, dummy_instant_bookable], axis=1)
dftrain = dftrain.drop('instant_bookable', axis=1)


#%%
#AGREGO LAS DUMMYS 
#Creo variable dummy de super host
dummy_SuperHost = pd.get_dummies(dfval['host_is_superhost'], prefix='dummy_SH').astype(int)
dfval = pd.concat([dfval, dummy_SuperHost], axis=1)
dfval = dfval.drop('host_is_superhost', axis=1)


#Dummys para source
dummy_source = pd.get_dummies(dfval['source'], prefix='dummy').astype(int)
dfval = pd.concat([dfval, dummy_source], axis=1)
dfval = dfval.drop('source', axis=1)


#Dummys para host_has_profile_pic
dummy_host_has_profile_pic = pd.get_dummies(dfval['host_has_profile_pic'], prefix='dummyprofilepic').astype(int)
dfval = pd.concat([dfval, dummy_host_has_profile_pic], axis=1)
dfval = dfval.drop('host_has_profile_pic', axis=1)

#Dummys para host_identity_verified
dummy_host_identity_verified = pd.get_dummies(dfval['host_identity_verified'], prefix='dummyverfied').astype(int)
dfval = pd.concat([dfval, dummy_host_identity_verified], axis=1)
dfval = dfval.drop('host_identity_verified', axis=1)


#Dummys para has_availability
dummy_has_availability= pd.get_dummies(dfval['has_availability'], prefix='dummyavailability').astype(int)
dfval = pd.concat([dfval, dummy_has_availability], axis=1)
dfval = dfval.drop('has_availability', axis=1)

#Dummys para instant_bookable
dummy_instant_bookable= pd.get_dummies(dfval['instant_bookable'], prefix='dummybookable').astype(int)
dfval = pd.concat([dfval, dummy_instant_bookable], axis=1)
dfval = dfval.drop('instant_bookable', axis=1)


#%%
from sklearn.preprocessing import LabelEncoder

categorical_columns = dftrain.select_dtypes(include=['object']).columns

# Itera sobre las columnas categóricas
for column in categorical_columns:
    # Calcula la frecuencia de cada categoría
    frequencies = dftrain[column].value_counts(normalize=False)
    # Aplica la codificación de frecuencia a cada valor en la columna
    dftrain[column] = dftrain[column].map(frequencies).astype(int)


#%%
categorical_columns = dftrain.select_dtypes(include=['object']).columns

for column in categorical_columns:
    train_categories = set(dftrain[column].unique())
    val_categories = set(dfval[column].unique())
    if train_categories != val_categories:
        print(f"Las categorías en {column} no coinciden entre dftrain y dfval.")


for column in categorical_columns:
    train_categories = set(dftrain[column].str.strip().str.lower().unique())
    val_categories = set(dfval[column].str.strip().str.lower().unique())
    if train_categories != val_categories:
        print(f"Las categorías en {column} no coinciden entre dftrain y dfval después de aplicar formato.")

#%%
from sklearn.preprocessing import LabelEncoder
categorical_columns = dfval.select_dtypes(include=['object']).columns

# Itera sobre las columnas categóricas
for column in categorical_columns:
    # Calcula la frecuencia de cada categoría
    frequencies = dfval[column].value_counts(normalize=False)
    # Aplica la codificación de frecuencia a cada valor en la columna
    dfval[column] = dfval[column].map(frequencies).astype(int)

#%%
train_columns = print(dftrain.columns)
test_columns = print(dfval.columns)

#%%
# Obtener las columnas diferentes entre dos conjuntos de datos
columnas_diferentes = set(dftrain.columns).difference(set(dfval.columns))

# Imprimir las columnas diferentes
print("Columnas diferentes:")
for columna in columnas_diferentes:
    print(columna)
    
dftrain = dftrain.drop('dummyavailability_f', axis=1)
dftrain=dftrain.drop('dummy_SH_Null', axis=1)


columnas_diferentes = set(dftrain.columns).difference(set(dfval.columns))

    
#%%
#Columnas numericas que quiero para train
dftrain_num = dftrain.select_dtypes(exclude=['object'])
dftrain_num = dftrain_num.drop('review_scores_rating', axis=1)
dftrain_num = dftrain_num.drop('id', axis=1)


#Columnas numericas que quiero para val
dfval_num = dfval.select_dtypes(exclude=['object'])
dfval_num = dfval_num.drop('id', axis=1)

X_train = dftrain_num
y_train = dftrain['review_scores_rating']
X_test = dfval_num
print("Columnas en X_train:", X_train.columns)
print("Columnas en X_test:", X_test.columns)


#%%
X_test = X_test[X_train.columns]

# %%
#LIBRERIAS PARA ÁRBOLES DE DECISIÓN 

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb


#%%
# Obtener los nombres de columnas de cada DataFrame
columns_train = list(X_train.columns)
columns_test = list(X_test.columns)

# Verificar si los nombres de columnas son iguales
if set(columns_train) == set(columns_test):
    print("Los DataFrames tienen los mismos nombres de columnas.")
else:
    print("Los DataFrames no tienen los mismos nombres de columnas.")



#%%

base_model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=85, max_depth=5)
base_model.fit(X_train, y_train)
y_pred_xgb2 = base_model.predict(X_test)

#%%

dfval = pd.read_csv(r'C:\Users\sofia\Documents\Tp2 Análisis Predictivo\base_val.csv')
predictions_with_id = pd.DataFrame({'id': dfval['id'], 'review_scores_rating': y_pred_xgb2})
# Combinar con DataFrame original de dfval
dfval_with_predictions = dfval.merge(predictions_with_id, on='id')


# %%
#Guardo el csv con las respuestas
predictions_with_id.to_csv(r'C:/Users/sofia/Documents/Tp2 Análisis Predictivo/XGboostotroencoding0.738475.csv', index=False)

