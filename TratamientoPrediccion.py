#%%
import pandas as pd
import numpy as np
import json
import os
import zipfile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# os.system('kaggle competitions download -c unimelb')

# # Extraigo los archivos del zip
# for file in os.listdir():
#     if file.endswith('.zip'):
#         zip_ref = zipfile.ZipFile(file, 'r')
#         zip_ref.extractall()
#         zip_ref.close()



# %%
#Guardo el Dataframe Train
dftrain = pd.read_csv(r'C:\Users\sofia\Documents\Tp2 Análisis Predictivo\base_train.csv')
dfval = pd.read_csv(r'C:\Users\sofia\Documents\Tp2 Análisis Predictivo\base_val.csv')

#%%
#PREPARACIÓN DE LA BASE

#BORRO COLUMNAS VACIAS O INNECESARIAS
# Verifico si existen columnas que tienen todas las filas con valores faltantes
Columnas_vacias = dftrain.columns[dftrain.isna().all()].tolist()
#Como las columnas ['neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated'] tienen todos datos
#faltantes, las quito del modelo
dftrain = dftrain.drop(Columnas_vacias, axis=1)

#Borro las columnas que estoy segura que no sirven para la predicción
columnas_eliminar = ['id','calendar_last_scraped']
dftrain = dftrain.drop(columnas_eliminar, axis=1)
dftrain.columns




#%%
dftrain_cat = dftrain.select_dtypes(include=['object']).columns
dftrain_num = dftrain.select_dtypes(exclude=['object']).columns

#Imputo Missings

from feature_engine.imputation import CategoricalImputer
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder

# Imputación de variables categóricas colocando la palabra "Null" si sos nas
# Para las vaeiables numéricas las imputo con la media

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
from textblob import TextBlob

#Agrego variables nuevas con la informacion del sentiment score de las columnas que poseen texto que puede ser analizado

dftrain['sentiment_score_description'] = dftrain['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
dftrain['sentiment_score_neighborhood_overview'] = dftrain['neighborhood_overview'].apply(lambda x: TextBlob(x).sentiment.polarity)
dftrain['sentiment_score_host_about'] = dftrain['host_about'].apply(lambda x: TextBlob(x).sentiment.polarity)
dftrain['sentiment_score_name'] = dftrain['name'].apply(lambda x: TextBlob(x).sentiment.polarity)


#%%
#TRANSFORMO LAS CATEGORICAS 

#Ahora reemplazo/agrego las columnas que solo me interesa saber si tienen información o son nulas

#Solo me interesa saber si el registro contiene el dato de name o no
dftrain['name'] = np.where(dftrain['name']=='Null', 0, dftrain['name'].apply(lambda x: len(str(x))))
#Para description creo una columna extra que me de información sobre el número de caracteres que posee
dftrain['description?'] = np.where(dftrain['description']=='Null', 0, 1)
dftrain['description'] = np.where(dftrain['description']=='Null', 0, dftrain['description'].apply(lambda x: len(str(x))))
#Para neighborhood_overview y para host_about hago el mismo procesos que se utilizó para description. Para el host_name 
#solo cuento la cantidad de caracteres ya que la variable no contiene nulos.
dftrain['neighborhood_overview?'] = np.where(dftrain['neighborhood_overview']=='Null', 0, 1)
dftrain['neighborhood_overview'] = np.where(dftrain['neighborhood_overview']=='Null', 0, dftrain['neighborhood_overview'].apply(lambda x: len(str(x))))
dftrain['host_about?'] = np.where(dftrain['host_about']=='Null', 0, 1)
dftrain['host_about'] = np.where(dftrain['host_about']=='Null', 0, dftrain['host_about'].apply(lambda x: len(str(x))))
dftrain['host_name'] = np.where(dftrain['host_name']=='Null', 0, dftrain['host_name'].apply(lambda x: len(str(x))))

#%%
#SIGO TRANSFORMANDO
#Pongo 1 si esta excempto y 0 si no lo está
dftrain['license'] = np.where(dftrain['license'].str.contains('Exempt', case=False), 1, 0)

#Agrego dos variables que cuenten la cantidad de amenities que tiene el alojamiento
dftrain['amenities_count'] = dftrain['amenities'].str.count(',')+1


#%%
#TRANSFORMO LAS COLUMNAS QUE QUIZÁS PUEDEN SERVIR EN OTRO FORMATO
#A la columna host_since, first_review y last_review  le dejo solo el dato del año y la convierto en numerica
dftrain['host_since'] = pd.to_datetime(dftrain['host_since'])
dftrain['host_since'] = dftrain['host_since'].dt.strftime('%Y%m%d').astype(int)
dftrain['first_review'] = pd.to_datetime(dftrain['first_review'])
dftrain['first_review'] = dftrain['first_review'].dt.strftime('%Y%m%d').astype(int)
dftrain['last_review'] = pd.to_datetime(dftrain['last_review'])
dftrain['last_review'] = dftrain['last_review'].dt.strftime('%Y%m%d').astype(int)

#%%
#TRANSFORMO LAS VARIABLES DE PORCENTAJE A NUMÉRICAS E IMPUTO MISSINGS CON LA MEDIA

#Para ello primero paso las variables que son porcentajes a variables numéricas e imputo los faltantes
#con la media

print(dftrain['host_response_rate'].dtype)
dftrain['host_response_rate'] = dftrain['host_response_rate'].str.rstrip('%')
dftrain['host_response_rate'].replace('Null',95, inplace=True)
dftrain['host_response_rate'] = dftrain['host_response_rate'].astype(int)
dftrain['host_acceptance_rate'] = dftrain['host_acceptance_rate'].str.rstrip('%')
dftrain['host_acceptance_rate'].replace('Null',80, inplace=True)
dftrain['host_acceptance_rate'] = dftrain['host_acceptance_rate'].astype(int)

#%%
#TRANSFORMO LA VARIABLE DE PRECIO A NUMÉRICA

dftrain['price'] = dftrain['price'].str.replace('$', '')
dftrain['price'] = dftrain['price'].str.replace(',', '')
dftrain['price'] = dftrain['price'].astype(float)


#%%
#AGREGO LAS DUMMYS 
#Creo dummys con las variables que contienen dos posibles categorías.

#Creo variable dummy de super host
dummy_SuperHost = pd.get_dummies(dftrain['host_is_superhost'], prefix='dummy_SH').astype(int)
dftrain = pd.concat([dftrain, dummy_SuperHost], axis=1)
dftrain=dftrain.drop('host_is_superhost', axis=1)

#Dummys para source
dummy_source = pd.get_dummies(dftrain['source'], prefix='dummy').astype(int)
dftrain = pd.concat([dftrain, dummy_source], axis=1)
dftrain = dftrain.drop('source', axis=1)

#Dummys para host_has_profile_pic
dummy_host_has_profile_pic = pd.get_dummies(dftrain['host_has_profile_pic'], prefix='dummyprofilepic').astype(int)
dftrain = pd.concat([dftrain, dummy_host_has_profile_pic], axis=1)
dftrain = dftrain.drop('host_has_profile_pic', axis=1)


#Dummys para host_identity_verified
dummy_host_identity_verified = pd.get_dummies(dftrain['host_identity_verified'], prefix='dummyverfied').astype(int)
dftrain = pd.concat([dftrain, dummy_host_identity_verified], axis=1)
dftrain = dftrain.drop('host_identity_verified', axis=1)


#Dummys para has_availability
dummy_has_availability= pd.get_dummies(dftrain['has_availability'], prefix='dummyavailability').astype(int)
dftrain = pd.concat([dftrain, dummy_has_availability], axis=1)
dftrain = dftrain.drop('has_availability', axis=1)


#Dummys para instant_bookable
dummy_instant_bookable= pd.get_dummies(dftrain['instant_bookable'], prefix='dummybookable').astype(int)
dftrain = pd.concat([dftrain, dummy_instant_bookable], axis=1)
dftrain = dftrain.drop('instant_bookable', axis=1)

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
#VEO COMO QUEDÓ LA NUEVA BASE
print(dftrain)

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
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import lightgbm as lgb
from catboost import CatBoostRegressor

# %%
# PREPARACIÓN PARA ÁRBOLES DE DECISIÓN

dfad=dftrain.drop('review_scores_rating',axis=1)
dfad= dfad.select_dtypes(exclude=['object'])

#Defino X e y
X = dfad
y = dftrain['review_scores_rating']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)



# %%
#Prueba Modelo 1

#ÁRBOL DE DECICIÓN DecisionTreeRegressor

print("Pruebo el árbol de decisión de DecisionTreeRegressor")

#Elijo los hiperparámetros y defino la regresión
regr1 = DecisionTreeRegressor(max_depth=6, min_samples_split=6)
#Entreno el modelo
regr1.fit(X_train, y_train)

#Hago la predicción
y_pred1 = regr1.predict(X_test)

# Calculo las métricas de evaluación
mse_1 = mean_squared_error(y_test, y_pred1)
mae_1 = mean_absolute_error(y_test, y_pred1)
r2_1 = r2_score(y_test, y_pred1)


# Imprimo las métricas de evaluación
print("MSE del Decision Tree Regressor:", mse_1)
print("MAE del Decision Tree Regressor:", mae_1)
print("R2 Score del Decision Tree Regressor:", r2_1)

# Realizo validación cruzada con 10-fold
scores = cross_val_score(regr1,  X, y, cv=10, scoring='r2')
# Convierto las puntuaciones negativas a positivas y calculo la media
mse_1_cv = scores.mean()
print("R2 Score dela validación cruzada es:", mse_1_cv)



# %%

#Prueba Modelo 2
#ÁRBOL DE DECISIÓN AdaBoostRegressor

print("Pruebo el árbol de decisión de AdaBoostRegressor")

#Elijo los hiperparámetros y defino la regresión
regr2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6, min_samples_split=6))

#Entreno el modelo
regr2.fit(X_train, y_train)

#Hago la predicción
y_pred2 = regr2.predict(X_test)

#Calculo las métricas de evaluación
mse_2 = mean_squared_error(y_test, y_pred2)
mae_2 = mean_absolute_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)

# Imprimo las métricas de evaluación
print("MSE del AdaBoost Regressor:", mse_2)
print("MAE del AdaBoost Regressor:", mae_2)
print("R2 Score del AdaBoost Regressor:", r2_2)

# Realizo validación cruzada con 10-fold
scores = cross_val_score(regr2,  X, y, cv=10, scoring='r2')
# Convierto las puntuaciones negativas a positivas y calculo la media
mse_2_cv = scores.mean()
print("R2 Score dela validación cruzada es:", mse_2_cv)

# %%
#Prueba Modelo 3

#ÁRBOL DE DECISIÓN Random Forest Regressor

print("Pruebo el árbol de decisión de Random Forest Regressor")

# Elijo los hiperparámetros y defino la regresión
regr3 =  RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=10)
regr3.fit(X_train, y_train)
# Hago la predicción
y_pred3 = regr3.predict(X_test)

# Calculo las métricas de evaluación
mse_3 = mean_squared_error(y_test, y_pred3)
mae_3 = mean_absolute_error(y_test, y_pred3)
r2_3 = r2_score(y_test, y_pred3)

# Imprimo las métricas de evaluación
print("MSE del Random Forest Regressor:", mse_3)
print("MAE del Random Forest Regressor:", mae_3)
print("R2 Score del Random Forest Regressor:", r2_3)


r2_scores = cross_val_score(regr3, X, y, scoring='r2', cv=10).mean()

# Imprimir las métricas de evaluación

print("R2 Score de XGBoost Regressor:", r2_scores)

#%%
#Prueba Modelo 14

#ÁRBOL DE DECISIÓN de Extra Trees Regressor

print("Pruebo el árbol de decisión de Extra Trees Regressor")

# Elijo los hiperparámetros y defino la regresión
regr4 = ExtraTreesRegressor(n_estimators=100, min_samples_split=2)
# Entreno el modelo
regr4.fit(X_train, y_train)

# Hago la predicción
y_pred4 = regr4.predict(X_test)

# Calculo las métricas de evaluación
mse_4 = mean_squared_error(y_test, y_pred4)
mae_4 = mean_absolute_error(y_test, y_pred4)
r2_4 = r2_score(y_test, y_pred4)

# Imprimo las métricas de evaluación
print("MSE del Extra Trees Regressor:", mse_4)
print("MAE del Extra Trees Regressor:", mae_4)
print("R2 Score del Extra Trees Regressor:", r2_4)

# Realizo validación cruzada con 10-fold
scores = cross_val_score(regr4, X, y, cv=10, scoring='r2')
# Convierto las puntuaciones negativas a positivas y calculo la media
mse_4_cv = scores.mean()
print("R2 Score dela validación cruzada es:", mse_4_cv)

# %%
#Prueba Modelo 5
#ÁRBOL DE DECISIÓN de Gradient Boosting Regressor

print("Pruebo el árbol de decisión de Gradient Boosting Regressor")

# Elijo los hiperparámetros y defino la regresión
regr5 = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1, max_depth=11)
# Entreno el modelo
regr5.fit(X_train, y_train)

# Hago la predicción
y_pred5 = regr5.predict(X_test)

# Calculo las métricas de evaluación
mse_5 = mean_squared_error(y_test, y_pred5)
mae_5 = mean_absolute_error(y_test, y_pred5)
r2_5 = r2_score(y_test, y_pred5)

# Imprimo las métricas de evaluación
print("MSE del Gradient Boosting Regressor:", mse_5)
print("MAE del Gradient Boosting Regressor:", mae_5)
print("R2 Score del Gradient Boosting Regressor:", r2_5)

# Realizo validación cruzada con 10-fold
scores = cross_val_score(regr5, X, y, cv=10, scoring='r2')
mse_5_cv = scores.mean()
print("R2 Score dela validación cruzada es:", mse_5_cv)

#%%image.png

#Me fijo cuales son los mejores hiperparametros para XGboost Regressor

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'learning_rate': [0.1],
    'n_estimators': [70, 80, 85, 90, 100],
    'subsample': [0.8,0.99,0.9, 1.0],
    'max_depth': [3, 4,5, 6,7,8],
   }

# Crear el modelo base
base_model = xgb.XGBRegressor()

# Realizar la búsqueda en cuadrícula
grid_search = GridSearchCV(base_model, param_grid, scoring='neg_mean_squared_error', cv=7)
grid_search.fit(X, y)

# Obtener los mejores hiperparámetros y el modelo final
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params)
print(best_model)



#%%
#Prueba Modelo 6
#ÁRBOL DE DECISIÓN de XGBoost 

print("Pruebo XGBoost para regresión")


# Crear el modelo de XGBoost para regresión con los hiperparámetros
# Definir los hiperparámetros del modelo

    
base_model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=85, max_depth=5)
base_model.fit(X_train, y_train)
y_pred_xgb2 = base_model.predict(X_test)


# Calcular las métricas de evaluación
mse_xgb2 = mean_squared_error(y_test, y_pred_xgb2)
mae_xgb2 = mean_absolute_error(y_test, y_pred_xgb2)
r2_xgb2 = r2_score(y_test, y_pred_xgb2)

# Imprimir las métricas de evaluación
print("MSE de XGBoost Regressor:", mse_xgb2)
print("MAE de XGBoost Regressor:", mae_xgb2)
print("R2 Score de XGBoost Regressor:", r2_xgb2)

r2_scores = cross_val_score(base_model, X, y, scoring='r2', cv=10).mean()

# Imprimir las métricas de evaluación

print("R2 Score de XGBoost Regressor:", r2_scores)



#%%

#Me fijo cuales son los mejores hiperparametros para Lgb Regressor

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'num_leaves': [5,10,20,50,100],
    'learning_rate': [0.1,0.001,0.0001],
    'n_estimators': [50,80,90,100,150,200],
    'max_depth': [3,5,7,10]
}

# Crear el modelo base
base_model = lgb.LGBMRegressor()

# Realizar la búsqueda en cuadrícula
grid_search = GridSearchCV(base_model, param_grid, scoring='neg_mean_squared_error', cv=10)
grid_search.fit(X, y)

# Obtener los mejores hiperparámetros y el modelo final
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params)
print(best_model)

#%%
#Prueba Modelo 7
#ÁRBOL DE DECISIÓN de LightGBM 

print("Pruebo LightGBM para regresión")

# Configurar los parámetros de LightGBM
params = {
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.1,
    'n_estimators': 300,
    'max_depth': 8
}

# Crear el modelo base de LightGBM
base_model = lgb.LGBMRegressor(**params)
base_model.fit(X_train, y_train)
# Realizar predicciones en los datos de prueba
y_pred = base_model.predict(X_test)

# Calcular las métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir las métricas de evaluación
print("MSE de LightGBM:", mse)
print("MAE de  LightGBM:", mae)
print("R2 Score de LightGBM:", r2)

# Realizar validación cruzada
r2_scores = cross_val_score(base_model, X, y, scoring='r2', cv=5).mean()

# Imprimir el puntaje R2 promedio de la validación cruzada
print("R2 Score de LightGBM (validación cruzada):", r2_scores)

#%%
#Prueba Modelo 9
#ÁRBOL DE DECISIÓN de CatBoost

print("Pruebo CatBoost para regresión")

model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=13)

# Ajustar el modelo BaggingRegressor
model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Calcular las métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir las métricas de evaluación
print("MSE de CatBoostRegressor:", mse)
print("MAE de CatBoostRegressor:", mae)
print("R2 Score de CatBoostRegressor:", r2)

# Realizar validación cruzada
r2_scores = cross_val_score(model, X, y, scoring='r2', cv=5).mean()
# Imprimir el puntaje R2 promedio de la validación cruzada
print("R2 Score de CatBoostRegressor (validación cruzada):", r2_scores)

