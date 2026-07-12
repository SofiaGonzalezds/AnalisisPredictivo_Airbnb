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
#TRANSFORMO LA VARIABLE DE PRECIO A NUMÉRICA

dftrain['price'] = dftrain['price'].str.replace('$', '')
dftrain['price'] = dftrain['price'].str.replace(',', '')
dftrain['price'] = dftrain['price'].astype(float)

#%%
#Investigo un poco la base

#ANALIZO TIPO DE DATOS, PRIMERAS FILAS Y ESTADISTICAS
from IPython.display import display

display(dftrain.info())#Analizo que tipo de datos tiene cada columna
Head= display(dftrain.head()) #Veo las primeras filas de la base
Estadisticas=display(dftrain.describe()) #Veo algunas estadisticas de las variables numericas
dftrain.shape #La base tiene 68 columnas y 4928 filas

#%%

# CANTIDAD DE VARIABLES CATEGÓRICAS Y NUMÉRICAS QUE HAY
data_types = dftrain.dtypes
cant_num = sum((data_types == 'int64') | (data_types == 'float64'))
cant_cat = sum(data_types == 'object')
print("Variables numéricas:", cant_num)
print("Variables categóricas:", cant_cat)

#%%
#ANALISIS DE MISSINGS
#Me fijo el porcentaje de missings que hay por variable

porcentaje_faltantes = (dftrain.isna().mean() * 100).round(2)
print(porcentaje_faltantes)

#%%
# ANALIZO VARIABLES NUMÉRICAS

#Para ello primero paso las variables que son porcentajes a variables numéricas e imputo los faltantes
#con la media

dftrain['host_response_rate'] = dftrain['host_response_rate'].astype(str).str.rstrip('%')
dftrain['host_response_rate'].replace('nan',95, inplace=True) # 95 es la media
dftrain['host_response_rate'] = dftrain['host_response_rate'].astype(int)
dftrain['host_acceptance_rate'] = dftrain['host_acceptance_rate'].astype(str).str.rstrip('%')
dftrain['host_acceptance_rate'].replace('nan',80, inplace=True)  #80 es la media
dftrain['host_acceptance_rate'] = dftrain['host_acceptance_rate'].astype(int)

#%%
#Observo mis variables numéricas
dftrain_num = dftrain.select_dtypes(exclude=['object'])
print("Columnas numéricas:")
view_numercial_columns = display(dftrain.select_dtypes(exclude=['object']).columns)

# Calculo la matriz de correlación
correlation_matrix = dftrain_num.corr()
see_correlation=display(dftrain_num.corr())

# Mostrar la matriz de correlación en un mapa de calor
import seaborn as sns
import matplotlib.pyplot as plt
cmap = sns.diverging_palette(240, 10, s=99, l=50, n=256, as_cmap=True)
cmap.set_bad('white')  
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, cmap=cmap, center=0)
heatmap.set_title('Matriz de correlación')
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
plt.show()

#%%
reviewcore=correlation_matrix['review_scores_rating']

#%%
#Analizamos outliers
dftrain_num=dftrain.select_dtypes(exclude=['object'])
Q1 = dftrain_num.quantile(0.25)
Q3 = dftrain_num.quantile(0.75)
IQR = Q3 - Q1

# Identificar los valores atípicos utilizando el rango intercuartílico (IQR)
outliers = ((dftrain_num < (Q1 - 1.5 * IQR)) | (dftrain_num > (Q3 + 1.5 * IQR))).sum()

print(outliers)


#%%
# ANALIZO VARIABLES CATEGÓRICAS

# Observo mis variables categóricas
dftrain_cat = dftrain.select_dtypes(include=['object'])
view_categorical_columns = display(dftrain.select_dtypes(include=['object']).columns)

# Me fijo las frecuencias de algunas de mis variables

#source
source_frecuencias = dftrain['source'].value_counts()
print(source_frecuencias)

#host location
host_location_frecuencias = dftrain['host_location'].value_counts()
print(host_location_frecuencias)

#host_is_superhost
host_is_superhost_frecuencias = dftrain['host_is_superhost'].value_counts()
print(host_is_superhost_frecuencias)

#host_verifications
host_verifications_frecuencias = dftrain['host_verifications'].value_counts()
print(host_verifications_frecuencias)

#host_has_profile_pic
host_has_profile_pic_frecuencias = dftrain['host_has_profile_pic'].value_counts()
print(host_has_profile_pic_frecuencias)

#host_identity_verified
host_identity_verified_frecuencias = dftrain['host_identity_verified'].value_counts()
print(host_identity_verified_frecuencias)

#property_type
property_type_frecuencias = dftrain['property_type'].value_counts()
print(property_type_frecuencias)

#room_type
room_type_frecuencias = dftrain['room_type'].value_counts()
print(room_type_frecuencias)

#bathrooms_text
bathrooms_text_frecuencias = dftrain['bathrooms_text'].value_counts()
print(bathrooms_text_frecuencias)

#has_availability
has_availability_frecuencias = dftrain['has_availability'].value_counts()
print(has_availability_frecuencias)

#instant_bookable
instant_bookable_frecuencias = dftrain['instant_bookable'].value_counts()
print(instant_bookable_frecuencias)


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

view_categorical_columns = display(dftrain.select_dtypes(include=['object']).columns)
view_numercial_columns = display(dftrain.select_dtypes(exclude=['object']).columns)

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

print(dftrain["beds"])


#%%
from textblob import TextBlob

#Creo 4 variables nuevas que muestran el sentimiento de las variables con texto

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
#A la columna host_since, first_review y last_review la convierto en numerica
dftrain['host_since'] = pd.to_datetime(dftrain['host_since'])
dftrain['host_since'] = dftrain['host_since'].dt.strftime('%Y%m%d').astype(int)
dftrain['first_review'] = pd.to_datetime(dftrain['first_review'])
dftrain['first_review'] = dftrain['first_review'].dt.strftime('%Y%m%d').astype(int)
dftrain['last_review'] = pd.to_datetime(dftrain['last_review'])
dftrain['last_review'] = dftrain['last_review'].dt.strftime('%Y%m%d').astype(int)

#%%
#AGREGO LAS DUMMYS para las variables que contienen solo dos tipos de valores

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

#Las variables que contienen categorias adicionales en dfval que no estan en dftrain las imputo con el metodo de frecuencia

categorical_columns =  dftrain.select_dtypes(include=['object']).columns

# Itera sobre las columnas categóricas
for column in categorical_columns:
    # Calcula la frecuencia de cada categoría
    frequencies = dftrain[column].value_counts(normalize=False)
    # Aplica la codificación de frecuencia a cada valor en la columna
    dftrain[column] = dftrain[column].map(frequencies).astype(int)
    


#%%
#Observo mis variables numéricas
dftrain_num = dftrain.select_dtypes(exclude=['object'])

# Seleccionar las variables numéricas y la variable 'review_scores_rating'
dftrain_num_new = dftrain_num.drop(columns=['review_scores_rating'])
# Calcular las correlaciones con respecto a 'review_scores_rating'
correlations = dftrain_num_new.corrwith(dftrain_num['review_scores_rating'])

# Ordenar las correlaciones de forma descendente
correlations = correlations.sort_values(ascending=False)

# Mostrar la matriz de correlación en un mapa de calor
# Dividir las variables en grupos de 10
groups = [correlations[i:i+10] for i in range(0, len(correlations), 10)]

# Generar imágenes separadas para cada grupo
for i, group in enumerate(groups):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(240, 10, s=99, l=50, n=256, as_cmap=True)
    cmap.set_bad('white')
    heatmap = sns.heatmap(group.to_frame(), annot=True, fmt=".2f", cmap=cmap, center=0, cbar=False, ax=ax)
    heatmap.set_title(f'Matriz de correlación - Review Scores Rating (Grupo {i+1})')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    plt.savefig(f'correlation_group_{i+1}.png')  # Guardar la imagen en un archivo
    plt.show()
     
# %%
#Ahora Exploro el Dataset de validación


#TRANSFORMO LA VARIABLE DE PRECIO A NUMÉRICA

dfval['price'] = dfval['price'].str.replace('$', '')
dfval['price'] = dfval['price'].str.replace(',', '')
dfval['price'] = dfval['price'].astype(float)

#%%
#Investigo un poco la base

#ANALIZO TIPO DE DATOS, PRIMERAS FILAS Y ESTADISTICAS
from IPython.display import display

display(dfval.info())#Analizo que tipo de datos tiene cada columna
Head= display(dfval.head()) #Veo las primeras filas de la base
Estadisticas=display(dfval.describe()) #Veo algunas estadisticas de las variables numericas
dfval.shape #La base tiene 68 columnas y 4928 filas

#%%

# CANTIDAD DE VARIABLES CATEGÓRICAS Y NUMÉRICAS QUE HAY
data_types = dfval.dtypes
cant_num = sum((data_types == 'int64') | (data_types == 'float64'))
cant_cat = sum(data_types == 'object')
print("Variables numéricas:", cant_num)
print("Variables categóricas:", cant_cat)

#%%
#ANALISIS DE MISSINGS
porcentaje_faltantes = (dfval.isna().mean() * 100).round(2)
print(porcentaje_faltantes)

#%%
# ANALIZO VARIABLES NUMÉRICAS

dfval['host_response_rate'] = dfval['host_response_rate'].astype(str).str.rstrip('%')
dfval['host_response_rate'].replace('nan',95, inplace=True) # 95 es la media
dfval['host_response_rate'] = dfval['host_response_rate'].astype(int)
dfval['host_acceptance_rate'] = dfval['host_acceptance_rate'].astype(str).str.rstrip('%')
dfval['host_acceptance_rate'].replace('nan',80, inplace=True)  #80 es la media
dfval['host_acceptance_rate'] = dfval['host_acceptance_rate'].astype(int)

#%%
#Observo mis variables numéricas
dfval_num = dfval.select_dtypes(exclude=['object'])
print("Columnas numéricas:")
view_numercial_columns = display(dfval.select_dtypes(exclude=['object']).columns)

# Calculo la matriz de correlación
correlation_matrix = dfval_num.corr()
see_correlation=display(dfval_num.corr())

# Mostrar la matriz de correlación en un mapa de calor
import seaborn as sns
import matplotlib.pyplot as plt
cmap = sns.diverging_palette(240, 10, s=99, l=50, n=256, as_cmap=True)
cmap.set_bad('white')  
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, cmap=cmap, center=0)
heatmap.set_title('Matriz de correlación')
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
plt.show()

#%%
#Analizamos outliers
# Calcular los límites de los valores atípicos utilizando el método de los cuartiles
Q1 = dfval_num.quantile(0.25)
Q3 = dfval_num.quantile(0.75)
IQR = Q3 - Q1

# Identificar los valores atípicos utilizando el rango intercuartílico (IQR)
outliers = ((dfval_num < (Q1 - 1.5 * IQR)) | (dfval_num > (Q3 + 1.5 * IQR))).sum()

print(outliers)


#%%
# ANALIZO VARIABLES CATEGÓRICAS

# Observo mis variables categóricas
dfval_cat = dfval.select_dtypes(include=['object'])
view_categorical_columns = display(dfval.select_dtypes(include=['object']).columns)

# Me fijo las frecuencias de algunas de mis variables

#source
source_frecuencias = dfval['source'].value_counts()
print(source_frecuencias)

#host location
host_location_frecuencias = dfval['host_location'].value_counts()
print(host_location_frecuencias)

#host_is_superhost
host_is_superhost_frecuencias = dfval['host_is_superhost'].value_counts()
print(host_is_superhost_frecuencias)

#host_verifications
host_verifications_frecuencias = dfval['host_verifications'].value_counts()
print(host_verifications_frecuencias)

#host_has_profile_pic
host_has_profile_pic_frecuencias = dfval['host_has_profile_pic'].value_counts()
print(host_has_profile_pic_frecuencias)

#host_identity_verified
host_identity_verified_frecuencias = dfval['host_identity_verified'].value_counts()
print(host_identity_verified_frecuencias)

#property_type
property_type_frecuencias = dfval['property_type'].value_counts()
print(property_type_frecuencias)

#room_type
room_type_frecuencias = dfval['room_type'].value_counts()
print(room_type_frecuencias)

#bathrooms_text
bathrooms_text_frecuencias = dfval['bathrooms_text'].value_counts()
print(bathrooms_text_frecuencias)

#has_availability
has_availability_frecuencias = dfval['has_availability'].value_counts()
print(has_availability_frecuencias)

#instant_bookable
instant_bookable_frecuencias = dfval['instant_bookable'].value_counts()
print(instant_bookable_frecuencias)

#%%
#PREPARACIÓN DE LA BASE VALIDACION

#BORRO COLUMNAS VACIAS O INNECESARIAS
# Verifico si existen columnas que tienen todas las filas con valores faltantes
Columnas_vacias = dfval.columns[dfval.isna().all()].tolist()
#Como las columnas ['neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated'] tienen todos datos
#faltantes, las quito del modelo
dfval = dfval.drop(Columnas_vacias, axis=1)

#Borro las columnas que estoy segura que no sirven para la predicción
columnas_eliminar = ['id','calendar_last_scraped']
dfval = dfval.drop(columnas_eliminar, axis=1)
dfval.columns


#%%
dfval_cat = dfval.select_dtypes(include=['object']).columns
dfval_num = dfval.select_dtypes(exclude=['object']).columns

view_categorical_columns = display(dfval.select_dtypes(include=['object']).columns)
view_numercial_columns = display(dfval.select_dtypes(exclude=['object']).columns)

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

print(dfval["beds"])


#%%
from textblob import TextBlob

dfval['sentiment_score_description'] = dfval['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
dfval['sentiment_score_neighborhood_overview'] = dfval['neighborhood_overview'].apply(lambda x: TextBlob(x).sentiment.polarity)
dfval['sentiment_score_host_about'] = dfval['host_about'].apply(lambda x: TextBlob(x).sentiment.polarity)
dfval['sentiment_score_name'] = dfval['name'].apply(lambda x: TextBlob(x).sentiment.polarity)


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
dfval['host_name'] = np.where(dfval['host_name']=='Null', 0, dfval['host_name'].apply(lambda x: len(str(x))))
dfval['host_about?'] = np.where(dfval['host_about']=='Null', 0, 1)
dfval['host_about'] = np.where(dfval['host_about']=='Null', 0, dfval['host_about'].apply(lambda x: len(str(x))))

#%%
#SIGO TRANSFORMANDO
#Pongo 1 si esta excempto y 0 si no lo está
dfval['license'] = np.where(dfval['license'].str.contains('Exempt', case=False), 1, 0)
#Agrego dos variables que cuenten la cantidad de amenities que tiene el alojamiento
dfval['amenities_count'] = dfval['amenities'].str.count(',')+1

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
#AGREGO LAS DUMMYS 
#Creo variable dummy de super host
dummy_SuperHost = pd.get_dummies(dfval['host_is_superhost'], prefix='dummy_SH').astype(int)
dfval = pd.concat([dfval, dummy_SuperHost], axis=1)
dfval=dfval.drop('host_is_superhost', axis=1)

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

categorical_columns = dfval.select_dtypes(include=['object']).columns

# Itera sobre las columnas categóricas
for column in categorical_columns:
    # Calcula la frecuencia de cada categoría
    frequencies = dfval[column].value_counts(normalize=False)
    # Aplica la codificación de frecuencia a cada valor en la columna
    dfval[column] = dfval[column].map(frequencies).astype(int)
    



#%%

#Me fijo si tienen columnas diferentes

# Obtener las columnas diferentes entre dos conjuntos de datos
columnas_diferentes = set(dftrain.columns).difference(set(dfval.columns))

# Imprimir las columnas diferentes
print("Columnas diferentes:")
for columna in columnas_diferentes:
    print(columna)
    
#%%

#Diferencias entre Datasets

dftrain = pd.read_csv(r'C:\Users\sofia\Documents\Tp2 Análisis Predictivo\base_train.csv')
dfval = pd.read_csv(r'C:\Users\sofia\Documents\Tp2 Análisis Predictivo\base_val.csv')


#%%
#Veo las nuevas categorias
#Quiero ver si las variables poseen las mismas categorias, o si en la base de test aparecen categorias nuevas

columnas_comunes = ['host_location','host_response_time', 'host_verifications', 'property_type', 'room_type', 'bathrooms_text','host_neighbourhood','host_verifications','neighbourhood','neighbourhood_cleansed','amenities']
for columna in columnas_comunes:
        categorias_dftrain = dftrain[columna].unique()
        categorias_dfval = dfval[columna].unique()
        
        categorias_nuevas = set(categorias_dfval) - set(categorias_dftrain)
        cantidad_categorias_nuevas = len(categorias_nuevas)
        # print(f"Tipos nuevos en columna {columna}:")
        # for categoria in categorias_nuevas:
        #      print(categoria)
        print(f"Cantidad de categorías nuevas en columna {columna}: {cantidad_categorias_nuevas}")

