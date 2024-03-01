# %% [markdown]
# Previsão de crescimento da Indústria de Transformação

# %% [markdown]
# Preparação dos Dados

# %%
#Importanto Bibliotecas
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import glob
import re
from functools import reduce
from datetime import datetime
import locale
import pingouin as pg
pd.set_option('display.max_columns', None)

# %%
locale.setlocale(locale.LC_ALL, 'pt_pt.UTF-8')

# %%
# Listando as bases
lista_de_bases = glob.glob("bases/*.csv")
lista_de_bases

# %%
#Carregando bases
bases = []
for base_name in lista_de_bases:
    df_base = pd.read_csv(base_name, sep = ';', encoding='ANSI', decimal=',')
    bases.append(df_base)

# %%
#Juantando bases horizontalmente
df = reduce(lambda a, b : pd.merge(left = a, right = b, how = 'inner', on = ['Data']), bases)
df = df.loc[df['Data'] != 'Fonte']
df

# %%
df['Data'] = df['Data'].apply(lambda x : datetime.strptime(x, '%b/%y'))

# %% [markdown]
# Análise Exploratória de Dados 

# %%
df.describe()

# %%
df.isnull().any()

# %%
for col in df.columns[1:]:
    df[col] = df[col].apply(lambda x : x.strip().replace(',', '.')).replace('-',np.NaN).astype(float)

# %%
#Média Móvel Simples
df['6-month-SMA'] = df['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'].rolling(window=6).mean()
df['12-month-SMA'] = df['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'].rolling(window=12).mean()
df

# %%
df_ind_transf = df.set_index('Data')[['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice']].copy()
df_ind_transf.rename(columns={'21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice':'Indústria de transformação'}, inplace=True)
df_ind_transf.plot(figsize=(18,10))

# %%
df_ind_transf.head(2)

# %%
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

# %%
industria_outliers = df_ind_transf['Indústria de transformação']  
outliers_iqr = detect_outliers_iqr(industria_outliers)
print("Outliers detectados pelo método IQR:")
print(industria_outliers[outliers_iqr])

# %%
dfSMA = df.set_index('Data')[['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice', '6-month-SMA', '12-month-SMA']].copy()
dfSMA.rename(columns={'21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice':'Indústria de transformação'}, inplace=True)
dfSMA

# %%
dfSMA.plot(figsize=(18,10))

# %%
#Média Móvel Exponencialmente Ponderada
df['6-month-EWMA'] = df['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'].ewm(span=6,adjust=False).mean()
df['12-month-EWMA'] = df['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'].ewm(span=12,adjust=False).mean()
df

# %%
dfEWMA = df.set_index('Data')[['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice', '6-month-SMA', '12-month-SMA']].copy()
dfEWMA.rename(columns={'21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice':'Indústria de transformação'}, inplace=True)
dfEWMA

# %%
dfEWMA.plot(figsize=(18,10))

# %%
#Outliers
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers


# %%
from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(df_ind_transf,model='add').plot()



# %%
seasonal_decompose(df_ind_transf,model='mul').plot()

# %% [markdown]
# Modelo 1: ETS - Suavização Exponencial Tripla

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# %%
len(df_ind_transf)

# %%
dados_treinados = df_ind_transf.iloc[:210] # Primeiras 80% das linhas (262 * 0.8 = 209,6)
dados_testados= df_ind_transf.iloc[210:] # Últimos 20% das linhas (262 - 210 = 52)


# %%
# Criando o modelo de Suavização Exponencial Tripla
fitted_model = ExponentialSmoothing(
    dados_treinados['Indústria de transformação'],
    trend='mul',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Fazendo previsões
etsprevi = fitted_model.forecast(52).rename('ETS Previsão')

# %%
etsprevi

# %%
# Ajustando o índice para corresponder aos dados testados
etsprevi.index = dados_testados.index

# %%
# Calcular as métricas de avaliação
mae = mean_absolute_error(dados_testados['Indústria de transformação'], etsprevi)
mse = mean_squared_error(dados_testados['Indústria de transformação'], etsprevi)
r2 = r2_score(dados_testados['Indústria de transformação'], etsprevi)
# Imprimir as métricas
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# %%
# Plotando os dados
dados_treinados['Indústria de transformação'].plot(legend=True, label='Dados Treinados')
dados_testados['Indústria de transformação'].plot(legend=True, label='Dados Testados', figsize=(15, 10))
etsprevi.plot(legend=True, label='Previsão')

# %% [markdown]
# Modelo 2: A Regressão Linear Múltipla

# %%
# Criar um novo DataFrame (df2) sem as colunas especificadas
df2 = df.drop(['6-month-SMA', '12-month-SMA', '6-month-EWMA', '12-month-EWMA','21868 - Indicadores da produção (2022=100) - Insumos da construção civil - Índice'], axis=1).copy()


# %%
#df2 = df2.reset_index()

# %%

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# %%
# Definir as variáveis independentes (X) e a variável dependente (y)
X = df2[['21859 - Indicadores da produção (2022=100) - Geral - Índice',
         '21861 - Indicadores da produção (2022=100) - Extrativa mineral - Índice',
         '21863 - Indicadores da produção (2022=100) - Bens de capital - Índice',
         '21864 - Indicadores da produção (2022=100) - Bens intermediários - Índice',
         '21865 - Indicadores da produção (2022=100) - Bens de consumo - Índice',
         '21866 - Indicadores da produção (2022=100) - Bens de consumo duráveis - Índice',
         '21867 - Indicadores da produção (2022=100) - Semiduráveis e não duráveis - Índice',
         '11752 - Índice da taxa de câmbio real efetiva (IPCA) - Jun/1994=100 - Índice',
         '11753 - Índice da taxa de câmbio real (IPCA) - Jun/1994=100 - Dólar americano - Índice',
         '20360 - Índice da taxa de câmbio efetiva nominal - Jun/1994=100 - Índice']]

y = df2['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice']



# %%
# Definir as variáveis independentes (X) excluindo a variável alvo
X_train = df2.iloc[:210].drop(['Data', '21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'], axis=1)
X_test = df2.iloc[210:262].drop(['Data', '21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'], axis=1)

# Definir a variável dependente (y) para os conjuntos de treinamento e teste
y_train = df2['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'][:210]
y_test = df2['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'][210:262]


# %%
# Criar o modelo de regressão linear
model = LinearRegression()

# %%
# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# %%

# Imprimir as métricas de desempenho
print("Erro médio quadrático:", mse)
print("Coeficiente de determinação (R²):", r2)

# %%
# Calcular métricas de desempenho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# %%
# Fazer previsões usando os dados de teste
y_pred = model.predict(X_test)

# %%
y_pred

# %%
# Criar uma série de datas correspondentes aos meses para os dados de teste
dates_test = df2.iloc[210:262]['Data']

# %%
# Criar um DataFrame com as datas e as previsões
predictions_df = pd.DataFrame({'Data': dates_test, 'Previsões': y_pred})

# %%
predictions_df

# %%
# Extrair o histórico correspondente aos dados de teste
historical_data = df2.iloc[210:262]

# %%
# Configurar o índice do DataFrame como as datas
predictions_df.set_index('Data', inplace=True)

# %%
# Converter a coluna de datas para uma lista de datas
datas = historical_data['Data'].tolist()

# %%
import pandas as pd

# Criar uma lista de datas para as previsões
dates = pd.date_range(start='2023-01-01', periods=len(predictions_df), freq='MS')

# Adicionar as datas ao DataFrame predictions_df
predictions_df['Data'] = dates

# Definir a coluna 'Data' como índice
predictions_df.set_index('Data', inplace=True)


# %%
# Criar DataFrame apenas com as colunas 'Data' e '21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'
df_transformacao = df2[['Data', '21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice']].copy()


# Definir a coluna 'Data' como índice do DataFrame df_transformacao
df_transformacao.set_index('Data', inplace=True)
# Exibir as primeiras linhas do DataFrame para verificação
df_transformacao.head()



# %%
import matplotlib.pyplot as plt

# Remover linhas com valores NaN
df_ind_clean = df_ind.dropna(subset=['Previsões'])



# %%
# Converter o índice de predictions_df para o tipo datetime
predictions_df.index = pd.to_datetime(predictions_df.index)

# Selecionar as previsões até junho de 2024
predictions_until_june_2024 = predictions_df.loc[predictions_df.index <= '2024-06-01']

# %%
predictions_until_june_2024

# %%
df_transformacao

# %%
df_combined = pd.merge(df_transformacao.reset_index(), predictions_until_june_2024.reset_index(), on = 'Data', how = 'left').reset_index(drop = True).set_index('Data')
df_combined

# %%
# Plotar o histórico e as previsões até junho de 2024
plt.figure(figsize=(12, 6))
plt.plot(df_combined['21862 - Indicadores da produção (2022=100) - Indústria de transformação - Índice'], marker='o', label='Histórico')
plt.plot(df_combined['Previsões'], marker='o', label='Previsões')
plt.xlabel('Mês/Ano')
plt.ylabel('Índice da Indústria de Transformação')
plt.title('Histórico e Previsões da Indústria de Transformação')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# Modelo 3: Modelo auto-regressivo

# %%
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import arma_order_select_ic

# %%
data = df_ind_transf['Indústria de transformação']

# %%
max_lags = 10 
res = arma_order_select_ic(data, max_ar=max_lags, ic='aic', trend='c')
p = res.aic_min_order[0] 

# %%
# Dividindo os dados em conjuntos de treinamento e teste
train_data = data[:210]
test_data = data[210:]

# %%
model = AutoReg(train_data, lags=p)
model_fit = model.fit()

# %%
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

# %%
plt.figure(figsize=(15, 10))
plt.plot(train_data, label='Treinamento')
plt.plot(test_data, label='Teste')
plt.plot(predictions, label='Previsões')
plt.legend()
plt.show()

# %%
predictions

# %%
# Calcular as previsões
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

# Calcular o MAE
mae = mean_absolute_error(test_data, predictions)

# Calcular o MSE
mse = mean_squared_error(test_data, predictions)

# Calcular o R-squared (R²)
r2 = r2_score(test_data, predictions)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)


