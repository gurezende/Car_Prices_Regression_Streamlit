'''
Projeto: Criação de um Web App em Streamlit com o modelo treinado para previsão de preços de carros.
O aplicativo receberá input do usuário com as informações do carro que deseja consultar e retornará um valor estimado
para o automóvel, segundo o modelo Random Forest.
'''
# Basico
import pandas as pd
import numpy as np
import pickle

# Streamlit
import streamlit as st

# Imports Scikit Learn
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Image Reading
from PIL import Image

# ----------------------------------------------------------------------------------

# Carregar dataset de treinamento
path = 'carros.csv'
df = pd.read_csv(path)

# ----------------------------------------------------------------------------------
# Imagem de Cabeçalho
# ----------------------------------------------------------------------------------
# Codifica imagem
image = Image.open('consultor-logo.png')
# Aplica Imagem no App
st.image(image)

# ----------------------------------------------------------------------------------
# Texto inicial
# ----------------------------------------------------------------------------------
"""
##  | O Consultor Virtual
O Consultor Virtual é um Web App desenvolvido para te ajudar a estimar o valor do seu automóvel.
Para usá-lo, basta preencher o formulário e apertar o botão *Submeter*.

##  | Nossos Dados
Os dados usados para treinamento foram coletados da internet no mês de Agosto/2021.
Foram coletados 1347 registros para o treinamento do modelo.
"""

# Cria colunas para o sumário de dados
col1, col2, col3, col4 = st.columns(4)
# coluna 1
with col1:
    st.title(df.car.count())
    st.text('registros')
#coluna 2
with col2:
    st.title(df.car.nunique())
    st.text('modelos')
# coluna 3
with col3:
    st.title(df.fabricante.nunique())
    st.text('fabricantes')
# coluna 4
with col4:
    st.title(df.estado.nunique())
    st.text('estados')

# ----------------------------------------------------------------------------------
# Mapa
# ----------------------------------------------------------------------------------
df_map = df[['estado','latitude', 'longitude', 'preco']]
df_map.preco = df_map.preco/1000

url = 'C:/Users/1770858/Documents/Gus/Streamlit/'
#import json

# Carregar o arquivo json

#state_geo = json.load(open(f'{url}brazil_geo.json'))
# Criar o mapa base
#m = folium.Map(location=[-15.7801, -47.9292], zoom_start=4)
#Criar a camada Choroplet
#folium.Choropleth(
#    geo_data=state_geo,
#    name='choropleth',
#    data=df_map,
#    columns=['estado', 'preco'],
#    key_on='feature.id',
#    fill_color='YlOrRd',
#    fill_opacity=0.7,
#    line_opacity=0.2,
#    legend_name='Preço dos Automóveis Anunciados (em mil R$)'
#).add_to(m)
# Mostrar o mapa no App
#folium_static(m)
st.map(df_map, zoom=4)

# ----------------------------------------------------------------------------------
# PREVISÕES
# ----------------------------------------------------------------------------------
st.title('ESTIME O PREÇO DO SEU AUTOMÓVEL')
"""
Abra a aba lateral, complete o formulário com os dados do seu veículo

Aperte Submeter para estimar o valor.
## Preço estimado em Reais:
"""

# Carrega o modelo
filename = url + 'RF_car_prices.sav'
model = pickle.load(open(filename, 'rb'))

with st.form("my_form"):
    # 'motor'
    i_motor = st.sidebar.select_slider("Potência do Motor",
                                options= sorted( df.motor.unique() ))

    # 'cambio'
    i_cambio = st.sidebar.radio('Tipo de Cambio',
                                ('Manual', 'Automático'))
    if i_cambio == 'Manual':
        i_cambio = 0
    else:
        i_cambio = 1


    # 'ano_fabrica'
    i_ano = st.sidebar.select_slider("Ano de Fabricação",
                                     options= sorted( df.ano_fabrica.unique() ),
                                     value=2020 )

    #'km'
    i_km = st.sidebar.text_input('Quilometragem', '10550')

    # 'fabricante'
    i_fabricante = st.sidebar.selectbox('Fabricante', sorted( df.fabricante.unique() ) )

    # 'combustível'
    i_combustivel = st.sidebar.selectbox('Tipo de Combustível', df.combustivel.unique() )

    #'anunciante'
    i_anunciante = st.sidebar.radio('Você é',
                                    ('Pessoa Física', 'Loja', 'Concessionária') )

    # 'região'
    i_regiao = st.sidebar.radio('Você está localizado em qual região do Brasil?',
                                ('Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul') )

    # df com escolhas do cliente
    df_to_predict = pd.DataFrame({'motor': [i_motor], 'automatico': [i_cambio], 'ano_fabrica': [i_ano], 'km': [int(i_km)],
                                  'fabricante_AUDI': [1 if i_fabricante == 'AUDI' else 0],'fabricante_BMW': [1 if i_fabricante == 'BMW' else 0],
                                  'fabricante_CHERY': [1 if i_fabricante == 'CHERY' else 0], 'fabricante_CHEVROLET': [1 if i_fabricante == 'CHEVROLET' else 0],
                                  'fabricante_CITROEN': [1 if i_fabricante == 'CITROEN' else 0],'fabricante_DODGE': [1 if i_fabricante == 'DODGE' else 0],
                                  'fabricante_FIAT': [1 if i_fabricante == 'FIAT' else 0], 'fabricante_FORD': [1 if i_fabricante == 'FORD' else 0],
                                  'fabricante_HONDA': [1 if i_fabricante == 'HONDA' else 0], 'fabricante_HYUNDAI': [1 if i_fabricante == 'HYUNDAI' else 0],
                                  'fabricante_JAC': [1 if i_fabricante == 'JAC' else 0], 'fabricante_JAGUAR': [1 if i_fabricante == 'JAGUAR' else 0],
                                  'fabricante_JEEP': [1 if i_fabricante == 'JEEP' else 0], 'fabricante_KIA': [1 if i_fabricante == 'KIA' else 0],
                                  'fabricante_LAND ROVER': [1 if i_fabricante == 'LAND ROVER' else 0], 'fabricante_LEXUS': [1 if i_fabricante == 'LEXUS' else 0],
                                  'fabricante_LIFAN': [1 if i_fabricante == 'LIFAN' else 0], 'fabricante_MERCEDES-BENZ': [1 if i_fabricante == 'MERCEDES-BENZ' else 0],
                                  'fabricante_MINI': [1 if i_fabricante == 'MINI' else 0],'fabricante_MITSUBISHI': [1 if i_fabricante == 'MITSUBISHI' else 0],
                                  'fabricante_NISSAN': [1 if i_fabricante == 'NISSAN' else 0], 'fabricante_PEUGEOT': [1 if i_fabricante == 'PEUGEOT' else 0],
                                  'fabricante_PORSCHE': [1 if i_fabricante == 'PORSCHE' else 0], 'fabricante_RENAULT': [1 if i_fabricante == 'RENAULT' else 0],
                                  'fabricante_SUZUKI': [1 if i_fabricante == 'SUZUKI' else 0], 'fabricante_TOYOTA': [1 if i_fabricante == 'TOYOTA' else 0],
                                  'fabricante_VOLKSWAGEN': [1 if i_fabricante == 'VOLKSWAGEN' else 0],'fabricante_VOLVO': [1 if i_fabricante == 'VOLVO' else 0],
                                  'combustivel_DIESEL': [1 if i_combustivel == 'DIESEL' else 0], 'combustivel_FLEX': [1 if i_combustivel == 'FLEX' else 0],
                                  'combustivel_GASOLINA': [1 if i_combustivel == 'GASOLINA' else 0], 'combustivel_HIBRIDO': [1 if i_combustivel == 'HIBRIDO' else 0],
                                  'anunciante_Concessionária': [1 if i_anunciante == 'Concessionária' else 0], 'anunciante_Loja': [1 if i_anunciante == 'Loja' else 0],'anunciante_Pessoa Física': [1 if i_anunciante == 'Pessoa Física' else 0],
                                  'regiao_Centro-Oeste': [1 if i_regiao == 'Centro-Oeste' else 0], 'regiao_Nordeste': [1 if i_regiao == 'Nordeste' else 0],
                                  'regiao_Norte': [1 if i_regiao == 'Norte' else 0], 'regiao_Sudeste': [1 if i_regiao == 'Sudeste' else 0], 'regiao_Sul': [1 if i_regiao == 'Sul' else 0]})

    #Previsão do modelo
    X = pd.get_dummies(df_to_predict)
    previsao = model.predict(X)
    previsao = previsao.tolist()[0]
    previsao = f' R$ { round( previsao , 2 ):,}'


    # Every form must have a submit button.
    submitted = st.form_submit_button("Submeter")
    if submitted:
        st.title(previsao)




# Sobre o autor:
st.sidebar.write('')
st.sidebar.markdown('---')
st.sidebar.write('| Sobre:')
st.sidebar.write('Este WebApp foi desenvolvido por Gustavo R Santos.')
st.sidebar.write('Gustavo tem 13 anos de experiência no mercado de TI e atualmente é Cientista de Dados do Food Lion, nos EUA.')
st.sidebar.write('| Contato:')
st.sidebar.write('[Visite minha página no Linkedin](https://www.linkedin.com/in/gurezende/)')



