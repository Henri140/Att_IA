import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os


NOME_ARQUIVO_DADOS = '../assets/dados_aluno - dados.csv'

@st.cache_resource
def treinar_modelo():
    if not os.path.exists(NOME_ARQUIVO_DADOS):
        return None
    
    dados = pd.read_csv(NOME_ARQUIVO_DADOS)
    X = dados[['horas_estudo', 'faltas', 'nota_p1']]
    y = dados['nota_final']
    
    modelo = RandomForestRegressor()
    modelo.fit(X, y)
    return modelo

def main():
    modelo = treinar_modelo()

    st.title('Previsão de Desempenho Escolar')
    st.subheader('Estudo de Caso da Imersão em IA')

    if modelo is None:
        st.error(f"Arquivo '{NOME_ARQUIVO_DADOS}' não encontrado na pasta.")
        st.stop()

    st.sidebar.header('Insira os dados do Aluno:')

    horas_estudo = st.sidebar.slider('Média de Horas de Estudo/semana', 0, 40, 10)
    faltas = st.sidebar.number_input('Número Total de Faltas', min_value=0, max_value=50, value=3)
    nota_p1 = st.sidebar.number_input('Nota da Primeira Prova (0-10)', min_value=0.0, max_value=10.0, value=5.0, step=0.5)

    if st.button('Prever Resultado'):
        entrada = pd.DataFrame([{
            'horas_estudo': horas_estudo,
            'faltas': faltas,
            'nota_p1': nota_p1
        }])

        nota_prevista = modelo.predict(entrada)[0]
        
        st.metric(label='Nota Final Prevista', value=f'{nota_prevista:.2f}')

    st.markdown('---')
    st.write('Este App foi construído no curso de Programação em IA Generativa.')

if __name__ == "__main__":
    main()