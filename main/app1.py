import pandas as pd
import joblib
import streamlit as st
import os
from sklearn.neighbors import KNeighborsClassifier

NOME_ARQUIVO_MODELO = './assets/dados_aluno - dados.csv.pkl'
dados= './assets/dados_aluno - dados.csv.pkl'

@st.cache_resource
def carregar_modelo(caminho_modelo):

    if not os.path.exists(caminho_modelo):
        return None, None
    try:
        modelo = joblib.load(caminho_modelo)
        classes_modelo = modelo.classes_
        return modelo, classes_modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None

def main():

    modelo, classes_modelo = carregar_modelo(NOME_ARQUIVO_MODELO)

    st.title('QQC')
    st.subheader('Estudo de Caso da Imersão em IA (Aulas 1-3)')

    if modelo is None:
        st.error(f"Arquivo do modelo ('{NOME_ARQUIVO_MODELO}') não encontrado.")
        st.warning("Execute o script 'python train.py' no terminal para treinar e criar o modelo.")
        st.stop()

    st.sidebar.header('Insira os dados do Aluno:')

    
    horas_estudo = st.sidebar.slider( 'Média de Horas de Estudo/semana',0, 20, 5)

 
    faltas = st.sidebar.number_input( 'Quantidade de faltas', min_value=0, max_value=50, value=3)

    nota_p1 = st.sidebar.number_input('Nota da Primeira Prova (0-10)', min_value=0.0, max_value=10.0, value=5.0, step=0.5)

    st.markdown("---")
    st.write("Este App foi construído no curso de Programação em IA Generativa.")
    if st.button("Q"):
        dados = pd.DataFrame(
            [[horas_estudo, faltas, nota_p1]],
            columns=['horas_estudo', 'faltas', 'nota_p1']
        )

        resultado = modelo.predict(dados)[0]

        if resultado == 1:
            st.success("✅ Aluno aprovado")
        else:
            st.error("❌ Aluno reprovado")

if __name__ == "__main__":
    main()