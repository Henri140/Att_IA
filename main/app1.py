import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# Nome do arquivo do modelo (deve ser o mesmo salvo pelo train.py)
dados= './assets/dados_aluno - dados.csv.pkl'

@st.cache_resource # Cache para carregar o modelo apenas uma vez
def carregar_modelo(caminho_modelo):
    """
    Carrega o modelo do arquivo .pkl.
    """
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
    """
    Função principal que executa o App Streamlit.
    """
    
    # --- Carregamento do Modelo ---
    modelo, classes_modelo = carregar_modelo(NOME_ARQUIVO_MODELO)
    
    # --- Título e Subtítulo ---
    # LACUNA 4: Dê um título ao seu App usando 'st.title()'
    st.title('...')
    st.subheader('Estudo de Caso da Imersão em IA (Aulas 1-3)')

    # Se o modelo não existir, exibe um aviso (pronto)
    if modelo is None:
        st.error(f"Arquivo do modelo ('{NOME_ARQUIVO_MODELO}') não encontrado.")
        st.warning("Execute o script 'python train.py' no terminal para treinar e criar o modelo.")
        st.stop() # Para a execução do app

    # --- Interface do Usuário (Entradas) na Sidebar ---
    # LACUNA 5: Crie um cabeçalho na 'sidebar' (barra lateral)
    # Dica: st.sidebar.header('...')
    st.sidebar.header('Insira os dados do Aluno:')

    # LACUNA 6: Crie um 'st.sidebar.slider' para 'Horas de Estudo'
    # (Nome: 'Média de Horas de Estudo/semana', min: 0, max: 20, padrão: 5)
    horas_estudo = st.sidebar.slider('...', 0, 20, 5)

    # LACUNA 7: Crie um 'st.sidebar.number_input' para 'Faltas'
    # (Nome: 'Número Total de Faltas', min: 0, max: 50, padrão: 3)
    faltas = st.sidebar.number_input('...', min_value=0, max_value=50, value=3)

    # (Este vem pronto)
    nota_p1 = st.sidebar.number_input('Nota da Primeira Prova (0-10)', min_value=0.0, max_value=10.0, value=5.0, step=0.5)

    st.markdown("---")
    st.write("Este App foi construído no curso de Programação em IA Generativa.")

# (Bloco pronto)
if __name__ == "__main__":
    main()