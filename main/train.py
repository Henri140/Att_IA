import pandas as pd
import joblib
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import io

# -------------------------------------------------------------------
# IMPORTANTE: Instruções para o Repositório de Dados
# -------------------------------------------------------------------
# 1. Crie um Google Sheet
# 2. Insira dados com estes cabeçalhos: horas_estudo,faltas,nota_p1,resultado
#    Ex:
#    horas_estudo,faltas,nota_p1,resultado
#    2.5,8,4.5,Reprovado
#    5.0,2,7.0,Aprovado
#    ... (adicione 10-15 linhas)
# 3. Vá em "Arquivo" -> "Compartilhar" -> "Publicar na web"
# 4. Selecione "Valores separados por vírgula (.csv)" e clique em "Publicar"
# 5. COPIE O LINK GERADO E COLE ABAIXO:
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# IMPORTANTE: Cole a URL pública do seu Google Sheet CSV aqui
# -------------------------------------------------------------------
# LACUNA 4: Cole a URL pública do seu Google Sheet CSV
URL_DADOS = "COLE_A_URL_AQUI"

NOME_ARQUIVO_MODELO = 'Com lacunas\modelo_desempenho.pkl'

def treinar_modelo():
    print(f"Baixando dados de {URL_DADOS}...")
    
    # LACUNA 5: Use 'pd.read_csv()' para ler a 'URL_DADOS'
    data = pd.read_csv(...)
    
    print("--- Dados Carregados ---")
    print(data.head())

    # (Bloco pronto)
    print("--- Preparando dados para o treino ---")
    features = ['horas_estudo', 'faltas', 'nota_p1']
    target = 'resultado'
    X = data[features]
    Y = data[target]

    # LACUNA 6: Instancie o 'KNeighborsClassifier' com 3 vizinhos (n_neighbors=3)
    modelo = ...(...)

    # LACUNA 7: Treine o 'modelo' usando o método '.fit()'
    # Dica: Passe as entradas (X) e as saídas (Y)
    modelo.fit(..., ...)

    print(f"--- Modelo Treinado! Classes: {modelo.classes_} ---")

    # LACUNA 8: Use 'joblib.dump()' para salvar o 'modelo'
    # Dica: Salve no arquivo 'NOME_ARQUIVO_MODELO'
    joblib.dump(..., ...)
    
    print(f"--- Modelo salvo com sucesso em '{NOME_ARQUIVO_MODELO}' ---")

# (Bloco pronto)
if __name__ == "__main__":
    treinar_modelo()