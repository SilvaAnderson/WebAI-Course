
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carrega o modelo treinado
model = joblib.load('random_forest_model.joblib')

# Carrega os dados para obter as features e valores possíveis
df = pd.read_csv('Database/1. Registros abertos e fechados _ Todas as mudanças.csv', sep=';')

# Features usadas no modelo (ajuste conforme o notebook)
feature_cols = [
    'Tier',
    'Prod Impact',
    'Reg. Impact',
    'Change Driver',
    'pre_implementation_count',
    'implementation_count',
    'post_implementation_count',
    'tempo_medio_tarefas',
    'desviopad_tarefas',
    'Is there artwork impact?'
]

# Função para entrada de dados do usuário
def user_input_features():
    col1, col2 = st.columns(2)
    with col1:
        tier = st.selectbox('Tier', sorted(df['Tier'].dropna().unique()))
        prod_impact = st.selectbox('Prod Impact', ['No', 'Yes'])
        reg_impact = st.selectbox('Reg. Impact', ['No', 'Yes'])
        change_driver = st.selectbox('Change Driver', sorted(df['Change Driver'].dropna().unique()))
        pre_impl = st.number_input('Pre-Implementation Count', min_value=0, step=1)
    with col2:
        impl = st.number_input('Implementation Count', min_value=0, step=1)
        post_impl = st.number_input('Post-Implementation Count', min_value=0, step=1)
        tempo_medio = st.number_input('Tempo médio tarefas', min_value=0.0, step=1.0)
        desvio = st.number_input('Desvio padrão tarefas', min_value=0.0, step=1.0)
        artwork_impact = st.selectbox('Is there artwork impact?', ['No', 'Yes'])
    data = {
        'Tier': tier,
        'Prod Impact': prod_impact,
        'Reg. Impact': reg_impact,
        'Change Driver': change_driver,
        'pre_implementation_count': pre_impl,
        'implementation_count': impl,
        'post_implementation_count': post_impl,
        'tempo_medio_tarefas': tempo_medio,
        'desviopad_tarefas': desvio,
        'Is there artwork impact?': artwork_impact
    }
    return pd.DataFrame([data])

st.title('Classificação de Mudanças')
st.write('Este formulário permite classificar mudanças propostas usando um modelo de aprendizado de máquina.')
st.write('Preencha o formulário para classificar a mudança proposta:')

input_df = user_input_features()

# Pré-processamento: dummies igual ao notebook
input_proc = pd.get_dummies(input_df)

# Garante que só usa colunas que existem no DataFrame
existing_cols = [col for col in feature_cols if col in df.columns]
df_proc = pd.get_dummies(df[existing_cols])
input_proc = input_proc.reindex(columns=df_proc.columns, fill_value=0)

if st.button('Classificar'):
    # Garante que as colunas estejam na mesma ordem e quantidade do modelo
    input_proc = input_proc.reindex(columns=model.feature_names_in_, fill_value=0)
    pred = model.predict(input_proc)[0]
    st.success(f'Classificação: {pred}')
