import os
from pathlib import Path

import plotly.express as px
import streamlit as st

from data import Loader, project


def get_totals(min_power_of_ten=1, max_power_of_ten=4):
    values = []
    for exp in range(min_power_of_ten, max_power_of_ten):
        values += list(range(10 ** exp, 10 ** (exp + 1), 10 ** exp))
    values += [10 ** max_power_of_ten]
    return values


def model_controls(suffix):
    cols = st.beta_columns(2)
    with cols[0]:
        architecture = st.selectbox('Architecture', ['MusiCNN', 'VGG', 'VGGish'], key=f'architecture-{suffix}')

    if architecture in ['MusiCNN', 'VGG']:
        datasets = ['MSD', 'MTT']
    elif architecture == 'VGGish':
        datasets = ['AudioSet']
    else:
        datasets = []

    with cols[1]:
        dataset = st.selectbox('Dataset', datasets, key=f'dataset-{suffix}')

    layers = ['Embeddings'] + (['Taggrams'] if architecture != 'VGGish' else [])
    layer = st.selectbox('Layer', layers, key=f'layer-{suffix}')

    return architecture, dataset, layer


def model_plot(loader, architecture, dataset, layer, projection, total, shuffle):
    data = loader.load_embeddings(dataset.lower(), architecture.lower(), layer.lower(), total, shuffle)

    projection = project(data, projection)
    fig = px.scatter(projection, x=0, y=1, height=800)

    fig.update_layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
    st.plotly_chart(fig, use_container_width=True)


def main():
    try:
        data_path = os.environ['DATA_PATH']
    except KeyError:
        raise RuntimeError('Please set the DATA_PATH environment variable')

    title = 'Visualizing MTG-Jamendo embeddings'
    st.set_page_config(title, layout='wide', initial_sidebar_state='expanded')
    st.title(title)

    with st.sidebar:
        st.header('Amount of data')
        total = st.select_slider('Number of tracks', get_totals(), 10)
        shuffle = st.checkbox('Random')

        st.header('Model')
        model = model_controls('')
        compare = st.checkbox('Compare')
        if compare:
            st.header('Other model')
            model_to_compare = model_controls('cmp')

        st.header('Visualization')
        projection = st.selectbox('Projection', ['PCA', 't-SNE', 'UMAP'])

    loader = Loader(Path(data_path))
    if not compare:
        model_plot(loader, *model, projection, total, shuffle)
    else:
        cols = st.beta_columns(2)
        with cols[0]:
            model_plot(loader, *model, projection, total, shuffle)
        with cols[1]:
            model_plot(loader, *model_to_compare, projection, total, shuffle)


if __name__ == '__main__':
    main()
