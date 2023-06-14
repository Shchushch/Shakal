import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import os
#"st.session_state object:" , st.session_state

st.title("Шакализатор")
"### Тут можно ~~зашакалить~~ сжать картинку"
info=("Эта программа использует **сингулярное разложение**, для понижения ранга матрицы картинки до заданного пользователем. "
    "В результате, мы получаем сжатую (aka зашакаленную) картинку. "
    "Сразу предупрежу, что такой алгоритм очень аккуратный, так что реально заметный результат можно будет увидеть только если сжимать строку "
    "до **пары процентов** от изначального размера картинки в пикселях. "
    "Также хочу отметить, что **алгоритм небыстрый**, так что обработка изображения с высоким разрешением занимает достаточно много времени")
#my_container = st.container()
# if 'my_container' not in st.session_state:
#     st.session_state.my_container = my_container
if st.button("Что это такое",type='secondary'):
    st.divider()
    info
    st.divider()

    

file = st.file_uploader("Кладёшь картинку сюда")
#pic=st.image()
plt.rcParams.update({'image.cmap': 'gray'})
size =10

@st.cache_data
def shakalizator(img:pd.DataFrame,shakal:int)->pd.DataFrame: 
    U, sing_vals, V = np.linalg.svd(img)
    trunc_U = U[:, :shakal]
    trunc_V = V[:shakal, :]
    sigma = np.zeros((img.shape[0], img.shape[1]), dtype=float)
    np.fill_diagonal(sigma, sing_vals)
    trunc_sigma = sigma[:shakal, :shakal]
    img2=trunc_U@trunc_sigma@trunc_V
    #st.write(type(img2))
    return img2
def update_slider():
    st.session_state.slider = st.session_state.numeric
def update_numin():
    st.session_state.numeric = st.session_state.slider 

if file is not None:
    img = plt.imread(file)[:, :, 0]
    size = img.shape[0]
with st.container():
    col1, col2 = st.columns([1,1])
with col1:
    "Изображение до шакализации"
    if st.button('Чё тут у нас?',type='primary',key='pic_show'):
        st.image(img)
        f'Размер картинки: **{size}**x**{img.shape[1]}**'
shakal = st.number_input('Введите количество шакалов',key = 'numeric', min_value=1, max_value=size, value=size,on_change=update_slider)
slider_value = st.slider('Можете ещё вот так выбрать', min_value = 1, 
                        value = shakal, 
                        max_value = size,
                        step = 1,
                        key = 'slider', on_change= update_numin)
fig,ax = plt.subplots()
ax.grid(False)
ax.axis('off')
with col2:
    "Изображение до шакализации"
    if st.button('Шакализовать',type='primary',key='shakal_pic'):
        shakaled = shakalizator(img,shakal)
        ax.imshow(shakaled)
        #st.image(fig)
        #st.pyplot(ax.imshow(shakaled))
        fig
    else:
        text='Ещё не зашакалили'
        st.markdown(f"<div style='text-align: center;vertical-align: middle'>{text}</div>",unsafe_allow_html=True)


