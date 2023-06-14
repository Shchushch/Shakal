import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import os
from sys import getsizeof
from PIL import Image as im
from sklearn.metrics import mean_absolute_error


st.title("Шакализатор")
"### Тут можно ~~зашакалить~~ сжать картинку"
info=("Эта программа использует **сингулярное разложение**, для понижения ранга матрицы картинки до заданного пользователем. "
    "В результате, мы получаем сжатую (aka зашакаленную) картинку. "
    "Сразу предупрежу, что такой алгоритм очень аккуратный, так что реально заметный результат можно будет увидеть только если сжимать строку "
    "до **пары процентов** от изначального размера картинки в пикселях. "
    "Также хочу отметить, что **алгоритм небыстрый**, так что обработка изображения с высоким разрешением занимает достаточно много времени")

if st.button("Что это такое",type='secondary'):
    st.divider()
    info
    st.divider()

    

file = st.file_uploader("Кладёшь картинку сюда")
plt.rcParams.update({'image.cmap': 'gray'})
size =10

@st.cache_data

def shakalizator_1c(img_1c:pd.DataFrame,shakal:int)->pd.DataFrame: 
    U, sing_vals, V = np.linalg.svd(img_1c)
    trunc_U = U[:, :shakal]
    trunc_V = V[:shakal, :]
    sigma = np.zeros((img_1c.shape[0], img_1c.shape[1]), dtype=float)
    np.fill_diagonal(sigma, sing_vals)
    trunc_sigma = sigma[:shakal, :shakal]
    img2=trunc_U@trunc_sigma@trunc_V
    return img2
def shakalizator_rgb(img:pd.DataFrame,shakal:int)->pd.DataFrame:
    img2=np.zeros((img.shape[0],img.shape[1],3))
    for i in range(3):
        img2[:,:,i]=shakalizator_1c(img[:,:,i],shakal)#:]=shakalizator_1c(img[:,:,i],shakal)
    img2=img2.astype(np.uint8)
    return img2
def update_slider():
    st.session_state.slider = st.session_state.numeric
def update_numin():
    st.session_state.numeric = st.session_state.slider 

with st.container():
        if file is not None:
            img = plt.imread(file)[:, :, 0]
            size = img.shape[0]
        shakal_pic=st.button('Шакализовать',type='primary',key='shakal_pic',use_container_width=True)
        shakal = st.number_input('Введите ~количество шакалов~ до какого ранга сжать',key = 'numeric', min_value=1, max_value=size, value=size,on_change=update_slider)
        slider_value = st.slider('Можете ещё вот так выбрать', min_value = 1, 
                            value = shakal, 
                            max_value = size,
                            step = 1,
                            key = 'slider', on_change= update_numin)
        col1, col2 = st.columns([1,1])
if file is not None:
    img = plt.imread(file)
    size = img[:, :, 0].shape[0]

    with col1:
        "Изображение до шакализации"
        st.divider()
        st.image(img)
        st.divider()
        f'Размер картинки: **{size}**x**{img.shape[1]}**'#\
           # Или {img.size*img.mode}'
   
    with col2:
        "Изображение после шакализации"
        st.divider()
    if shakal_pic:
        shakaled = shakalizator_rgb(img,shakal)
        with col2:
            st.image(im.fromarray(shakaled).convert('RGB'))
            st.divider()
            nsamples, nx, ny = shakaled.shape
            d2_shakaled,d2_img = shakaled.reshape((nsamples,nx*ny)),img.reshape((nsamples,nx*ny))
            R,G,B=mean_absolute_error(shakaled[:,:,0],img[:,:,0]),mean_absolute_error(shakaled[:,:,1],img[:,:,1]),mean_absolute_error(shakaled[:,:,2],img[:,:,2])
            RGB= mean_absolute_error(d2_shakaled,d2_img)

            f'Размер картинки: **{size}**x**{shakaled.shape[1]}**\
            \nСжато на **{(100-(shakal/size*100)):.2f}%**'
            f"Средняя абсолютная ошибка:\
            \nRGB: **{RGB}** \
            \nR: **{R}** \
            \nG: **{G}** \
            \nB: **{B}** \
                "
            
    else:
        text='Ещё не зашакалили'
        st.markdown(f"<div style='text-align: center;vertical-align: middle'>{text}</div>",unsafe_allow_html=True)


