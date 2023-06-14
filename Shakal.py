import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import os

st.title("Шакализатор")
st.write("### Тут можно зашакалить картинку")
file = st.file_uploader("Кладёшь картинку сюда")
plt.rcParams.update({'image.cmap': 'gray'})
size =10

@st.cache_data
def shakalizator(img:pd.DataFrame,shakal:int):
    U, sing_vals, V = np.linalg.svd(img)
    trunc_U = U[:, :shakal]
    trunc_V = V[:shakal, :]
    sigma = np.zeros((img.shape[0], img.shape[1]), dtype=float)
    np.fill_diagonal(sigma, sing_vals)
    trunc_sigma = sigma[:shakal, :shakal]
    img2=trunc_U@trunc_sigma@trunc_V
    return img2


if file is not None:
    img = plt.imread(file)[:, :, 0]
    size = img.shape[0]

fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
fig.tight_layout()
ax[0].set_title("Исходное Изображение")
ax[0].grid(False)
ax[0].axis('off')


ax[1].set_title("Изображение после шакализации")
ax[1].grid(False)
ax[1].axis('off')
if st.button('Чё тут у нас?'):
    ax[0].imshow(img)
    #ax[1].figure(figsize=img.shape())
    fig
shakal = st.number_input('Введите количество шакалов', min_value=1, max_value=size, value=size)

if st.button('Шакализовать'):
    ax[0].imshow(img,)
    ax[1].imshow(shakalizator(img,shakal))#shakalizator(img,shakal)
    fig
else:
    st.write("Ещё не зашакалили")


