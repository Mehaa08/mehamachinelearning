import streamlit as st
import pandas as pd


st.title('˚˖𓍢ִ໋🌷͙֒✧˚.🎀༘⋆Meha Machine Learning App')

st.info('This app builds a ml models')

with st.expander('Data'):
  st.write('**Raw Data**')
  a = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  st.write(a)

  st.write('**X**')
  X = a.drop('species',axis = 1)
  X

  st.write('**y**')
  y = a.species
  y

