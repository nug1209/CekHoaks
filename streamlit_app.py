import streamlit as st
import pandas as pd

df = pd.read_csv('results.csv')
# print(df_res.head())
st.write('Cek Hoaks')

input_text = st.text_area('Teks yang ingin dicek...')

submit_button = st.button('Kirim')

if submit_button:
    st.write('Submit button clicked.')

display_text1 = 'Ditemukan hoaks baru yang mirip teks. Kemungkinan besar teks adalah hoaks.'
display_text2 = 'Ditemukan hoaks lama yang mirip teks. Mungkin teks adalah hoaks lama, ada ada kemungkinan juga teks telah menjadi fakta.'
display_text3 = 'Tidak ditemukan hoaks yang mirip teks. Mungkin teks bukan hoaks, mungkin teks adalah hoaks yang tidak atau belum dicek oleh tim.'

if (df['similarity'][0] > 0.5) and (df['time_delta'][0] < 15):
  st.write(display_text1)
elif (df['similarity'][0] > 0.5) and (df['time_delta'][0] > 14):
  st.write(display_text2)
else:
  st.write(display_text3)

df_res = df.drop(columns=['text'])


html_str = f"""

<p><a href={df['link'][0]}>{df['title'][0]}</a></p>
"""
st.markdown(html_str, unsafe_allow_html=True)



df_res
