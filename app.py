import pandas as pd
import numpy as np

import re
import string
from torch import clamp
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

class TokenSimilarity:

  def load_pretrained(self, from_pretrained:str='indobenchmark/indobert-base-p1'):
    self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
    self.model = AutoModel.from_pretrained(from_pretrained)
  
  def __cleaning(self, text:str):
    text = text.translate(str.maketrans('', ''))

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'/s+', ' ', text).strip()

    return text

  def __process(self, first_token:str, second_token:str):

    inputs = self.tokenizer([first_token, second_token], max_length=self.max_length, truncation=self.truncation, padding=self.padding, return_tensors='pt')
    
    attention = inputs.attention_mask
    
    outputs = self.model(**inputs)

    embeddings = outputs[0]

    embeddings = outputs.last_hidden_state

    mask = attention.unsqueeze(-1).expand(embeddings.shape).float()

    masked_embeddings = embeddings * mask

    summed = masked_embeddings.sum(1)

    counts = clamp(mask.sum(1), min = 1e-9)

    mean_pooled = summed / counts

    return mean_pooled.detach().numpy()

  def predict(self, first_token:str, second_token:str, return_as_embeddings:bool=False, max_length:int=16, truncation:bool=True, padding:str='max_length'):
    
    self.max_length = max_length
    self.truncation = truncation
    self.padding = padding

    first_token = self.__cleaning(first_token)

    second_token = self.__cleaning(second_token)

    mean_pooled_arr = self.__process(first_token, second_token)

    if return_as_embeddings:
      return mean_pooled_arr
    
    similarity = cosine_similarity([mean_pooled_arr[0]], [mean_pooled_arr[1]])

    return similarity

st.title('CEK HOAKS')

model = TokenSimilarity()
model.load_pretrained('indobenchmark/indobert-base-p2')

df = pd.read_excel('hoax.xlsx')

# def clear_submit():
#   st.session_state['submit'] = False

to_check = st.text_area('Teks yang mau dicek...')


if to_check:
  
  for i in np.arange(len(df['text'])):
    result = model.predict(to_check, df['text'][i])
    st.write(result)