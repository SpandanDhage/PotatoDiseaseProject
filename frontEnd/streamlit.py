
import streamlit as st
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json


endpoint="http://localhost:8000"

st.set_page_config(layout="wide", page_title="DLApp")
st.title("DL App")

uploaded_image = st.file_uploader("Upload Image Here")

if uploaded_image is not None :
    # try:
        m=MultipartEncoder(
            fields={'file': ('filename', uploaded_image, 'image/jpeg')}
        )
        response=requests.post(f"{endpoint}/predict", data=m, headers={'Content-Type': m.content_type}, timeout=8000)

        if response.status_code == 200:
            cola,colb,colc=st.columns(3)
            with colb:
                st.image(uploaded_image)
                colb.markdown(f"Predicted as : {(json.loads(response.text))['Class']}")
                colb.markdown(f"Confidence : {round(((json.loads(response.text))['Confidence']),4)}")
        else:
            st.error(f"Error while loading Image... \n Error Details: {(json.loads(response.text))['details']}")
    # except:
    #     st.error(f"Error in connecting server...")