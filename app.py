# Core packages
from PIL import Image
import numpy as np 
import io
import base64
import h5py
import json
import pymysql.cursors
from datetime import datetime
import time
import tempfile
import os
import matplotlib.pyplot as plt
import cv2

import requests
import streamlit as st

# FastAPI server endpoint
fastapi_server = "http://192.168.0.17:8000/"  

# Endpoint url refer FastAPI docs
url_getinfo = fastapi_server+"info"
url_uploadimage = fastapi_server+"load"
url_uploadmodel = fastapi_server+"load_model"
url_gradcam = fastapi_server+"gradcam"
url_crop = fastapi_server+"rectangle_crop"

# ----- Function -----
# controllers.py
def upload_file(url, file):  
    files = {'file': (file.name, file.getvalue())}
    response = requests.post(url, files=files)
    response_json = json.loads(response.text)
    return response_json

def compute_gradcam(url, cnn_layer, image_file, model_file):
    
    multiple_files = {'image_input_file': (image_file.name, image_file.getvalue()),
                      'model_input_file': (model_file.name, model_file.getvalue())}
    data = {'selected_cnn_layer': cnn_layer}
    
    response = requests.post(url, files=multiple_files, data=data)
    response_json = json.loads(response.text)
    return response_json

def base64toimg(img_base64):
    # print("img_base64", img_base64)
    image_bytes = base64.b64decode(img_base64.encode('utf-8'))
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    return image

# ----- Main Program -----
def main():
    
    # page config
    st.set_page_config(page_title='Gradcam Visualization WebApp', 
                       layout="wide", 
                       initial_sidebar_state="collapsed", 
                       page_icon="icon.png")

    panels = ["Grad-CAM Visualization", "About"]
    selection = st.sidebar.selectbox("Menu", panels)
    
    # ----- Main page -----
    if selection == "Grad-CAM Visualization":
        st.title("Grad-CAM Visualization")
        st.text("To visualize where AI model is focusing.")
        
        # ----- Load model -----
        model_file = st.file_uploader("Upload .h5 Model", type=".h5")

        # upload model as file
        if model_file is not None:            
            # if st.button("Check model"):
            try:    
                response = upload_file(url=url_uploadmodel, file=model_file)
                
                st.success("Load model success.")
                load_model_status = "Success"
                
            except Exception as ex:
                print(str(ex))
                st.error("Load model failed. Please re-check your file path!")
                load_model_status = "Fail"
        
        
            if load_model_status == "Success":
                # ----- Get model layer -----
                input_layer_name = response["input_layer"]
                cnn_layer_names = response["cnn_layers"]
                input_shape = response["input_shape"]                      

                # Remide to input image size
                st.info("Please input image with size (Hight, Width, Dimension) as "+ str(input_shape) +"  ")

                # ---- Load image -----
                image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

                if image_file is not None:
                        
                    # Select only CNN layer
                    selected_layer = st.selectbox("Select a CNN Layer", cnn_layer_names)
                    st.write(selected_layer)
                    
                    if selected_layer is not None:

                        st.write("call gradcam")
                        response = compute_gradcam( url=f"{url_gradcam}/{selected_layer}", 
                                                    cnn_layer=selected_layer, 
                                                    image_file=image_file, 
                                                    model_file=model_file)
                        if response["status"] == "Success":
                            img_input = base64toimg(response["img_input_base64"])
                            img_gradcam_result = base64toimg(response["img_result_base64"])
                            
                            # Display the original image and Grad-CAM map
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
                            if len(img_input.shape)==3 and img_input.shape[-1]==3:
                                ax1.imshow(img_input)
                            else:
                                ax1.imshow(img_input, cmap = "gray")
                            ax1.set_title("Original Image: " + image_file.name)
                            ax2.imshow(img_gradcam_result, cmap='jet', alpha=0.5)
                            ax2.set_title("Grad-CAM")

                            st.pyplot(fig)
                                
                        else:
                            st.error(response["status"])

# ----- App Run -----
if __name__ == '__main__':
    # Call the main function
    main()