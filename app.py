# Core packages
import numpy as np 
import base64
import json
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import os
import io

# ----- Web API -----
import requests
import streamlit as st

# ----- Load Secret -----
from dotenv import load_dotenv
load_dotenv()

# FastAPI server endpoint
# fastapi_server = "http://192.168.0.17:8000/"
# fastapi_server = "https://fitelsmart-gradcam-api.azurewebsites.net/"
fastapi_server = os.getenv("GRADCAM_API_URL")

# Endpoint url refer FastAPI docs
url_getinfo = fastapi_server+"info"
url_uploadimage = fastapi_server+"load"
url_uploadmodel = fastapi_server+"load_model"
url_gradcam = fastapi_server+"gradcam"
url_crop = fastapi_server+"rectangle_crop"

# ----- Function -----
# controllers.py
def upload_file(url, file, demo:bool):
    if demo:
        with open(file, 'rb') as file_read:
            files = {'file': (file_read.name, file_read.read())}    
    else:       
        files = {'file': (file.name, file.read())}
    response = requests.post(url, files=files)
    response_json = json.loads(response.text)
    return response_json

def upload_file_from_demo(url, file, filebyteio):
    files = {'file': (file.name, filebyteio.getvalue())}   
    response = requests.post(url, files=files)
    response_json = json.loads(response.text)
    return response_json

def compute_gradcam(url, cnn_layer, image_file, model_file, demo:bool):
    if demo:
        multiple_files = {'image_input_file': None,
                          'model_input_file': None}
        with open(image_file, 'rb') as image:
            multiple_files["image_input_file"] = (image.name, image.read())
        with open(model_file, 'rb') as model:
            multiple_files["model_input_file"] = (model.name, model.read())
    else:   
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
        
        # Demo
        demo_enable = st.checkbox('Get Model for DEMO')

        if demo_enable:
            directory = 'demo/'  # Replace with the actual directory path
            extension = '.h5'
            file_list = [file_name for file_name in os.listdir(directory) if file_name.endswith(extension)]
            selected_model = st.selectbox(  label='Select demo .h5 model', 
                                            options=file_list)

            file_path = directory+selected_model
            model_file = file_path
            
        else:
            # ----- Load model -----
            model_file = st.file_uploader("Upload .h5 Model", type=".h5")

        # upload model as file
        if model_file is not None:            
            # if st.button("Check model"):
            try:

                response = upload_file(url=url_uploadmodel, file=model_file, demo=demo_enable)
                
                st.success("Load model success.")
                load_model_status = "Success"
                
            except Exception as ex:
                print(str(ex))
                st.error("Load model failed. Please re-check your file path!\n" + "Error: "+str(ex))
                load_model_status = "Fail"
        
        
            if load_model_status == "Success":
                # ----- Get model layer -----
                input_layer_name = response["input_layer"]
                cnn_layer_names = response["cnn_layers"]
                input_shape = response["input_shape"]                      

                # Remide to input image size
                st.info("Please input image with size (Hight, Width, Dimension) as "+ str(input_shape) +"  ")


                if demo_enable:
                    img_dir = f'demo/{selected_model[:-3]}/'  # Replace with the actual directory path
                    img_ext = ("jpg", "jpeg", "png", "bmp")
                    file_list = [file_name for file_name in os.listdir(img_dir) if file_name.endswith(img_ext)]
                    selected_image = st.selectbox(  label=f'Select image demo for {selected_model} model', 
                                                    options=file_list)

                    image_file_path = img_dir+selected_image
                    image_file = image_file_path
                    
                else:    
                    # ---- Load image -----
                    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])

                if image_file is not None:
                        
                    # Select only CNN layer
                    selected_layer = st.selectbox("Select a CNN Layer", cnn_layer_names)
                    
                    if selected_layer is not None:
                        with st.spinner('GradCam is processing ...'):
                        
                            response = compute_gradcam( url=f"{url_gradcam}/{selected_layer}", 
                                                        cnn_layer=selected_layer, 
                                                        image_file=image_file, 
                                                        model_file=model_file,
                                                        demo=demo_enable)
                            if response["status"] == "Success":
                                img_input = base64toimg(response["img_input_base64"])
                                img_gradcam_result = base64toimg(response["img_result_base64"])
                                
                                # Display the original image and Grad-CAM map
                                plt.figure(figsize=(19, 5))
                                plt.subplot(1, 2, 1)
                                if len(img_input.shape)==3 and img_input.shape[-1]==3:
                                    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                                    plt.imshow(img_input)
                                else:
                                    plt.imshow(img_input, cmap = "gray")
                                plt.colorbar()
                                if demo_enable:
                                    plt.title("Original Image: " + selected_image)
                                else:
                                    plt.title("Original Image: " + image_file.name)
                                    
                                plt.subplot(1, 2, 2)
                                plt.imshow(img_gradcam_result, cmap='jet', alpha=0.5)
                                plt.colorbar()
                                plt.title("Grad-CAM")

                                st.pyplot(plt)
                                st.balloons()
                            else:
                                st.error(response["status"])
                                


    if selection == "About":
        # documentation
        doc_url = "http://192.168.0.17:8000/Jakkapat-dew/RDAPI000-GradCam/codeExample/"
        st.info("Documentation :books: : [https://fitelsmart-gradcam.azurewebsites.net/docs](%s) " % doc_url)
        api_doc_url = "https://fitelsmart-gradcam-api.azurewebsites.net/docs"
        st.info("API Reference :books: : [https://fitelsmart-gradcam-api.azurewebsites.net/docs](%s)" % api_doc_url)
    
# ----- App Run -----
if __name__ == '__main__':
    # Call the main function
    main()