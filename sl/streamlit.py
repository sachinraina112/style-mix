import streamlit as st
import io
from PIL import Image
from configs.AppConfig import path_config
import uuid
import os
from result import get_result
from s3_url import get_url


input_path = os.path.join(path_config["data"], "input/")
def get_ext(name):
    name = str(name)
    ext = "." + str(name.split(".")[-1])
    return ext


first = st.container()
second = st.container()
third = st.container()
fourth = st.container()
fifth = st.container()
sixth = st.container() 

with first:
    st.title("Deep Style Blend Mix Project")
    st.text("Use Deep Learning Model to generate a blend image from your input content image")

with second:
    st.header("For Inserting Name of Image")
    output_description = st.text_input("Add Input text")
    
with third:
    st.header("Adjust Style Params")
    var_col, disp_col = st.columns(2)
    intensity = var_col.selectbox("Select style intensity", options=['low','high'], index=0)
    blend_type = var_col.selectbox("Select if content and input images are same or not for style intensity", options=['only-style','style-content','new-blend'], index=0)
    # 

if blend_type == "only-style":
    with fourth:
        st.header("Upload Image")
        Uploadimage = st.file_uploader('Choose the content/input image to upload')
        if Uploadimage is not None:
            # To read file as bytes:
            name = Uploadimage.name
            ext = get_ext(name)
            
            imagefile = io.BytesIO(Uploadimage.read())
            im = Image.open(imagefile)
            # Do something with im
            st.image(im)
            random = str(uuid.uuid4()) + ext
            ims = Image.open(imagefile).convert('RGB')
            save_name_input = input_path + random
            ims.save(save_name_input)
            url_1 = get_url(save_name_input)
            random_list = [random, "wave.jpg", random]
            url_list = [url_1, False, False] 
            result_link = get_result(random_list, blend_type, intensity, url_list)
            st.header("Output Image")
            st.markdown(f"**{output_description}**")
            st.image(result_link)

            


if blend_type == "style-content":
    with fifth:
        st.header("Upload Content and Style Image")
        st.subheader("Upload Content Image")
        Uploadimage_content = st.file_uploader('Choose the content image to upload')
        if Uploadimage_content is not None:
            # To read file as bytes:
            imagefile = io.BytesIO(Uploadimage_content.read())

            im = Image.open(imagefile)
            name = Uploadimage_content.name
            print(f"name of content {name}")
            ext = get_ext(name)
            # Do something with im
            st.image(im)
            random = str(uuid.uuid4()) + ext
            ims = Image.open(imagefile).convert('RGB')
            save_name_input = input_path + random
            ims.save(save_name_input)
            url_1 = get_url(save_name_input)
            

        st.subheader("Upload Style Image")
        Uploadimage_style = st.file_uploader('Choose the style image to upload')
        if Uploadimage_style is not None:
            # To read file as bytes:
            imagefile2 = io.BytesIO(Uploadimage_style.read())

            im2 = Image.open(imagefile2)
            
            name2 = Uploadimage_style.name
            print(f"name of style {name2}")
            ext2 = get_ext(name2)
            # Do something with im
            st.image(im2)
            random2 = str(uuid.uuid4()) + ext2
            ims2 = Image.open(imagefile2).convert('RGB')
            save_name_input = input_path + random2
            ims2.save(save_name_input)
            url_2 = get_url(save_name_input)

        if Uploadimage_content and Uploadimage_style:
            random_list = [random, random2 , random] 
            url_list = [url_1, url_2, False]
            result_link = get_result(random_list, blend_type, intensity, url_list)
            st.header("Output Image")
            st.markdown(f"**{output_description}**")
            st.image(result_link)


if blend_type == "new-blend":
    with sixth:
        st.header("Upload Content, Style and Input Image")
        st.subheader("Upload Content Image")
        Uploadimage_content = st.file_uploader('Choose the content image to upload')
        if Uploadimage_content is not None:
            # To read file as bytes:
            imagefile = io.BytesIO(Uploadimage_content.read())


            im = Image.open(imagefile)
            name = Uploadimage_content.name
            ext = get_ext(name)
            # Do something with im
            st.image(im)
            random = str(uuid.uuid4()) + ext
            ims = Image.open(imagefile).convert('RGB')
            save_name_input = input_path + random
            ims.save(save_name_input)
            url_1 = get_url(save_name_input)


        st.subheader("Upload Style Image")
        Uploadimage_style = st.file_uploader('Choose the style image to upload')
        if Uploadimage_style is not None:
            # To read file as bytes:
            imagefile2 = io.BytesIO(Uploadimage_style.read())

            im2 = Image.open(imagefile2)
            name2 = Uploadimage_style.name
            ext2 = get_ext(name2)
            # Do something with im
            st.image(im2)
            random2 = str(uuid.uuid4()) + ext2
            ims2 = Image.open(imagefile2).convert('RGB')
            save_name_input = input_path + random2
            ims2.save(save_name_input)
            url_2 = get_url(save_name_input)

        st.subheader("Upload Input Image to be changed")
        Uploadimage_inp = st.file_uploader('Choose the input image to upload')
        if Uploadimage_inp is not None:
            # To read file as bytes:
            imagefile3 = io.BytesIO(Uploadimage_inp.read())

            im3 = Image.open(imagefile3)
            name3 = Uploadimage_inp.name
            ext3 = get_ext(name3)
            # Do something with im
            st.image(im3)
            random3 = str(uuid.uuid4()) + ext3
            ims3 = Image.open(imagefile3).convert('RGB')
            save_name_input = input_path + random3
            ims3.save(save_name_input)
            url_3 = get_url(save_name_input)

        if Uploadimage_content and Uploadimage_style and Uploadimage_inp:
            random_list = [random, random2 , random3] 
            url_list = [url_1, url_2, url_3]
            result_link = get_result(random_list, blend_type, intensity, url_list)
            st.header("Output Image")
            st.markdown(f"**{output_description}**")
            st.image(result_link)


    