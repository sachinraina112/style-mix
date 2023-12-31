
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

from functions import run_style_transfer
from display_im import showImage
import uuid
from flask import Flask, jsonify
import flask


import os
import json
import sys
import signal
import traceback
from s3_url import get_url
import requests
from io import BytesIO




#app for web app
app = Flask(__name__)

def display(im, root=None, title=None):
    im_obj = showImage()
    im_obj.im_show(im, root, title)


def save_from_pil(im):
    imsize = (512,512) if torch.cuda.is_available() else (128, 128)
    loader = transforms.Compose([
            transforms.Resize(imsize),  # scale imported image
            transforms.ToTensor()])
    image = loader(im).unsqueeze(0)
    print(f" {list(image.size())}")
    ten = image.to("cpu", torch.float)
    print(type(ten))
    return ten


def tensor_from_link(url):
    response = requests.get(url)
    im = Image.open(BytesIO(response.content)).convert("RGB")
    tensor = save_from_pil(im)
    return tensor





def convert_tensor_to_response(inp_tensor):
    tensor_size = list(inp_tensor.size())
    flattened = torch.flatten(inp_tensor)
    fla_list = flattened.tolist()
    tensor_dict = {"value_list": fla_list, "size": tensor_size}
    return tensor_dict

def convert_to_tensor(dicts):
    from numpy import array
    lla = array(dicts["value_list"])
    output_tensor = torch.from_numpy(lla)
    output_tensor.resize_(dicts["size"])
    return output_tensor



def get_styled_image(pre_trained_model, mean_tn, std_tn, con_im, sty_im,
                        input_img, cl, sl, root):
    random = str(uuid.uuid4())
    im_obj = showImage()
    # cont_tensor, _ = im_obj.image_loader(con_im)
    # sty_tensor, _ = im_obj.image_loader(sty_im)

    if im_obj.check_size(con_im, sty_im):
        
        inp_path = "../data/input/"
        display(con_im, inp_path,title="Content")
        display(sty_im, inp_path,title="Style")
        display(input_img, inp_path,title="Input")
       
        output = run_style_transfer(pre_trained_model, mean_tn, std_tn,
                                con_im, sty_im, input_img,cl, sl)

        title = f'Output_{random}'
        display(output, root, title=title)
        path_to_output = root + title + ".jpg" 
        print(path_to_output)         
        url = get_url(path_to_output)
        user_list = [path_to_output, url]
        print(user_list)
        return output, user_list
    else:
        raise Exception




# prefix = '/ml/'
prefix = '../data/'
model_path = '../models'

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class StyleServiceML(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls, path=None):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if not path:
            if cls.model == None:
                with open(os.path.join(model_path, 'tensor.pt'), 'rb') as inp:
                    cls.model = torch.load(inp)
        else:
            if cls.model == None:
                with open(os.path.join(path, 'tensor.pt'), 'rb') as inp:
                    cls.model = torch.load(inp)
        return cls.model

    @classmethod
    def image_blend(cls, res_data):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        try:
            cnn = cls.get_model()
        except:
            path=res_data["mod_path"]
            print(path)
            cnn = cls.get_model(path=path)
        print(res_data)
        prefix = res_data["blob_path"]  
        base_out_path = prefix + "output/"
        path_to_images = prefix + "input/"

        cont_url =  res_data["cont_name"]
        content_img = tensor_from_link(cont_url)
        sty_url =  res_data["sty_name"]
        if sty_url:
            style_img = tensor_from_link(sty_url)
        else:
            sty_path = "../data/input/wave.jpg"
            ss = showImage()
            style_img, _ = ss.image_loader(sty_path)
        inp_url =  res_data["input_name"]
        if inp_url:
            input_img = tensor_from_link(inp_url)
        else:
            input_img = content_img
        
        # content_img = path_to_images + res_data["cont_name"]
        # style_img = path_to_images + res_data["sty_name"]
        print(type(content_img), type(style_img))
        print(content_img.size(), style_img.size())
        # blend_type = res_data["type"] # only style, style+content, 
        blend_intensity = res_data["intensity"] # style intensity-low/high
        content_layers_default = ['conv_4']
        if  blend_intensity == "low":
            style_layers_default = ['conv_1', 'conv_2', 'conv_3'] #,'conv_4', 'conv_5']
        elif blend_intensity == "high":
            style_layers_default = ['conv_1', 'conv_2', 'conv_3','conv_4', 'conv_5']
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        output, user_list = get_styled_image(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, content_layers_default, style_layers_default, base_out_path)
        if not res_data["tensor"]:
            out = []
        else:
            out = convert_tensor_to_response(output)
        return out, user_list



@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load given model successfully."""
    health = StyleServiceML.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/blend', methods=['POST'])
def blend():
    if flask.request.content_type == 'application/json':
        res_data = flask.request.json
        print(res_data)
        from datetime import datetime
        start = datetime.now()
        output, path = StyleServiceML.image_blend(res_data["data"])
        # result = (out, path)
        end = datetime.now()
        time = end-start
        print(time)

        result = {"time_taken": str(time),"path_to_output": path, "output_tensor": output}
        # print(result)
        result = jsonify(result)
        return result, 200
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='application/json')







if __name__ == "__main__":
    from datetime import datetime
    start = datetime.now()
    base_path = "../data/output/"
    content_img = "../data/input/sketch.png"
    style_img = "../data/input/rain_princess.jpg"
    mod_path = "../models/tensor.pt"
    input_img = content_img
    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3'] #,'conv_4', 'conv_5']
    # cnn = models.vgg19(pretrained=True).features.eval()
    cnn = torch.load(mod_path)
# python style-transfer.py
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    output, path = get_styled_image(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, content_layers_default, style_layers_default, base_path)
    end = datetime.now()
    
    print(output)
    print(f"Time taken to get styled image is {end-start}")














# if you want to use white noise by using the following code:
#
# ::
#
#    input_img = torch.randn(content_img.data.size())

# add the original input image to the figure:





