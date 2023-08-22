

from display_im import showImage
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO



import torch 




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


def save_from_link(url):
    response = requests.get(url)
    im = Image.open(BytesIO(response.content)).convert("RGB")
    tensor = save_from_pil(im)
    return tensor







link = "https://stybucket.s3.amazonaws.com//data/output/Output_deb62024-6410-46d3-bc9a-1333123b1dc4.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZEKZNBJKTAGEBPWK%2F20230822%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20230822T110333Z&X-Amz-Expires=6000&X-Amz-SignedHeaders=host&X-Amz-Signature=08768f7867cf723387c5923118dcb1faca80b0a4421dcbc8cfc7e0c9e4ef8df8"

response = requests.get(link)
# print(response)


image_tensor = save_from_link(link)

# ss = showImage()
# ss.im_show(image_tensor)
    

