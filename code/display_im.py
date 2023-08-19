import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms


class showImage:
    def __init__(self):
        self.imsize = (512,512) if torch.cuda.is_available() else (128, 128)
        self.loader = transforms.Compose([
                        transforms.Resize(self.imsize),  # scale imported image
                        transforms.ToTensor()])  # transform it into a torch tensor
        self.T = transforms.ToTensor()
        self.unloader = transforms.ToPILImage()

    def get_directory(self, path):
        rt = path.split("/")[:-1]
        root = "/".join(rt) + "/"
        return root

    def check_size(self,a,b):
        a_size = a.size()
        b_size = b.size()
        print(f"Size: {a_size} {b_size}")
        if a_size == b_size:
            return True
        else:
            print("we need to import style and content images of the same size")
            return False

    def image_loader(self, path):
        root = self.get_directory(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
        image = Image.open(path).convert('RGB')
        test = self.T(image)
        print(f"{path} {list(test.size())}")
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        print(f"{path} {list(image.size())}")
        return image.to(device, torch.float), root

    def im_show(self, tensor, root=None, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = self.unloader(image)
        plt.figure()
        plt.imshow(image)
        plt.ioff()
        plt.show() 
        t = title if title else 'x'
        print(t)
        if root:
            p = root + f"{t}.jpg"
        else:
            p = t
        plt.savefig(p)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated



if __name__ == "__main__":
    
    picasso = "../data/test/picasso.jpg"
    dancing = "../data/test/dancing.jpg"


    im_obj = showImage()
    tensor, root = im_obj.image_loader(picasso)
    im_obj.im_show(tensor, root,"picasso-pic")
    
    
