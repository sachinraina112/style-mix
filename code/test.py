import torch
import numpy




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



if __name__ == "__main__":
    a = torch.randn(2, 2)
    print(a, a.size())
    res = convert_tensor_to_response(a)
    print(res)
    initial = convert_to_tensor(res)
    print(initial, initial.size())