import requests
from configs.AppConfig import url_config

ip = str(url_config["ip"])
port = str(url_config["port"])
url = "http://" + ip + ":" + port + "/blend"
print(f"Submitting request to url: {url}")

def get_result(random, type, intensity, urls):
    payload = {
	"data": {
		"blob_path": "../data/",
		"mod_path": "../models",
		"cont_name": "save1.png",
		"sty_name": "wave.jpg",
		"input_name": "sketch.png",
		"type": "only-style",
		"intensity": "low",
		"tensor": False
	    }
    }
    random = [str(i) for i in random]
    dictionary_data = payload["data"]
    dictionary_data.update({"cont_name": urls[0],"sty_name":urls[1],
                            "input_name": urls[2], "type": type,
                            "intensity": intensity})
    new_payload = {"data": dictionary_data}

    
    print(f"new payload {new_payload}")
    result = requests.post(url, json=new_payload)
    res = result.json()
    # res = res.decode("utf-8")
    print(res["path_to_output"])
    if result.status_code == 200:
        save_link_output = res["path_to_output"][1]
        return save_link_output
    else:
        raise Exception(result.status_code)