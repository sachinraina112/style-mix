import os

import yaml
_fpath = os.path.join(os.path.dirname(__file__), "config.yml")

with open(_fpath) as fp:
    config = yaml.safe_load(fp)
    s3_config = config['s3_config']
    path_config = config['path_config']
    url_config = config['url_config']

    
