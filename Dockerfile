FROM python:3.9

COPY code/ /style-content/code
COPY data/ /style-content/data
WORKDIR /style-content
COPY requirements.txt /style-content/requirements.txt

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r /style-content/requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

EXPOSE 9999 7000
WORKDIR "/style-content/code/"

CMD ["/bin/sh", "serve.sh"]
