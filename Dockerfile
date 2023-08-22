FROM python:3.9

RUN mkdir style-mix
WORKDIR /style-mix
ADD data/ /style-mix/data/
ADD code/ /style-mix/code/
# ADD models/ /style-mix/models/
COPY requirements_st.txt /style-mix/requirements_st.txt

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r /style-mix/requirements_st.txt

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

EXPOSE 9999 7000
WORKDIR "/style-mix/code/"

CMD ["/bin/sh", "serve.sh"]
