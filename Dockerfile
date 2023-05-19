FROM continuumio/miniconda3

#Set ENV variable below to URL of selected MONAI BUNDLE -- spleen_ct_segmentation bundle used for example
#Requirements:
#-must have torchscript supported (model.ts)
#-must conform to MONAI bundle spec 
#-must be a segmentation model
#-preprocessing / transforms will may need to be added prior to inference operator in seg_app
ENV MONAI_BUNDLE_URL="https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/spleen_ct_segmentation_v0.3.8.zip"

RUN mkdir -m777 /app
WORKDIR /app
ADD docker-entrypoint.sh ./
ADD seg_app ./seg_app

RUN chmod 777 ./docker-entrypoint.sh
RUN conda create -n env python=3.7
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
RUN chmod -R 777 /opt/conda/envs

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y git build-essential cmake pigz
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y libsm6 libxrender-dev libxext6 ffmpeg
RUN apt-get install unzip

ADD environment.yml ./
RUN conda env create -f ./environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 ./environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 ./environment.yml | cut -d' ' -f2)/bin:$PATH

# Download MONAI bundle and extract model.ts file
RUN mkdir -m777 ./zip_tmp
RUN wget --directory-prefix ./zip_tmp ${MONAI_BUNDLE_URL} \ 
    && unzip ./zip_tmp/*.zip  -d ./zip_tmp/ \
    && cp ./zip_tmp/*/models/model.ts ./ && rm -r ./zip_tmp
RUN chmod 777 ./model.ts

CMD ["./docker-entrypoint.sh"]
