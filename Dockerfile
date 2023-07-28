FROM tensorflow/tensorflow:2.9.1

ENV HOME=/app
ENV DATA_DIR='/nlp_data'
ENV CUDA_VISIBLE_DEVICES=1
ENV NLTK_DATA=/app/nltk_data

COPY . ${HOME}

RUN set -eux; \
    python --version
RUN set -eux; \
    python -m pip install -U pip

RUN pip install joblib~=1.1.0
RUN pip install sklearn~=0.0
RUN pip install scikit-learn~=1.1.1
RUN pip install nltk~=3.7
RUN pip install PyYAML~=6.0

RUN set -eux; \
    python -m nltk.downloader stopwords

WORKDIR ${HOME}

ENTRYPOINT [ "python", "main.py" ]
