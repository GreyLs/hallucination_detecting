FROM python:3.10-slim
WORKDIR /app

# Копируем все файлы проекта в директорию /app
COPY . /app

VOLUME /app/data

ENV PATH=$PATH:/usr/lib/R/bin
ENV R_HOME="/usr/lib/R"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libgirepository1.0-dev \
    libpango1.0-dev \
    libpangocairo-1.0-0 \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libx11-dev \
    r-base \
    r-base-dev \
    p7zip-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Установка Python-зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir transformers && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf ~/.cache/pip

# Создание необходимых директорий
RUN mkdir -p /app/saved_models/best_model_Savvy_24 && \
    mkdir -p /app/model && \
    mkdir -p /app/tokenizer

# Загрузка модели и токенизатора
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
               model_name = 'DeepPavlov/rubert-base-cased'; \
               tokenizer = AutoTokenizer.from_pretrained(model_name); \
               model = AutoModelForSequenceClassification.from_pretrained(model_name); \
               tokenizer.save_pretrained('/app/tokenizer'); \
               model.save_pretrained('/app/saved_models/model_S')"

# Сохранение модели и токенизатора в pickle-файлы
RUN python -c "import pickle; from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
               tokenizer = AutoTokenizer.from_pretrained('/app/tokenizer'); \
               model = AutoModelForSequenceClassification.from_pretrained('/app/saved_models/model_S'); \
               pickle.dump(tokenizer, open('/app/tokenizer.pkl', 'wb')); \
               pickle.dump(model, open('/app/model.pkl', 'wb'))"

# Копируем оставшиеся файлы проекта
COPY make_prediction.py /app/make_prediction.py
COPY utils.py /app/utils.py

CMD ["python3", "/app/make_prediction.py"]
