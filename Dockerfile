FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# XGBoost 執行時需要 OpenMP。
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt ./
RUN python -m pip install --requirement requirements-docker.txt

COPY predict_nonsnp.py predict_snp.py prs.txt ./
COPY nonsnp_example.csv snp_example.csv ./
COPY model/ ./model/

# 預設顯示無 SNP 程式說明；執行時可將第一個參數換成任一預測程式。
ENTRYPOINT ["python"]
CMD ["predict_nonsnp.py", "--help"]
