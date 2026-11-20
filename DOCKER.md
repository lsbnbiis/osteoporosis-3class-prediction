# 骨質狀態預測 Docker

此映像包含：

- `predict_nonsnp.py`：無 SNP ensemble 預測
- `predict_snp.py`：PRS 計算與 SNP SVM 預測
- `prs.txt`
- `model/` 中兩套推論所需的 scaler 與模型
- 與模型相容的 Python 3.12 套件版本

## 建立映像

在本專案目錄執行：

```bash
docker build -t osteoporosis-predictor:latest .
```

## 執行無 SNP 預測

將目前目錄掛載到容器的 `/data`，輸出會保留在主機：

```bash
docker run --rm \
  -v "$PWD:/data" \
  osteoporosis-predictor:latest \
  predict_nonsnp.py \
  --input /data/nonsnp_example.csv \
  --output /data/nonsnp_docker_predictions.csv
```

## 執行 SNP／PRS 預測

映像內預設使用 `/app/prs.txt` 與 `/app/model`：

```bash
docker run --rm \
  -v "$PWD:/data" \
  osteoporosis-predictor:latest \
  predict_snp.py \
  --input /data/snp_example.csv \
  --output /data/snp_docker_predictions.csv
```

若要改用主機上的 PRS 權重檔：

```bash
docker run --rm \
  -v "$PWD:/data" \
  osteoporosis-predictor:latest \
  predict_snp.py \
  --input /data/snp_example.csv \
  --output /data/snp_docker_predictions.csv \
  --prs /data/prs.txt
```

## 查看參數

```bash
docker run --rm osteoporosis-predictor:latest predict_nonsnp.py --help
docker run --rm osteoporosis-predictor:latest predict_snp.py --help
```

輸出的 `PREDICTION` 類別：

- `0`：正常
- `1`：骨鬆
- `2`：輕微骨鬆
