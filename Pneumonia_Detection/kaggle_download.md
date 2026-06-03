Downloading the Kaggle Chest X-Ray Images dataset

1. Install and configure the Kaggle CLI:

```bash
pip install kaggle
# place your kaggle.json under ~/.kaggle/kaggle.json (chmod 600)
```

2. Download the dataset (from the dataset slug):

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d dataset/chest_xray
```

3. Verify the folder layout matches README.md expectations.

If you cannot use the Kaggle CLI, download manually from the dataset page and extract into `dataset/chest_xray`.
Downloading the Kaggle Chest X-Ray Images dataset

1. Install and configure the Kaggle CLI:

```bash
pip install kaggle
# place your kaggle.json under ~/.kaggle/kaggle.json (chmod 600)
```

2. Download the dataset (from the competition/dataset slug):

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/chest_xray
```

3. Verify the folder layout matches README.md expectations.

If you cannot use the Kaggle CLI, download manually from the dataset page and extract into `data/chest_xray`.
