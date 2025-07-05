# 🎭 脫口秀笑點預測模型 v4cl3

這是一個基於 BERT 的中文多分類模型，用來預測脫口秀段子文字稿的「笑點分數」，範圍為 **1~100 分**。

---

## 📦 專案說明

- **模型架構**：使用 `bert-base-chinese` 搭建的分類模型（num_labels=100）。
- **任務類型**：多分類（把 1~100 分數映射成 100 個類別）。
- **資料來源**：每筆資料含「逐字稿」與「對應的笑點分數」。
- **訓練標籤轉換**：原始分數 1~100 → 分類標籤 0~99。

---

## 📁 資料格式（JSON）

```json
{
  "影片名稱": {
    "整部影片的逐字稿": "我寫了一段沒有人會生氣的笑話，就是講台海戰爭。",
    "分數": 87
  }
}
```

---

## 🚀 預測用法

### ✅ 安裝環境

```bash
pip install torch transformers
```

### ✅ 預測程式 `predict.py`

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = r"C:\Users\user\OneDrive\桌面\專題企畫\脫口秀model_classification"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def predict_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_label = torch.argmax(outputs.logits, dim=1).item()
    return pred_label + 1  # 轉回 1~100 分

if __name__ == "__main__":
    text = "我今年找到了一個不會讓人不爽的方法，就是講沒人關心的主題，比如台海戰爭。"
    score = predict_score(text)
    print(f"預測笑點分數：{score}")
```

---

## 💾 訓練模型儲存格式

訓練後模型會儲存在指定資料夾，例如：

```
脫口秀model_classification/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
├── training_args.bin
```

---

## 📊 建議訓練參數

- 模型：`bert-base-chinese`
- 批次大小：4
- Epochs：5
- 學習率：2e-5
- 優化目標：分類精度（Accuracy）

---

## 🔧 效能提示

- 請優先使用 GPU 執行推論
- 若輸入句子過長，可考慮斷句後平均預測
- 若結果偏低，可考慮增加訓練資料、多跑幾輪、或針對高分樣本加權處理