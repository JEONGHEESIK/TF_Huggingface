"""dataset"""
from huggingface_hub import list_datasets

all_datasets = [ds.id for ds in list_datasets()]
# print(f"현재 허브에는 {len(all_datasets)}개의 데이터셋이 있음.")
# print(f"처음 10개 데이터셋: {all_datasets[:10]}")

from datasets import load_dataset

emotions = load_dataset("emotion")

# print(emotions)

train_ds = emotions["train"]
# print(train_ds)

# print(len(train_ds))

# print(train_ds[0])

# print(train_ds.column_names)

# print(train_ds.features)

# print(train_ds[:5])

# print(train_ds["text"][:5])


"""dtset 2 dtframe"""
import pandas as pd

emotions.set_format(type="pandas")
df = emotions["train"][:]

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)


"""클래스 분포"""
import matplotlib.pyplot as plt

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
# plt.show()

"""트윗 길이 확인"""
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name",grid=False,showfliers=False,color="black")

plt.suptitle("")
plt.xlabel("")
# plt.show()

emotions.reset_format()


"""문자 토큰화"""
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
# print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
# print(token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
# print(input_ids)

categorical_df = pd.DataFrame({"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]})
# print(categorical_df)

# print(pd.get_dummies(categorical_df["Name"]))

import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
# print(one_hot_encodings.shape)

# print(f"Token : {tokenized_text[0]}")
# print(f"Tensor Index : {input_ids[0]}")
# print(f"One-hot Encoding: {one_hot_encodings[0]}")


"""단어 토큰화"""
tokenized_text = text.split()
# print(tokenized_text)

"""부분단어 토큰화"""
# WordPiece, BERT-DistillBERT
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# DistilBERT Tokenizer
from transformers import DistilBertTokenizer

distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text)
# print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
# print(tokens)
# ['[CLS]', 'token', '##izing', 'text', 'is', 'a', 'core', 'task', 'of', 'nl', '##p', '.', '[SEP]']

# print(tokenizer.convert_tokens_to_string(tokens))
# [CLS] tokenizing text is a core task of nlp. [SEP]

# 어휘 사전 크기
# print(tokenizer.vocab_size)
# 30522

# 모델 최대 문맥 크기
# print(tokenizer.model_max_length)
# 512

# print(tokenizer.model_input_names)


"""전체 데이터셋 토큰화"""
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# print(tokenize(emotions["train"][:2]))
 
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None) # 전체 데이터셋,하나의 배치
# print(emotions_encoded["train"].column_names)

"""텍스트 분류 모델 훈련"""

"""트랜스포머 특성 추출기"""
# 사전 훈련된 모델
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# 마지막 은닉 상태 추출
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
# print(f"입력 텐서 크기: {inputs['input_ids'].size()}") # [batch_size, n_tokens]

inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
# print(outputs)
# 결과에 device='cuda:0' 가 나오지 않으면 cuda 버전 확인 후 재설치 바람

# print(outputs.last_hidden_state.size()) # [batch_size, n_tokens, hidden_dim]

# print(outputs.last_hidden_state[:,0].size())

def extract_hidden_states(batch):
    # 모델 입력을 GPU로 옮김
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    # 마지막 은닉 상태 추출
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # [CLS] 토큰에 대한 벡터 반환
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

emotions_encoded.set_format("torch", columns=["input_ids","attention_mask","label"])

emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# print(emotions_hidden["train"].column_names)

# 특성 행렬 만들기
import numpy as np

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
# print(X_train.shape, X_valid.shape)

# 훈련 세트 시각화
import umap
from sklearn.preprocessing import MinMaxScaler

# 특성 스케일 [0, 1] 범위 조정
X_scaled = MinMaxScaler().fit_transform(X_train)
# UMAP 객체 생성, 훈련
mapper = umap.UMAP(n_components=2, metric="cosine").fit(X_scaled)
# 2D 임베딩의 데이터프레임
df_emb = pd.DataFrame(mapper.embedding_, columns=["X","Y"])
df_emb["label"] = y_train
# print(df_emb.head())

fig, axes = plt.subplots(2, 3, figsize=(7,5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names

for i, (label,cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                   gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]),axes[i].set_yticks([])
    
# plt.tight_layout()
# plt.show()

# 간단한 분류 모델 훈련
from sklearn.linear_model import LogisticRegression

# 수렴 보장, 'max_iter' 증가
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
# print(lr_clf.score(X_valid, y_valid))


# 간단한 분류 모델
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
# print(dummy_clf.score(X_valid, y_valid))


# 혼동 행렬
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
    
y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)



"""트랜스포머 미세 튜닝"""

# 사전 훈련 모델 로드
from transformers import AutoModelForSequenceClassification

num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))

# 성공 지표 정의
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# 모델 훈련
import os
from huggingface_hub import HfFolder

# Hugging Face API 토큰 설정
os.environ["HF_TOKEN"] = "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"  # Hugging Face API 토큰
HfFolder.save_token(os.getenv("HF_TOKEN"))

from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name, num_train_epochs=2, learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01, eval_strategy="epoch", disable_tqdm=False, logging_steps=logging_steps,
                                  push_to_hub=True, save_strategy="epoch", load_best_model_at_end=True, log_level="error")

trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"], eval_dataset=emotions_encoded["validation"], tokenizer=tokenizer)

trainer.train()

# 모델 푸시
# trainer.push_to_hub() # 주피터에서는 notebook_login()을 사용.



# 검증 세트 지표 확인
preds_output = trainer.predict(emotions_encoded["validation"])

# print(preds_output.metrics)

y_preds = np.argmax(preds_output.predictions, axis=1)

plot_confusion_matrix(y_preds, y_valid, labels)

# 오류 분석
from torch.nn.functional import cross_entropy

def forward_pass_with_label(batch):
    # 모든 입력 텐서, 모델과 같은 장치로 이동
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=1)
        loss = cross_entropy(output.logits, batch["label"].to(device),reduction="none")
    
    # 다른 데이터셋 열과 호환되도록 출력을 CPU로 옮김
    return {"loss": loss.cpu().numpy(), "predicted_label":pred_label.cpu().numpy()}


# 데이터셋 파이토치 텐서 변환
emotions_encoded.set_format("torch", columns=["input_ids","attention_mask", "label"])

# 손실 값 계산
emotions_encoded["validation"] = emotions_encoded["validation"].map(forward_pass_with_label, batched=True, batch_size=16)

emotions_encoded.set_format("pandas")
cols = ["text","label","predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))

# 데이터셋 특이사항 확인
# print(df_test.sort_values("loss", ascending=False).head(10))

# print(df_test.sort_values("loss", ascending=True).head(10))


# 모델 저장 및 공유
trainer.push_to_hub(commit_message="Training completed.")

from transformers import pipeline

# 허브 사용자 이름
model_id = "JEONGHEESIK/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

custom_tweet = "I saw a movie today and it was really good."
preds = classifier(custom_tweet, top_k=None)

preds_sorted = sorted(preds, key=lambda d: d['label'])
preds_df = pd.DataFrame(preds_sorted)
plt.bar(labels, 100*preds_df["score"], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()
