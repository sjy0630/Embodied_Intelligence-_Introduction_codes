import numpy as np
from sentence_transformers import SentenceTransformer
import random
from sklearn.neural_network import MLPClassifier
import joblib
import os

# 配置路径
dataset_path = 'dataset.txt'
mlp_save_path = 'model/MLP/command_classifier.pkl'

# [新增] 确保保存目录存在，防止报错
os.makedirs(os.path.dirname(mlp_save_path), exist_ok=True)

print('正在加载语言模型...')
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print('语言模型加载完成')

dataset = []
print(f"正在读取数据集: {dataset_path} ...")

# --- [修改] 增强版文件读取逻辑 (防崩溃) ---
try:
    with open(dataset_path, encoding='utf-8') as fp:
        for line_num, line in enumerate(fp, 1):
            line = line.strip()
            
            # 1. 跳过空行
            if not line:
                continue
            
            # 2. 检查分隔符
            if '|' not in line:
                print(f"⚠️ 警告: 第 {line_num} 行格式错误 (缺少 '|'), 已跳过: {line}")
                continue
            
            # 3. 安全分割 (兼容有空格 ' | ' 和无空格 '|')
            try:
                parts = line.split('|')
                if len(parts) < 2:
                    print(f"⚠️ 警告: 第 {line_num} 行数据不完整, 已跳过: {line}")
                    continue
                
                # strip() 去除前后空格
                text = parts[0].strip()
                label = parts[1].strip()
                
                # 4. 再次检查内容是否为空
                if not text or not label:
                    print(f"⚠️ 警告: 第 {line_num} 行存在空内容, 已跳过: {line}")
                    continue

                dataset.append({"text": text, "label": label})
                
            except Exception as e:
                print(f"⚠️ 警告: 处理第 {line_num} 行时发生未知错误: {e}")

except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {dataset_path}，请确保它和本脚本在同一目录下。")
    exit()

if len(dataset) == 0:
    print("❌ 错误: 没有读取到任何有效数据！")
    exit()
# ----------------------------------------

random.seed(42)
random.shuffle(dataset)

# 划分训练集和测试集 (7:3)
split_idx = int(len(dataset) * 0.7)
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]

print("正在生成向量 (Embedding)...")

X_train = []
Y_train = []
for data in train_dataset:
    text = data["text"]
    label = data["label"]
    embedding = model.encode(text)
    X_train.append(embedding)
    Y_train.append(label)

X_test = []
Y_test = []
for data in test_dataset:
    text = data["text"]
    label = data["label"]
    embedding = model.encode(text)
    X_test.append(embedding)
    Y_test.append(label)

# 仅用于统计总数
X = np.concatenate([X_train, X_test], axis=0) if X_test else np.array(X_train)

print(f'成功导入 {len(dataset)} 条有效数据，自动划分训练集和测试集')
print(f'训练集：{len(X_train)}条，测试集：{len(X_test)}条')

print('\n正在训练语音指令分类模型...')
# 稍微增加网络层数以处理更复杂的语义
mlp = MLPClassifier(max_iter=2000, hidden_layer_sizes=(256, 128, 64), random_state=42)
mlp.fit(X_train, Y_train)
print('分类模型训练完成\n')

# 计算准确率
correct_count = 0
for x, y in zip(X_train, Y_train):
    pred = mlp.predict([x])
    if pred[0] == y:
        correct_count += 1
acc_train = correct_count / len(X_train) * 100 if len(X_train) > 0 else 0
print(f"训练集准确度: {acc_train:.2f}%")

correct_count = 0
if len(X_test) > 0:
    for x, y in zip(X_test, Y_test):
        pred = mlp.predict([x])
        if pred[0] == y:
            correct_count += 1
    acc_test = correct_count / len(X_test) * 100
    print(f"测试集准确度: {acc_test:.2f}%")
else:
    print("测试集为空，跳过准确度计算。")

joblib.dump(mlp, mlp_save_path)
print(f'\n分类模型已保存至：{mlp_save_path}')

# --- 简单测试 ---
print("\n--- 简单测试 ---")
test_phrases = ["不要左转", "别动", "往左拐"]
print(f"{'指令':<15} | {'预测结果'}")
print("-" * 30)
for text in test_phrases:
    vec = model.encode(text)
    pred = mlp.predict([vec])[0]
    print(f"{text:<15} | {pred}")