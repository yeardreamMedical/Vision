import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 1. 파일로 저장해 둔 임베딩과 레이블을 불러옵니다.
X_embeddings_raw = np.load('cxr_embeddings.npy')
y_labels = np.load('cxr_labels.npy', allow_pickle=True)

# 2. 임베딩 데이터 형태(shape) 변경
X_embeddings = X_embeddings_raw.reshape(X_embeddings_raw.shape[0], -1)
print(f"임베딩 데이터를 t-SNE 입력에 맞게 {X_embeddings_raw.shape} -> {X_embeddings.shape} 형태로 변경했습니다.")


print("\nt-SNE를 실행하여 2차원으로 차원 축소를 시작합니다...")
print("데이터 양에 따라 시간이 다소 소요될 수 있습니다.")

# 3. t-SNE 모델을 사용하여 2차원으로 차원 축소합니다.
# ✅ n_iter를 max_iter로 수정합니다.
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, learning_rate='auto')
embeddings_2d = tsne.fit_transform(X_embeddings)

# 4. 시각화를 위해 DataFrame을 생성합니다.
df_2d = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
df_2d['label'] = y_labels

# 5. Seaborn과 Matplotlib을 사용하여 결과를 플로팅합니다.
plt.figure(figsize=(16, 12))
sns.scatterplot(
    data=df_2d,
    x='x',
    y='y',
    hue='label',
    palette=sns.color_palette("hsv", n_colors=len(df_2d['label'].unique())),
    alpha=0.7,
    s=50
)
plt.title('t-SNE Visualization of CXR Image Embeddings', fontsize=16)
plt.xlabel('t-SNE component 1', fontsize=12)
plt.ylabel('t-SNE component 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.grid(True)
plt.show()