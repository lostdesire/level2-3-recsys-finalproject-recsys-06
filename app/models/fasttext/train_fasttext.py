from gensim.models import FastText
import os
import random
import numpy as np

# # 난수 시드 설정
# seed_value = 42
# random.seed(seed_value)
# np.random.seed(seed_value)

print('Py_start.')
fasttext_dir = '../fasttext'
if not os.path.exists(fasttext_dir):
    os.makedirs(fasttext_dir)
with open(f'{fasttext_dir}/corpus_total.txt', 'r') as f:
    corpus = f.read()
print('open_corpus_total.txt_done.')

num_workers = os.cpu_count() - 1
print('cpu_count?.', num_workers)

model = FastText(corpus_file=f'{fasttext_dir}/corpus_total.txt', vector_size=2000, window=20, min_count=5, workers=num_workers, sg=1, negative=100, seed=42)
print('Save_start.')
model.save(f'{fasttext_dir}/fasttext_vector2000_neg100.bin')
print('Save_done.')