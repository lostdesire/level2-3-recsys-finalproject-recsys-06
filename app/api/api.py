from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from gensim.models import FastText
from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy import or_
from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.engine import URL
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
import configparser
import pickle
import os
import numpy as np
import pandas as pd
from scipy import sparse

import json
import torch
import sys

sys.path.append(YOUR_DIR_PATH)
from FinalProject.api.app.models.EASE.models import EASE

model = EASE(1, 2)
model_version = "EASE 240327_094259"
model_path = YOUR_MODEL_PATH
## model load
model.load_state_dict(torch.load(f"{model_path}/EASE/{model_version}.pt"))
model.eval()

# interaction_matrix 로드 잠시만요~넵!
data_inf = pd.read_csv(f"{model_path}/EASE/data/inference.csv")

n_rows = data_inf.uid.nunique()
n_cols = data_inf.iid.nunique()

rows = data_inf["uid"]
cols = data_inf["iid"]

interaction_matrix = sparse.csr_matrix(
    (np.ones_like(rows), (rows, cols)), dtype="float64", shape=(n_rows, n_cols)
)

G = interaction_matrix.T @ interaction_matrix  # item_num * item_num
G += 100 * sparse.identity(G.shape[0])
G = G.todense()

P = np.linalg.inv(G)
B = -P / np.diag(
    P
)  # equation 8 in paper: B_{ij}=0 if i = j else -\frac{P_{ij}}{P_{jj}}
np.fill_diagonal(B, 0.0)

item_similarity = B  # item_num * item_num
item_similarity = np.array(item_similarity)
model.item_similarity = item_similarity

## json load
with open(
    f"{model_path}/EASE/json_id/id2show.json",
    "r",
) as json_file:
    id2show = json.load(json_file)
id2show = json.loads(id2show)

with open(
    f"{model_path}/EASE/json_id/show2id.json",
    "r",
) as json_file:
    show2id = json.load(json_file)
show2id = json.loads(show2id)

with open(
    f"{model_path}/EASE/json_id/profile2id.json",
    "r",
) as json_file:
    profile2id = json.load(json_file)
profile2id = json.loads(profile2id)


# 모델 저장
def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


# 모델 로드
def load_model(filename):
    if not os.path.exists(filename):
        return ""

    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


# FastText 모델 로드
# 이 경로는 사전 훈련된 FastText 모델 파일의 실제 경로로 대체해야 합니다.
print("fasttext_load_start")
fasttext_model = FastText.load(
    f"{model_path}/fasttext/fasttext_vector2000_neg100.bin"
)
print("fasttext_loaded")

# 데이터베이스 설정
config_path = YOUR_DB_PATH
config = configparser.ConfigParser()
config.read(config_path + "db.ini")

print("bm25 model load start")
item_model = load_model(config_path + "bm25_item.pkl")
store_model = load_model(config_path + "bm25_store.pkl")
print("bm25 model load end")

DATABASE_URL = URL.create(
    drivername="mariadb+aiomysql",
    username=config["DB"]["user"],
    password=config["DB"]["password"],
    host=config["DB"]["host"],
    port=int(config["DB"]["port"]),
    database=config["DB"]["database"],
)

engine = create_async_engine(DATABASE_URL)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# 모델 정의
Base = declarative_base()

# 형태소 분석기 초기화
okt = Okt()


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"))
    content = Column(Text)
    simple_contents = Column(Text)
    story = Column(Text)
    tags = Column(String(255))
    category = Column(String(255))
    view_count = Column(Integer, default=0)
    updated_at = Column(DateTime)


class Store(Base):
    __tablename__ = "stores"
    store_id = Column(Integer, primary_key=True)
    title = Column(String(255))
    content = Column(Text)
    alias = Column(String(255))
    view_count = Column(Integer, default=0)
    tags = Column(String(255))
    category = Column(String(255))
    updated_at = Column(DateTime)


class Event(Base):
    __tablename__ = "event_projects"
    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"))
    title = Column(String(255))
    simple_description = Column(Text)
    story = Column(Text)
    updated_at = Column(DateTime)


# 비동기 데이터베이스 세션 의존성
async def get_db():
    async with SessionLocal() as session:
        yield session


# FastText를 이용하여 검색어와 유사한 단어를 찾는 함수
def get_similar_words(query: str, topn: int = 5) -> List[str]:
    similar_words = fasttext_model.wv.most_similar(query, topn=topn)
    # get_nearest_neighbors 함수는 (유사도, 단어) 튜플을 반환하므로 단어만 추출

    print(similar_words)
    return [word for word, _ in similar_words]


# 검색어와 관련된 항목을 데이터베이스에서 검색하는 엔드포인트
@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI application!"}


@app.get("/search/{query}")
async def search_bm25(query: str, db: AsyncSession = Depends(get_db)):
    items = await db.execute(select(Item))
    items = list(set(items.scalars().all()))
    items.sort(key=lambda x: x.id)
    stores = await db.execute(select(Store))
    stores = list(set(stores.scalars().all()))
    stores.sort(key=lambda x: x.store_id)

    all_item_documents = []
    all_store_documents = []

    for item in items:
        content = item.content if item.content else ""
        simple_contents = item.simple_contents if item.simple_contents else ""
        story = item.story if item.story else ""
        tags = item.tags if item.tags else ""
        category = item.category if item.category else ""
        all_item_documents.append(
            content + " " + simple_contents + " " + story + " " + tags + " " + category
        )

    for store in stores:
        title = store.title if store.title else ""
        content = store.content if store.content else ""
        alias = store.alias if store.alias else ""
        tags = store.tags if store.tags else ""
        category = store.category if store.category else ""
        all_store_documents.append(
            title + " " + content + " " + alias + " " + tags + " " + category
        )

    # 검색어 확장
    print(query)
    expanded_queries = get_similar_words(query)
    queries = [query]
    queries.extend(expanded_queries)

    unique_item_ids = set()
    unique_store_ids = set()
    unique_items = []
    unique_stores = []

    for q in queries:
        item_ids = search(q, item_model, [item.id for item in items])
        store_ids = search(
            q,
            store_model,
            [store.store_id for store in stores],
        )

        # id, store_id 추출
        item_result = []
        for id in item_ids:
            item_result.append(id)
        store_result = []
        for id in store_ids:
            store_result.append(id)

        # 중복 없이 추가
        for item in items:
            if item.id not in unique_item_ids and item.id in item_result:
                unique_item_ids.add(item.id)
                unique_items.append(item)
        
        for store in stores:
            if store.store_id not in unique_store_ids and store.store_id in store_result:
                unique_store_ids.add(store.store_id)
                unique_stores.append(store)

    # 조회수 순으로 정렬
    unique_items.sort(key=lambda x: x.view_count, reverse=True)
    unique_stores.sort(key=lambda x: x.view_count, reverse=True)
    
    # events 검색
    events_result = await search_events(query, expanded_queries, db)

    return unique_items, unique_stores, events_result


@app.get("/save")
async def save_models(db: AsyncSession = Depends(get_db)):
    items = await db.execute(select(Item))
    items = list(set(items.scalars().all()))
    items.sort(key=lambda x: x.id)
    stores = await db.execute(select(Store))
    stores = list(set(stores.scalars().all()))
    stores.sort(key=lambda x: x.store_id)

    all_item_documents = []
    all_store_documents = []

    for item in items:
        content = item.content if item.content else ""
        simple_contents = item.simple_contents if item.simple_contents else ""
        story = item.story if item.story else ""
        tags = item.tags if item.tags else ""
        category = item.category if item.category else ""
        all_item_documents.append(
            content + " " + simple_contents + " " + story + " " + tags + " " + category
        )

    for store in stores:
        title = store.title if store.title else ""
        content = store.content if store.content else ""
        alias = store.alias if store.alias else ""
        tags = store.tags if store.tags else ""
        category = store.category if store.category else ""
        all_store_documents.append(
            title + " " + content + " " + alias + " " + tags + " " + category
        )
    
    global item_model
    global store_model
    item_model = train_bm25_model(all_item_documents)
    store_model = train_bm25_model(all_store_documents)

    save_model(item_model, config_path + "bm25_item.pkl")
    save_model(store_model, config_path + "bm25_store.pkl")

    return "model saved"


def tokenize(text):
    tokens = okt.nouns(text)
    tokens = [token for token in tokens if len(token) > 1]

    return tokens


# Okapi BM25 모델 초기화 및 학습
def train_bm25_model(documents):
    tokenized_docs = [tokenize(doc) for doc in documents]

    return BM25Okapi(tokenized_docs)


# 검색 함수
def search(query, model, ids):
    tokenized_query = tokenize(query)
    scores = model.get_scores(tokenized_query)

    # 점수가 0 초과인 항목들의 인덱스 찾기
    selected_indices = [i for i, score in enumerate(scores) if score > 0]
    
    # 선택된 인덱스를 점수가 큰 순서대로 정렬
    sorted_indices = sorted(selected_indices, key=lambda i: scores[i], reverse=True)

    # 정렬된 인덱스에 따라 선택된 ID 가져오기
    selected_ids = [ids[i] for i in sorted_indices]

    return selected_ids


async def search_events(original_query, expanded_queries, db):
    # 원본 및 확장된 검색어로 'title'에서 검색
    query_set = [original_query] + expanded_queries
    events_result = []
    for query in query_set:
        result = await db.execute(
            select(Event)
            .where(Event.title.contains(query))
            .order_by(Event.updated_at.desc())
        )
        events_result.extend(result.scalars().all())

    # 중복 제거
    unique_events = list({event.id: event for event in events_result}.values())
    return unique_events


## streamlit 용
@app.get("/recommend/{query}")
async def recommend(query: str):
    """
    query: 장바구니 목록 나열한 문자열 : "아이템1,아이템2,..."
    """

    global model, show2id, id2show, profile2id

    query = query.split(",")

    # 유저
    profile = query[0]

    try:
        user = profile2id[str(profile)]
    except KeyError:
        pass

    # 아이템
    basket = []
    for show in query[1:]:
        try:
            basket.append(show2id[str(show)])
        except KeyError:
            continue

    interaction = interaction_matrix[[user]]
    interaction[0, basket] = 9

    pred1 = model.rank_new(interaction)[0]
    result = [id2show[str(id)] for id in pred1]

    return result


print("done!")
