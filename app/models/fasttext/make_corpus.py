import pandas as pd
import json

import re
import os
from konlpy.tag import Komoran


"""
1. 역할: fasttext 학습을 위한 5개 문서데이터 생성

    1) item_corpus
    2) store_corpus
    3) tag_corpus
    4) tag_dict: tokenizing 시 사전으로 활용
    5) corpus_total

    
2. 파일 위치 == 데이터와 같은 위치

    1) tag_output_items.json
    2) tag_output_stores.json
    3) items.json
    4) stores.json
"""

# 작업 디렉토리 변경
new_directory = '/home/wooksbaby/boostcamp6th/CteeEDA'
os.chdir(new_directory)

# 변경된 작업 디렉토리 확인
print("변경된 현재 작업 디렉토리:", os.getcwd())

#### 1) 태그 만들기
## 데이터 임포트
item_tags = pd.read_json('tag_output_items.json')
store_tags = pd.read_json('tag_output_stores.json')


## 중복 태그 정리
item_tags = item_tags.groupby('item_id').tags.sum()
store_tags = store_tags.groupby('store_id').tags.sum() # 하나의 스토어에 태그 아무리 많아도 일단은 다 첨부 (중복은 제거)


## 태그 into corpus
item_corpus = item_tags.apply(lambda x: x.replace('#', ' '))
item_corpus = '\n'.join(item_corpus)

store_corpus = store_tags.apply(lambda x: ' '.join(set(x.replace('#', ' ').split())))
store_corpus = '\n'.join(store_corpus)


## tag corpus, dictionary 저장
tag_corpus = store_corpus + item_corpus # 학습용 corpus
tag_dict = '\tNNG'.join(list(set(tag_corpus.split()))) + '\tNNG' # 불용어 처리 용 태그사전

with open('tag_corpus.txt', 'w') as f:
    f.write(tag_corpus)

with open('tag_dict.txt', 'w') as f:
    f.write(tag_dict)



#### 2) item, stores corpus 만들기
## 데이터 임포트
items_df = pd.read_json(f"/home/wooksbaby/boostcamp6th/20240308-data/items.json")  # 콘텐츠
cats_df = pd.read_json(f"/home/wooksbaby/boostcamp6th/20240219-data/categories_events.json")  # 이벤트 분류기준
stores_df = pd.read_json(f"/home/wooksbaby/boostcamp6th/20240219-data/stores.json")  # 크리에이터 스토어

items_df = items_df.drop_duplicates(keep='last')
stores_df = stores_df.drop_duplicates(keep='last')

## story 정리
def remove_html_tags(text):
    if pd.isna(text) or text is None:  # NaN 또는 None 체크
        return text

    clean_text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
    return clean_text

items_df['story']=items_df['story'].apply(remove_html_tags)


## 필요한 컬럼만 slicing
selected_columns_items_df = items_df[['id', 'store_id', 'simple_contents', 'content', 'story']]
selected_columns_stores_df = stores_df[['id', 'title', 'content', 'alias']]
selected_columns_stores_df = stores_df[['id', 'title', 'content', 'alias']]


## 컬럼명 재정의
selected_columns_items_df = selected_columns_items_df.rename(columns={
    'id': 'item_id',
    'simple_contents': 'item_simple_contents',
    'content': 'item_content',
    'story': 'item_story'
})

selected_columns_stores_df = selected_columns_stores_df.rename(columns={
    'id': 'store_id',
    'title': 'store_title',
    'content': 'store_content',
    'alias': 'store_alias'
})


## Tokenizing
# 불용어 리스트 정의
stopwords = set(['은', '는', '이', '가', '에', '에서', '으로', '도', '와', '과', '의', '하다', "아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라",
    "의해", "을", "를", "에", "의", "가", "으로", "로", "에게", "뿐이다", "의거하여", "근거하여",
    "입각하여", "기준으로", "예하면", "예를 들면", "예를 들자면", "저", "소인", "소생", "저희", "지말고",
    "하지마", "하지마라", "다른", "물론", "또한", "그리고", "비길수 없다", "해서는 안된다", "뿐만 아니라",
    "만이 아니다", "만은 아니다", "막론하고", "관계없이", "그치지 않다", "그러나", "그런데", "하지만",
    "든간에", "논하지 않다", "따지지 않다", "설사", "비록", "더라도", "아니면", "만 못하다",
    "하는 편이 낫다", "불문하고", "향하여", "향해서", "향하다", "쪽으로", "틈타", "이용하여", "타다",
    "오르다", "제외하고", "이 외에", "이 밖에", "하여야", "비로소", "한다면 몰라도", "외에도", "이곳",
    "여기", "부터", "기점으로", "따라서", "할 생각이다", "하려고하다", "이리하여", "그리하여", "그렇게 함으로써",
    "하지만", "일때", "할때", "앞에서", "중에서", "보는데서", "으로써", "로써", "까지", "해야한다", "일것이다",
    "반드시", "할줄알다", "할수있다", "할수있어", "임에 틀림없다", "한다면", "등", "등등", "제", "겨우", "단지",
    "다만", "할뿐", "딩동", "댕그", "대해서", "대하여", "대하면", "훨씬", "얼마나", "얼마만큼", "얼마큼", "남짓",
    "여", "얼마간", "약간", "다소", "좀", "조금", "다수", "몇", "얼마", "지만", "하물며", "또한", "그러나",
    "그렇지만", "하지만", "이외에도", "대해 말하자면", "뿐이다", "다음에", "반대로", "반대로 말하자면", "이와 반대로",
    "바꾸어서 말하면", "바꾸어서 한다면", "만약", "그렇지않으면", "까악", "툭", "딱", "삐걱", "보드득", "비걱거리다",
    "꽈당", "응당", "해야한다", "에 가서", "각", "각각", "여러분", "각종", "각자", "제각기", "하도록하다", "와", "과",
    "그러므로", "그래서", "고로", "한 까닭에", "하기 때문에", "거니와", "이지만", "대하여", "관하여", "관한", "과연",
    "실로", "아니나다를가", "생각한대로", "진짜로", "한적이있다", "하곤하였다", "하", "하하", "허허", "아하", "거바", "와",
    "오", "왜", "어째서", "무엇때문에", "어찌", "하겠는가", "무슨", "어디", "어느곳", "더군다나", "하물며", "더욱이는",
    "어느때", "언제", "야", "입니다", "미치다", "예요", "빌다", "가다", "살다", "만들다", "살아가다", "되어다", "담다", "있다",

    "안녕하세요", "러스트", '취소', '환불', '할인', '수령', '경우', '해주시', '이후', '주문', '필요', '확인', '주세', '스타일', '된다', '니다', '동일', '주시', '구매', '교환', '뱉다'
    ])

# 사전에 추가하고 싶은 단어들 = ['스타일링', '일러스트', '해당', '감사', '연락', '리그오브레전드', '리그 오브 레전드' ,'롤','파이널 프로', ]

tokenizer = Komoran(userdic = 'tag_dict.txt')

def preprocess_text(text):
    if not isinstance(text, str):
        return ''

    # 잘못된 토큰화 대상 수정 예: '입니다' -> ' 입니다'
    text = re.sub(r'입니다', ' 입니다', text)
    # 특수 문자를 공백으로 치환
    text = re.sub(r'[()\[\]/]_-=', ' ', text)
    text = re.sub(r'[\"\[\(\ufeff]', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\(.+?\)', ' ', text)
    text = re.sub(r'\[.+?\]', ' ', text)
    text = re.sub(r'"', ' ', text)
    text = re.sub(r'\\ufeff', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', text).strip()

    words = tokenizer.nouns(clean_text)
    words = [word for word in words if word not in stopwords and len(word) > 1]     # 불용어 제거 및 한 글자 단어 제거는 필요에 따라 추가

    return ' '.join(words)


## tokenizing
stores_df['cleaned_content'] = stores_df['content'].apply(lambda x: preprocess_text(x))
items_df['cleaned_simple_contents'] = items_df['simple_contents'].apply(lambda x: preprocess_text(x))
items_df['cleaned_content'] = items_df['content'].apply(lambda x: preprocess_text(x))
items_df['cleaned_story'] = items_df['story'].apply(lambda x: preprocess_text(x))

items_selected = items_df[['cleaned_simple_contents', 'cleaned_content', 'cleaned_story']]
stores_selected = stores_df[['title', 'cleaned_content', 'alias']]

items_selected = items_selected.drop_duplicates()
stores_selected = stores_selected.drop_duplicates()


## FastText 입력 형식으로 변환
items_corpus = '\n'.join(items_selected.apply(lambda x: ' '.join(x.dropna().values), axis=1))
stores_corpus = '\n'.join(stores_selected.apply(lambda x: ' '.join(x.dropna().values), axis=1))

corpus = items_corpus +'\n'+ stores_corpus +'\n' + tag_corpus

# fasttext 폴더의 경로를 확인하고, 없으면 생성
fasttext_dir = '../fasttext'
if not os.path.exists(fasttext_dir):
    os.makedirs(fasttext_dir)

with open(f'{fasttext_dir}/items_for_fasttext.txt', 'w', encoding='utf-8') as f:
    f.write(items_corpus)

with open(f'{fasttext_dir}/stores_for_fasttext.txt', 'w', encoding='utf-8') as f:
    f.write(stores_corpus)

with open(f'{fasttext_dir}/corpus_total.txt', 'w', encoding='utf-8') as f:
    f.write(corpus)
print('Corpus_Save_Done!!')