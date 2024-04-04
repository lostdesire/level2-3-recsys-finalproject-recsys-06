import json
import api.streamlit as st

#streamlit run .\streamlit_demo.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
service = Service(executable_path='/home/wooksbaby/boostcamp6th/FinalProject/api/app/chromedriver-linux64/chromedriver')


import requests


@st.cache_data
def crawl_image_item(url):
    
    options = Options() 
    options.add_argument("--headless=new")
    options.add_argument('--disable-gpu')

    driver = webdriver.Chrome(service=service,options=options)

    with webdriver.Chrome(service=service,options=options) as driver:
        driver.get(url)
        
        img_element = driver.find_element(By.CLASS_NAME, 'item_img')
        img_url = img_element.get_attribute('src')

    return img_url

@st.cache_data
def crawl_image_store(url):
    
    options = Options() 
    options.add_argument("--headless=new")
    options.add_argument('--disable-gpu')

    driver = webdriver.Chrome(options=options)

    with webdriver.Chrome(options=options) as driver:
        driver.get(url)
        
        img_element = driver.find_element(By.CLASS_NAME, 'profile_img')
        img_url = img_element.get_attribute('src')

    return img_url


def search_stores(data, user_input):
    matching_items = []
    matching_id=set()
    for item in data:
        if (item["id"] not in matching_id) and (user_input in item["title"] or user_input in item["content"]):
            matching_id.add(item["id"])
            matching_items.append({"id": item["id"], "title": item["title"], "content": item["content"],"view_count": item["view_count"],"alias": item['alias'],"updated_at":item["updated_at"]})
    return matching_items

def search_items(data, user_input):
    matching_items = []
    matching_id=set()
    for item in data:
        content = item.get("content", "")
        simple_contents = item.get("simple_contents", "")
        if (item["id"] not in matching_id) and ((simple_contents and user_input in simple_contents)) :
            matching_id.add(item["id"])
            matching_items.append({"id": item["id"], "content": item["content"], "simple_contents": item["simple_contents"],"view_count": item["view_count"],"updated_at":item['updated_at']})
    return matching_items

def sorting_result_item(options,matching_items):
    if options== '조회순':
        matching_items= sorted(matching_items, key=lambda x: x['view_count'], reverse=True)
    elif options ==  '최신순':
        matching_items= sorted(matching_items, key=lambda x: x['updated_at'], reverse=True)

    return matching_items

def sorting_result_event(options,matching_items):
    if options ==  '최신순':
        matching_items= sorted(matching_items, key=lambda x: x['updated_at'], reverse=True)

    return matching_items
def get_alias_by_id(stores_data, store_id):
    for store in stores_data:
        if store["id"] == store_id:
            return store["alias"]
    return None

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def add_to_cart(item):
    shopping_cart.append(item)



def display_shopping_cart(shopping_cart):
    for item in shopping_cart:
        url = f"https://ctee.kr/item/store/{item}"
        st.sidebar.image(crawl_image_item(url), width=100)
        st.sidebar.write(f"{get_simple_contents(item_data, item)}")


def get_simple_contents(item_data, item_id):
    for item in item_data:
        if item['id'] == item_id:
            return item['simple_contents']
    return None


def display(item_data,shopping_cart):
    num_images = len(shopping_cart)
    num_images_per_row = min(num_images, 3)
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    for i in range(num_rows):
        row = st.columns(num_images_per_row)
        for j in range(num_images_per_row):
            index = i * num_images_per_row + j
            if index < num_images:
                item = shopping_cart[index]
                url = f"https://ctee.kr/item/store/{item}"
                with row[j]:
                    try:
                        st.markdown(f"<a href='{url}'><img src='{crawl_image_item(url)}' width='200'></a>", unsafe_allow_html=True)
                        st.write(f"{get_simple_contents(item_data, item)}")
                    except:
                        pass

def user_select(options):
    user_basket = []
    if options== '유저A':
        user_basket=[12072, 12070]
        recommend=[20532, 20970, 20088, 17255, 19524, 15632, 19666, 21117, 16881, 19701]
    elif options ==  '유저B':
        user_basket=[2052, 3125]
        recommend=[20532, 20970, 1035, 20088, 17255, 19524, 15632, 19666, 21117, 16881]
        

    return user_basket, recommend


def main(stores_data, item_data):
    st.sidebar.header("장바구니")
    display_shopping_cart(shopping_cart)

    with tab1:
        st.header("검색")
        options = ['조회순', '최신순','오래된순']
        selected_option = st.selectbox('정렬', options)
        user_input = st.text_input("찾고 계신 크리에이터 또는 콘텐츠가 있나요?", "")

        if st.button("검색"):
            #st.write("최근 검색어:", user_input)

            if not user_input:
                st.write("검색어를 입력하세요.")
            else: 
                server_url = "http://127.0.0.1:8000"
                query =user_input
                n=5
                res=requests.get(f"{server_url}/search/{query}/{n}")
                item_data = res.json()[0]
                store_data = res.json()[1]
                matching_items=sorting_result_item(selected_option,item_data)

                if matching_items:
                    st.subheader(f"콘텐츠상품 {len(matching_items)}")
                    num_items = len(matching_items)
                    num_items_per_row = min(num_items, 3)
                    num_rows = (num_items + num_items_per_row - 1) // num_items_per_row

                    for i in range(num_rows):
                        row = st.columns(num_items_per_row)
                        for j in range(num_items_per_row):
                            index = i * num_items_per_row + j
                            if index < num_items:
                                item = matching_items[index]
                                url = f"https://ctee.kr/item/store/{item['id']}"
                                with row[j]:    
                                    caption=f"{item['simple_contents']}"
                                    st.markdown(f"<a href='{url}'><img src='{crawl_image_item(url)}' width='200'></a>", unsafe_allow_html=True)
                                    st.write(f"<span style='font-size:14px; color:gray;'>{caption}</span>", unsafe_allow_html=True)

                                    checkbox_state = st.checkbox(f"장바구니 추가", key=f"checkbox_{item['id']}")
                                    #if checkbox_state:
                                        #add_to_cart(item)
                                    
                                    

                else:
                    st.write("아이템 검색 결과가 없습니다.")

                st.write("----------------------------------------------")
                
                matching_stores=sorting_result_event(selected_option,store_data)

                if matching_stores:
                    st.subheader(f"Event {len(matching_stores)}")
                    
                    # 이미지를 한 줄에 출력하기 위해 열 생성
                    num_images = len(matching_stores)
                    num_images_per_row = min(num_images, 4)
                    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row
                    

                    for i in range(num_rows):
                        row = st.columns(num_images_per_row)
                        for j in range(num_images_per_row):
                            index = i * num_images_per_row + j
                            if index < num_images:
                                store = matching_stores[index]
                                # alias=get_alias_by_id(stores_data,store['store_id'])
                                print(store)
                                url = f"https://ctee.kr/place/{store['alias']}"
                                with row[j]:
                                    caption=f"{store['title']}"
                                    try:
                                        st.markdown(f"<a href='{url}'><img src='{crawl_image_store(url)}' width='150'></a>", unsafe_allow_html=True)
                                        st.write(f"<span style='font-size:14px; color:gray;'>{caption}</span>", unsafe_allow_html=True)
                                    except:
                                        pass
                                        

                else:
                    st.write("스토어 검색 결과가 없습니다.")
    with tab2:
#         user_a=[12072, 12070]
# user_b=[2052, 3125]
        #options = ['유저A', '유저B','유저C']
        #selected_option = st.selectbox('유저선택', options)
        #user_basket, recommend=user_select(selected_option)

        user_input = st.text_input("유저의 과거구매정보 입력", "")

        if st.button("입력"):
            st.write("최근 검색어:", user_input)
            user_input_list = user_input.split(',')
            # user = user_input_list.pop(0)
            user = ["3"]

            st.subheader(f"과거 구매목록 {len(user_input_list)}")
            display(item_data, user_input_list)
            
            st.subheader(f"장바구니 {len(shopping_cart)}")  
            display(item_data, shopping_cart)
            
            server_url = "http://127.0.0.1:8000"
            print(user_input_list)
            query = ','.join(list(user) + user_input_list + list(map(str, shopping_cart)))
            res=requests.get(f"{server_url}/recommend/{query}")
            print()
            print(res.json())
            print()
            recommend = res.json()

            st.subheader(f"추천 {len(recommend)}")
            display(item_data,recommend)


            


if __name__ == "__main__":
    # stores.json 파일 로드
    with open('/home/wooksbaby/boostcamp6th/20240219-data/stores.json', 'r', encoding='utf-8') as f:
        stores_data = json.load(f)

    # item.json 파일 로드
    with open('/home/wooksbaby/boostcamp6th/20240308-data/items.json', 'r', encoding='utf-8') as f:
        item_data = json.load(f)

    st.title("DeConn")
    st.write("환영합니다! 원하는 디지털 콘텐츠를 한번에 DeConn")

    shopping_cart = [19660, 19504]
    tab1, tab2= st.tabs(['검색' , '장바구니'])
    
    main(stores_data, item_data)
    
