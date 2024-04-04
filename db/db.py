import pymysql
import mariadb
import sys
import json
import os
import re
import configparser

config_path = YOUR_DB_PATH
config = configparser.ConfigParser()
config.read(config_path + "db.ini")

try:
    conn = mariadb.connect(
        user=config["DB"]["user"],
        password=config["DB"]["password"],
        host=config["DB"]["host"],
        port=int(config["DB"]["port"]),
        database=config["DB"]["database"],
    )
except mariadb.Error as e:
    print(f"Error : {e}")
    sys.exit(1)

cur = conn.cursor()

# 여러 JSON 파일 경로 리스트
data_path = YOUR_DATA_PATH
json_files = [
    data_path + "stores.json",
    data_path + "items.json",
    data_path + "event_projects.json",
    data_path + "orders_items.json",
    data_path + "tag_output_stores.json",
    data_path + "tag_output_items.json",
]


def create_database():
    """
    데이터베이스 생성 또는 연결
    """

    # stores 테이블 생성
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stores (
            store_id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            alias TEXT,
            view_count INTEGER,
            donation_count INTEGER,
            tags TEXT,
            category TEXT,
            updated_at TEXT
        )
    """
    )

    # items 테이블 생성
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            store_id INTEGER,
            simple_contents TEXT,
            content TEXT,
            story LONGTEXT,
            price INTEGER,
            is_adult INTEGER,
            view_count INTEGER,
            updated_at TEXT,
            tags TEXT,
            category TEXT
        )
    """
    )

    # event_projects 테이블 생성
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS event_projects (
            id INTEGER PRIMARY KEY,
            store_id INTEGER,
            title TEXT,
            simple_description TEXT,
            updated_at TEXT,
            story LONGTEXT
        )
    """
    )

    # orders_items 테이블 생성
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS orders_items (
            id INTEGER PRIMARY KEY,
            store_id INTEGER,
            user_id INTEGER,
            item_id INTEGER,
            orders_basket_id INTEGER,
            state INTEGER,
            price INTEGER,
            discount_price INTEGER,
            total_price INTEGER,
            product_currency_code TEXT,
            currency_code TEXT,
            exchange_calculation_points TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """
    )

    conn.commit()
    return conn


def insert_json_to_table(conn, table_name, json_file_path):
    """
    JSON 파일을 특정 테이블에 삽입
    """
    with open(json_file_path, "r", encoding="UTF-8") as file:
        json_list = json.load(file)

        # 중복을 제거하기 위한 세트(set)을 생성하여 중복을 체크
        unique_set = set()

        for json_data in json_list:
            # story HTML 태그 제거
            if "story" in json_data:
                clean_text = ""
                if json_data["story"] is not None:
                    clean_text = re.sub(r"<.*?>", "", json_data["story"])

            # primary key 중복 데이터 컨트롤 (첫 데이터만 받음)
            if table_name == "tag_output_stores":
                key = json_data["store_id"]
            elif table_name == "tag_output_items":
                key = json_data["item_id"]
            elif table_name == "funnels":
                pass
            else:
                key = json_data["id"]

            # 각 테이블에 맞게 데이터 삽입
            # key가 사용되지 않은 데이터 삽입
            if key not in unique_set:
                if table_name == "stores":
                    cur.execute(
                        "REPLACE INTO stores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            json_data["id"],
                            json_data["title"],
                            json_data["content"],
                            json_data["alias"],
                            json_data["view_count"],
                            json_data["donation_count"],
                            json_data["updated_at"],
                            "",
                            "",
                        ),
                    )
                elif table_name == "items":
                    cur.execute(
                        "REPLACE INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            json_data["id"],
                            json_data["store_id"],
                            json_data["simple_contents"],
                            json_data["content"],
                            clean_text,
                            json_data["price"],
                            json_data["is_adult"],
                            json_data["view_count"],
                            json_data["updated_at"],
                            "",
                            "",
                        ),
                    )
                elif table_name == "event_projects":
                    cur.execute(
                        "REPLACE INTO event_projects VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            json_data["id"],
                            json_data["store_id"],
                            json_data["title"],
                            json_data["simple_description"],
                            json_data["updated_at"],
                            clean_text,
                        ),
                    )
                elif table_name == "orders_items":
                    cur.execute(
                        "REPLACE INTO orders_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            json_data["id"],
                            json_data["store_id"],
                            json_data["user_id"],
                            json_data["item_id"],
                            json_data["orders_basket_id"],
                            json_data["state"],
                            json_data["price"],
                            json_data["discount_price"],
                            json_data["total_price"],
                            json_data["product_currency_code"],
                            json_data["currency_code"],
                            json_data["exchange_calculation_points"],
                            json_data["created_at"],
                            json_data["updated_at"],
                        ),
                    )
                elif table_name == "tag_output_stores":
                    sql = (
                        "UPDATE stores SET tags = %s, category = %s WHERE store_id = %s"
                    )
                    cur.execute(
                        sql,
                        (
                            json_data["tags"],
                            json_data["category"],
                            json_data["store_id"],
                        ),
                    )
                elif table_name == "tag_output_items":
                    sql = "UPDATE items SET tags = %s, category = %s WHERE id = %s"
                    cur.execute(
                        sql,
                        (
                            json_data["tags"],
                            json_data["category"],
                            json_data["item_id"],
                        ),
                    )
                unique_set.add(key)

    conn.commit()


def insert_all_json_to_db(json_files):
    """
    모든 JSON 파일을 데이터베이스에 삽입
    """
    for json_file in json_files:
        table_name = os.path.splitext(os.path.basename(json_file))[0]
        insert_json_to_table(conn, table_name, json_file)

    conn.close()


# 함수 호출
create_database()
insert_all_json_to_db(json_files)
