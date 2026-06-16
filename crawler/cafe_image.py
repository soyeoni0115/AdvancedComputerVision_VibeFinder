import os
import time
import urllib.request
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

import undetected_chromedriver as uc

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ==========================================
# 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

SAVE_DIR = BASE_DIR / "data" / "final_raw"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# DB
# ==========================================

def get_db_connection():

    return psycopg2.connect(
        os.getenv("DATABASE_URL")
    )


def get_target_cafes():

    conn = get_db_connection()

    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            cafe_name,
            map_url
        FROM cafes_final
        WHERE id = 57
        ORDER BY id
    """)

    cafes = cur.fetchall()

    cur.close()
    conn.close()

    return cafes


# ==========================================
# 크롤링
# ==========================================

def crawl_and_save(
    driver,
    cafe_id,
    cafe_name,
    map_url
):

    try:

        print()
        print("================================")
        print("ID   :", cafe_id)
        print("NAME :", cafe_name)
        print("URL  :", map_url)
        print("================================")

        print("🌐 이동 시작")

        driver.get(map_url)

        print("✅ 이동 완료")
        print("현재 URL :", driver.current_url)
        print("현재 제목:", driver.title)

        time.sleep(5)

        # ==================================
        # 검색결과 페이지 처리
        # ==================================

        if "검색 - 네이버지도" in driver.title:

            print("🔍 검색결과 페이지")

            WebDriverWait(driver, 10).until(
                EC.frame_to_be_available_and_switch_to_it(
                    (By.ID, "searchIframe")
                )
            )

            links = driver.find_elements(
                By.TAG_NAME,
                "a"
            )

            target = None

            for link in links:

                try:

                    text = link.text.strip()

                    if cafe_name in text:

                        target = link
                        break

                except:
                    pass

            if target:

                driver.execute_script(
                    "arguments[0].click();",
                    target
                )

                print("✅ 검색결과 클릭")

                time.sleep(5)

            else:

                print("❌ 검색결과 없음")

                driver.switch_to.default_content()

                return "ERROR"

            driver.switch_to.default_content()

        # ==================================
        # entryIframe
        # ==================================

        WebDriverWait(driver, 15).until(
            EC.frame_to_be_available_and_switch_to_it(
                (By.ID, "entryIframe")
            )
        )

        print("✅ entryIframe 진입")

        # ==================================
        # 사진 탭
        # ==================================

        photo_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//span[text()='사진']"
                )
            )
        )

        driver.execute_script(
            "arguments[0].click();",
            photo_tab
        )

        print("📸 사진 탭 클릭 성공")

        time.sleep(3)

        # ==================================
        # 내부 탭
        # ==================================

        interior_elements = driver.find_elements(
            By.XPATH,
            "//*[contains(text(),'내부')]"
        )

        if not interior_elements:

            print("⚠️ 내부 탭 없음")

            driver.switch_to.default_content()

            return "NO_INTERIOR"

        interior_btn = interior_elements[0]

        driver.execute_script(
            "arguments[0].click();",
            interior_btn
        )

        print("🏠 내부 탭 클릭 성공")

        time.sleep(5)

        # ==================================
        # 이미지 수집
        # ==================================

        imgs = driver.find_elements(
            By.CSS_SELECTOR,
            "img[id^='INTERIOR_']"
        )

        print(f"🔍 INTERIOR 이미지 수: {len(imgs)}")

        image_urls = []

        for img in imgs:

            src = img.get_attribute("src")

            if src and src not in image_urls:

                image_urls.append(src)

            if len(image_urls) >= 10:

                break

        if len(image_urls) == 0:

            print("⚠️ 내부 이미지 없음")

            driver.switch_to.default_content()

            return "NO_INTERIOR"

        # ==================================
        # 저장
        # ==================================

        for idx, url in enumerate(image_urls, start=1):

            filename = f"{cafe_id}_{idx}.jpg"

            save_path = SAVE_DIR / filename

            urllib.request.urlretrieve(
                url,
                str(save_path)
            )

            print(f"💾 저장 완료: {filename}")

        driver.switch_to.default_content()

        return "SUCCESS"

    except Exception as e:

        print(f"❌ [{cafe_name}] 실패")
        print(type(e).__name__)
        print(e)

        try:
            driver.switch_to.default_content()
        except:
            pass

        return "ERROR"


# ==========================================
# 메인
# ==========================================

if __name__ == "__main__":

    cafes = get_target_cafes()

    print(f"🔥 총 {len(cafes)}개 카페 시작")

    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")

    driver = uc.Chrome(
        options=options
    )

    print("✅ 크롬 생성 완료")

    success = 0
    no_interior = 0
    error = 0

    try:

        for cafe_id, cafe_name, map_url in cafes:

            result = crawl_and_save(
                driver,
                cafe_id,
                cafe_name,
                map_url
            )

            if result == "SUCCESS":

                success += 1

            elif result == "NO_INTERIOR":

                no_interior += 1

            else:

                error += 1

            time.sleep(3)

    finally:

        try:
            driver.quit()
        except:
            pass

    print()
    print("====================")
    print(f"성공: {success}")
    print(f"내부 없음: {no_interior}")
    print(f"에러: {error}")
    print("====================")