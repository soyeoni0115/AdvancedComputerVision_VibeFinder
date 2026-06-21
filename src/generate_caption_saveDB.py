import os
import psycopg2
from PIL import Image
from tqdm import tqdm
from database.postgres_new import DATABASE_URL
from transformers import BlipProcessor, BlipForConditionalGeneration

# ===== DB 연결 =====
conn = psycopg2.connect(
DATABASE_URL)
cur = conn.cursor()

# ===== BLIP 모델 로드 =====
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

cur.execute("TRUNCATE TABLE cafe_images RESTART IDENTITY;")
conn.commit()

# ===== 이미지 폴더 =====
IMAGE_DIR = "../data/raw"

#===스마트캡션==========
def smart_caption(blip_caption):
    caption = blip_caption.lower()
    mood_en = "seongsu cafe"
    mood_kr = "성수, 서울숲, 카페"

    if "coffee" in caption or "cup" in caption:
        mood_en = "cozy cafe, relaxing place"
        mood_kr = "여유롭고 감성적인 카페"

    elif "table" in caption or "desk" in caption:
        mood_en = "good for studying, quiet place"
        mood_kr = "공부하기 좋은 공간"

    elif "cake" in caption or "dessert" in caption:
        mood_en = "dessert cafe, nice atmosphere"
        mood_kr = "디저트가 맛있는 분위기 좋은 카페"

    elif "window" in caption or "light" in caption:
        mood_en = "bright and airy cafe"
        mood_kr = "채광 좋은 밝은 카페"

    elif "office" in caption or "conference" in caption:
        mood_en = "good at studying, conference"
        mood_kr = "공부하기 좋은"

    elif "plant" in caption or "green" in caption:
        mood_en = "nature vibe cafe"
        mood_kr = "식물이 많은 자연 느낌 카페"

    elif "bunch" in caption or "several" in caption:
        mood_en = "large cafe"
        mood_kr= "대형 카페"

    

    return f"{blip_caption}, {mood_en}, {mood_kr}"

# ===== 메인 루프 =====
for i, img_name in enumerate(tqdm(os.listdir(IMAGE_DIR))):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)

    # cafe_id 추출 (정수로 변환)
    cafe_id = int(img_name.split("_")[0])

    try:
        image = Image.open(img_path).convert("RGB")
    except:
        continue

    # ===== BLIP caption 생성 =====
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # caption 보정
    #caption = "cafe interior, " + caption   -> 버전1
    caption = smart_caption(caption)

    # ===== 중복 체크 =====
    cur.execute("""
        SELECT 1 FROM cafe_images WHERE image_path = %s
    """, (img_path,))
    
    if cur.fetchone():
        continue

    # ===== INSERT =====
    cur.execute("""
        INSERT INTO cafe_images (cafe_id, image_path, caption)
        VALUES (%s, %s, %s)
    """, (cafe_id, img_path, caption))

    # ===== 10개마다 commit =====
    if i % 10 == 0:
        conn.commit()

# 마지막 commit
conn.commit()

cur.close()
conn.close()

print("caption 생성 + DB 저장 완료")