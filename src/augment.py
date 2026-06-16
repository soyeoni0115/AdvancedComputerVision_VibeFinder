import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

INPUT_DIR = "../data/processed/train"
OUTPUT_DIR = "../data/train_aug"
AUG_PER_IMAGE = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

#augment = transforms.Compose([
#    transforms.Resize((256,256)),   # 먼저 키우고
#    transforms.RandomCrop(224),     # 그 다음 crop
#    transforms.RandomHorizontalFlip(),
#    transforms.ColorJitter(0.1,0.1,0.1,0.02),  # 약하게
#])
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1,0.02),
    transforms.RandomRotation(5),
])

for cafe in os.listdir(INPUT_DIR):
    src_path = os.path.join(INPUT_DIR, cafe)
    dst_path = os.path.join(OUTPUT_DIR, cafe)
    os.makedirs(dst_path, exist_ok=True)

    for img_name in tqdm(os.listdir(src_path), desc=f"Aug {cafe}"):
        img_path = os.path.join(src_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        base = img_name.split(".")[0]

        # 원본
        img.save(os.path.join(dst_path, f"{base}.jpg"))

        # 증강
        for i in range(AUG_PER_IMAGE):
            aug_img = augment(img)
            aug_img.save(os.path.join(dst_path, f"{base}_aug{i}.jpg"))

print("✅ 증강 완료")