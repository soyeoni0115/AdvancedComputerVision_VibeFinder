import torch
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model

##lora 업로드코드######################################

def get_lora_clip_model():
    # 1. Hugging Face에서 기본 CLIP 모델과 프로세서 로드 (CPU 모드 고정)
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    # 2. LoRA 설정을 정의합니다. 
    # CLIP 내부의 텍스트/이미지 인코더의 핵심 레이어(q_proj, v_proj)를 타겟
    lora_config = LoraConfig(
        r=8,                         # LoRA Rank 
        lora_alpha=16,               # Scaling 팩터
        target_modules=["q_proj", "v_proj"],  # CLIP 내부에서 학습할 레이어 지정
        lora_dropout=0.05,
        bias="none",
        modules_to_save=[],          # 추가로 저장할 레이어 (없음)
    )
    
    # 3. 기본 CLIP 모델에 LoRA 레이어를 연결(주입)합니다.
    lora_model = get_peft_model(model, lora_config)
    
    # 4. 학습 가능한 파라미터가 얼마나 줄어들었는지 프린트해 봅니다.
    lora_model.print_trainable_parameters()
    
    return lora_model, processor

if __name__ == "__main__":
    # 코드 작동 테스트
    model, processor = get_lora_clip_model()
    print("CLIP 모델에 LoRA 연결 성공!")