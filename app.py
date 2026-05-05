import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. 페이지 설정
st.set_page_config(page_title="포켓몬 분류기", layout="centered")
st.title("Pokemon Classifier")

# 2. 설정 및 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = './PokemonData'
MODEL_PATH = './Exp3_MobileNet_FineTune/model.pth' # 가장 결과 좋은 모델 경로

@st.cache_resource # 모델을 한 번만 로드하도록 캐싱
def load_trained_model():
    class_names = sorted(os.listdir(DATA_DIR))
    #model = models.resnet18(pretrained=False) # resnet18 모델로 고정
    model = models.mobilenet_v2(pretrained=False) # mobilenet_v2 모델로 고정
    #model.fc = nn.Linear(model.fc.in_features, len(class_names)) # resnet18의 fully connected layer 수정
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names)) # mobilenet_v2의 classifier 수정

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, class_names

try:
    my_model, class_names = load_trained_model()
except:
    st.error("모델 파일을 찾을 수 없습니다. 경로를 확인해주세요!")

# 3. 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. GUI 구성: 이미지 업로드
uploaded_file = st.file_uploader("포켓몬 이미지를 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 이미지 표시
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_container_width=True)
    st.write("분류 중...")

    # 예측 수행
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = my_model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence = probs[predicted[0]] * 100
        result_label = class_names[predicted[0]]

    # 결과 출력
    st.success(f"이 포켓몬은 **{result_label}** 입니다!")
    st.info(f" 확신도: {confidence:.2f}%")