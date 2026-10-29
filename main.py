import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="인공지능 사진 구분",
    page_icon="🔍",
    layout="wide"
)

# 2. 메인 타이틀
st.title("인공지능 사진 구분")
st.caption("빅데이터 기반 주파수 스펙트럼(FFT) 분석을 통해 실제 카메라 사진과 AI 생성 이미지를 판별합니다.")

st.markdown("---")

# 3. 사진이 구별되는 이유 및 예시 그래프 섹션
st.header("💡 사진이 구별되는 이유와 주파수 모양 예시")
st.write("사람의 눈으로는 AI 사진과 실제 사진을 구분하기 어렵지만, 파이썬 코드로 **픽셀 데이터(빅데이터)**를 주파수 영역으로 변환하면 명확한 형태 차이가 나타납니다.")

# 시시각각 비교할 수 있도록 예시 주파수 패턴 생성 함수
def generate_sample_spectrum(pattern_type="real"):
    size = 200
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    if pattern_type == "real":
        # 실제 사진: 중심에서 부드럽게 퍼지는 동그란 원형 패턴
        spectrum = np.exp(-R * 3) + np.random.normal(0, 0.05, (size, size))
    else:
        # AI 사진: 특정 십자가(+) 및 격자 무늬 패턴
        grid = (np.sin(X * 20) ** 2) * (np.cos(Y * 20) ** 2)
        spectrum = np.exp(-R * 2.5) + grid * 0.4 + np.random.normal(0, 0.03, (size, size))
    
    return np.clip(spectrum, 0, 1)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.subheader("📷 실제 카메라 사진")
    st.markdown("""
    - **원인:** 빛이 동그란 렌즈를 통과해 센서(CMOS)에 다다를 때 발생하는 **자연스러운 빛 산란과 무작위 노이즈**
    - **주파수 특징:** 중심(저주파)에서 외곽(고주파)으로 **부드럽고 자연스러운 동그란 원형** 형태를 띱니다.
    """)
    real_sample = generate_sample_spectrum("real")
    fig_real = px.imshow(real_sample, color_continuous_scale="gray", title="[예시] 실제 사진의 동그란 주파수 스펙트럼")
    fig_real.update_layout(coloraxis_showscale=False, height=280, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_real, use_container_width=True)

with col_exp2:
    st.subheader("🤖 인공지능 생성 사진")
    st.markdown("""
    - **원인:** AI 생성 알고리즘(Diffusion 등)이 픽셀을 격자 단위로 연산하고 늘리는 과정에서 남는 **수학적 격자 잔상**
    - **주파수 특징:** 외곽 영역에 **십자가(+) 모양이나 점 형태의 규칙적인 격자 무늬**가 관찰됩니다.
    """)
    ai_sample = generate_sample_spectrum("ai")
    fig_ai = px.imshow(ai_sample, color_continuous_scale="gray", title="[예시] AI 사진의 규칙적인 격자 주파수 스펙트럼")
    fig_ai.update_layout(coloraxis_showscale=False, height=280, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_ai, use_container_width=True)

st.markdown("---")

# 4. 주파수 실시간 분석 및 판별 함수
def analyze_image_spectrum(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 2D 고속 푸리에 변환 (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # 정규화
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 고주파 격자 패턴 검출 (중심 저주파 마스킹)
    h, w = mag_norm.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    
    high_freq_spectrum = mag_norm.copy()
    cv2.circle(high_freq_spectrum, (cx, cy), r, 0, -1)

    # 격자 규칙성 점수 (Grid Score) 계산
    horizontal_profile = np.mean(high_freq_spectrum, axis=0)
    vertical_profile = np.mean(high_freq_spectrum, axis=1)
    
    grid_score = (np.max(horizontal_profile) + np.max(vertical_profile)) / (np.mean(high_freq_spectrum) + 1e-5)
    
    # AI 및 실제 사진 확률 판별 공식
    ai_prob = min(max((grid_score - 2.1) * 35 + 50, 2.0), 98.0)
    real_prob = 100.0 - ai_prob

    return mag_norm, ai_prob, real_prob, grid_score

# 5. 사용자 파일 업로드 및 분석 실행
st.header("🔍 직접 사진 업로드하여 분석하기")
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # 분석 계산
    mag_spectrum, ai_prob, real_prob, grid_score = analyze_image_spectrum(img_array)

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. 입력된 사진")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("2. 추출된 주파수 그래프")
        fig_user = px.imshow(mag_spectrum, color_continuous_scale="gray")
        fig_user.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_user, use_container_width=True)
        st.caption("▲ 위 그래프 형태를 상단 예시 그래프와 비교해 보세요!")

    with col3:
        st.subheader("3. 분석 및 구분 결과")
        
        if ai_prob > 50:
            st.error("🤖 **[인공지능 생성 사진]으로 구분되었습니다.**")
            st.caption("주파수 그래프 외곽에서 규칙적인 격자 무늬 피크 패턴이 감지되었습니다.")
        else:
            st.success("📷 **[실제 카메라 사진]으로 구분되었습니다.**")
            st.caption("주파수 그래프가 동그랗고 자연스럽게 퍼져 있으며 무작위 노이즈 특성을 가집니다.")

        st.progress(int(ai_prob))
        st.write(f"• **인공지능 사진 확률:** `{ai_prob:.1f}%`")
        st.write(f"• **실제 사진 확률:** `{real_prob:.1f}%`")
        st.write(f"• **격자 주기성 수치(Grid Score):** `{grid_score:.2f}`")

else:
    st.info("👆 위 사진 업로드 버튼을 눌러 준비하신 실제 사진이나 AI 사진을 올려보세요.")
