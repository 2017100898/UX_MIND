import requests
import streamlit as st

st.set_page_config(
    page_title="",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Create a form in the sidebar for users to input a sentence
with st.sidebar:
    st.title("AIMS XR Dashboard")

    text_input = st.text_input("Enter condition of diffusion 👇", value = "")

    if text_input:
        requests.post("http://192.168.1.172:5000/diffusion_post_cmd", json={"text_input": text_input})


cols = st.columns([1, 1, 1])


card_css = """
<style>
.card {
    border-radius: 15px;
    padding: 20px;
    background-color: #191925;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px; /* 카드 사이 간격 추가 */
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.card-content {
    display: flex;
    flex-direction: column;
    align-items: left;
    text-align: left;
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 20px;
}
</style>
"""

st.markdown(card_css, unsafe_allow_html=True)

with cols[0].container():
    
    st.markdown('''
            <div class="card">
                <div class="card-content">Pose Estimation</div>
                <iframe src="http://192.168.1.172:5000/pose_feed" width="100%" height="500" frameborder="0" scrolling="no">
                </iframe>
            </div>''', unsafe_allow_html=True)

    st.markdown('''
            <div class="card">
                <div class="card-content">Diffusion Model</div>
                    <iframe src="http://192.168.1.172:5000/diffusion_feed" width="100%" height="500" frameborder="0" scrolling="no">
                    </iframe>
            </div>''', unsafe_allow_html=True)

with cols[1].container():
    st.markdown('''
            <div class="card">
                <div class="card-content">EEG</div>
                    <iframe src="http://192.168.1.172:5000/eeg_feed" width="100%" height="1110" frameborder="0" scrolling="no">
                    </iframe>
            </div>''', unsafe_allow_html=True)
    
with cols[2].container():

    st.markdown('''
            <div class="card">
                <div class="card-content">Emotion</div>
                    <iframe src="http://192.168.1.172:5000/emotion_feed" width="100%" height="500" frameborder="0" scrolling="no">
                    </iframe>
            </div>''', unsafe_allow_html=True)
    
    sub_cols = st.columns([1, 1])

    with sub_cols[0].container():
        st.markdown('''
                <div class="card">
                    <div class="card-content">MNE</div>
                        <iframe src="http://192.168.1.172:5000/mne_feed" width="100%" height="500" frameborder="0" scrolling="no">
                        </iframe>
                </div>''', unsafe_allow_html=True)
    

    with sub_cols[1].container():
        st.markdown('''
                <div class="card">
                    <div class="card-content" >Attention</div>
                </div>''', unsafe_allow_html=True)
        
