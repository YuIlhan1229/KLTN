import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import squarify
import random
import base64
import logging
import time
from datetime import datetime, timedelta

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from vnstock import Vnstock

# Cấu hình logging
logging.basicConfig(
    filename='app_log.txt',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

st.set_page_config(
    page_title="Applying deep learning to portfolio optimization in the Vietnamese stock market", 
    page_icon="📊"
)

plt.switch_backend('Agg')

SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

import warnings
warnings.filterwarnings('ignore')

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file_obj:
        encoded_string = base64.b64encode(image_file_obj.read())
    css_code = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string.decode()}); 
        background-size: cover; 
        background-color: rgba(255, 255, 255, 0.8); 
        background-blend-mode: overlay; 
    }}
    .custom-title {{
        color: #F05454;
    }}
    .stMarkdown, .stText {{
        color: #30475E !important;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

def main():
    start_time = time.time()
    logging.info("🚀 Người dùng bắt đầu chạy ứng dụng.")

    st.title("📈 Ứng dụng Demo Tối ưu danh mục - LSTM-GRU (có ghi log)")
    st.markdown("Ứng dụng này ghi lại toàn bộ quá trình xử lý vào `app_log.txt` để theo dõi hiệu suất.")

    if st.button("Bắt đầu demo ghi log"):
        logging.info("🟢 Người dùng nhấn nút bắt đầu demo.")
        time.sleep(2)
        logging.info("🧠 Mô phỏng xử lý mô hình xong.")

    end_time = time.time()
    logging.info(f"🎯 Ứng dụng hoàn thành sau {end_time - start_time:.2f} giây.")

    # Hiển thị log
    with st.expander("📄 Xem log hệ thống"):
        try:
            with open("app_log.txt", "r") as f:
                log_content = f.read()
            st.text_area("📋 Log nội bộ:", value=log_content, height=300)
        except FileNotFoundError:
            st.info("Chưa có log nào được ghi.")

if __name__ == '__main__':
    main()
