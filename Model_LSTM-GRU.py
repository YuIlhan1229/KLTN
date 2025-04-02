import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import squarify
import random
import base64
from datetime import datetime, timedelta

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from vnstock import Vnstock

st.set_page_config(
    page_title="Applying deep learning to portfolio optimization in the Vietnamese stock market", 
    page_icon="📊"
)

# Vì Streamlit dùng cơ chế vẽ inline, ta chuyển backend của matplotlib
plt.switch_backend('Agg')

#========================
# 0) (Tuỳ chọn) Cố định random seed để so sánh chặt hơn
#========================
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

#========================
# Tắt cảnh báo
#========================
import warnings
warnings.filterwarnings('ignore')

#========================
# 1) ĐỊNH NGHĨA CÁC HÀM, MÔ HÌNH
#========================

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file_obj:
        encoded_string = base64.b64encode(image_file_obj.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string.decode()});
        background-size: cover;
        background-color: rgba(255, 255, 255, 0.7);
        background-blend-mode: overlay;
    }}
    .custom-title {{
        color: #F05454;
    }}
    .stMarkdown, .stText {{
        color: #30475E !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('background.png')

#========================
# Hiển thị logo và tiêu đề
#========================
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("Logo_HUB.png", width=400)
with col_title:
    st.markdown(
        """
        <h1 style="color: #0B5394; text-align: center; font-size: 32px;">TRƯỜNG ĐẠI HỌC NGÂN HÀNG THÀNH PHỐ HỒ CHÍ MINH</h1>
        """,
        unsafe_allow_html=True
    )
st.markdown(
    """
    <h2 style="color: #333; text-align: center; font-size: 40px; margin-top: 10px;">
        Xây dựng danh mục đầu tư tối ưu bằng mô hình LSTM - GRU
    </h2>
    """,
    unsafe_allow_html=True
)

#========================
# Các hàm lấy dữ liệu và xây dựng mô hình
#========================

def fetch_stock_data(ticker, start_date, end_date):
    """Tải dữ liệu giá đóng cửa, trả về DataFrame gồm cột 'close' và index='time'."""
    try:
        dt = Vnstock().stock(symbol=ticker, source='VCI').quote.history(
            start=start_date, 
            end=end_date
        )
        dt['time'] = pd.to_datetime(dt['time'])
        dt.set_index('time', inplace=True)
        dt = dt[['close']].copy()
        dt['ticker'] = ticker
        return dt
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu cho {ticker}: {e}")
        return None

class SharpeLossModel:
    def __init__(self, data):
        # data shape (T, n_assets)
        self.data = tf.constant(data.values, dtype=tf.float32)

    def sharpe_loss(self, _, y_pred):
        """Mất mát = -Sharpe => mục tiêu là tối đa hóa Sharpe Ratio."""
        data_normalized = self.data / (self.data[0] + K.epsilon())
        portfolio_values = tf.reduce_sum(data_normalized * y_pred[0], axis=1)
        pvals_shift = portfolio_values[:-1]
        pvals_curr  = portfolio_values[1:]
        daily_ret = (pvals_curr - pvals_shift) / (pvals_shift + K.epsilon())

        mean_r = K.mean(daily_ret)
        std_r  = K.std(daily_ret) + K.epsilon()

        rf = 0.016
        rf_period = rf / 252

        sharpe = (mean_r - rf_period) / std_r
        return -sharpe

def build_lstm_gru_model(timesteps, n_assets):
    """Xây dựng mô hình LSTM + GRU + Dense(softmax)."""
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(timesteps, n_assets)))
    model.add(GRU(256, return_sequences=False))
    model.add(Dense(n_assets, activation='softmax'))
    return model

def port_char(weights_df, returns_df):
    """
    Tính kỳ vọng lợi nhuận (Er) và độ lệch chuẩn (std_dev) của danh mục.
    - weights_df: DataFrame gồm ['Asset','Weight'].
    - returns_df: DataFrame các cột là tên Asset, giá trị là returns.
    """
    Er_ = returns_df.mean().reset_index()
    Er_.columns = ['Asset','Er']
    weights_merged = pd.merge(weights_df, Er_, on='Asset', how='left')
    weights_merged['Er'].fillna(0, inplace=True)
    portfolio_er = np.dot(weights_merged['Weight'], weights_merged['Er'])
    cov_matrix = returns_df.cov()
    asset_order = weights_merged['Asset']
    cov_matrix = cov_matrix.loc[asset_order, asset_order]
    w = weights_merged['Weight'].values
    portfolio_std_dev = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    return portfolio_er, portfolio_std_dev

def sharpe_port(weights_df, returns_df, rf=0.016, freq=252):
    portfolio_er, portfolio_std_dev = port_char(weights_df, returns_df)
    rf_period = rf / freq
    sharpe_ratio_ = (portfolio_er - rf_period) / (portfolio_std_dev + 1e-12)
    return sharpe_ratio_

#========================
# 2) CODE STREAMLIT
#========================

def main():
    st.markdown("""
    Ứng dụng này có hai tùy chọn:
    1. Tải lên file CSV có dữ liệu 'time', 'ticker', 'close'.
    2. Tự động tải dữ liệu từ `vnstock` (nếu không upload).
    """)
    industries = [
        'Bán lẻ phức hợp',
        'Bảo hiểm nhân thọ',
        'Bảo hiểm phi nhân thọ',
        'Bất động sản',
        'Chuyển phát nhanh',
        'Chăm sóc y tế',
        'Chất thải & Môi trường',
        'Containers & Đóng gói',
        'Công nghiệp phức hợp',
        'Công nghệ sinh học',
        'Dược phẩm',
        'Dịch vụ Máy tính',
        'Dịch vụ giải trí',
        'Dịch vụ tiêu dùng chuyên ngành',
        'Dịch vụ truyền thông',
        'Dịch vụ vận tải',
        'Dụng cụ y tế',
        'Giải trí & Truyền thông',
        'Giầy dép',
        'Hàng May mặc',
        'Hàng cá nhân',
        'Hàng không',
        'Hàng điện & điện tử',
        'Internet',
        'Khai khoáng',
        'Khai thác Than',
        'Khai thác vàng',
        'Kho bãi, hậu cần và bảo dưỡng',
        'Khách sạn',
        'Kim Loại màu',
        'Lâm sản và Chế biến gỗ',
        'Lốp xe',
        'Máy công nghiệp',
        'Môi giới chứng khoán',
        'Ngân hàng',
        'Nhà cung cấp thiết bị',
        'Nhà hàng và quán bar',
        'Nhôm',
        'Nhựa, cao su & sợi',
        'Nuôi trồng nông & hải sản',
        'Nước',
        'Phân phối dược phẩm',
        'Phân phối hàng chuyên dụng',
        'Phân phối thực phẩm',
        'Phân phối xăng dầu & khí đốt',
        'Phần cứng',
        'Phần mềm',
        'Phụ tùng ô tô',
        'Quản lý tài sản',
        'Sách, ấn bản & sản phẩm văn hóa',
        'Sản phẩm hóa dầu, Nông dược & Hóa chất khác',
        'Sản xuất & Phân phối Điện',
        'Sản xuất bia',
        'Sản xuất giấy',
        'Sản xuất và Khai thác dầu khí',
        'Sản xuất ô tô',
        'Thiết bị gia dụng',
        'Thiết bị viễn thông',
        'Thiết bị và Dịch vụ Dầu khí',
        'Thiết bị văn phòng',
        'Thiết bị y tế',
        'Thiết bị điện',
        'Thuốc lá',
        'Thép và sản phẩm thép',
        'Thực phẩm',
        'Tiện ích khác',
        'Tài chính cá nhân',
        'Tài chính đặc biệt',
        'Tái bảo hiểm',
        'Tư Vấn, Định giá, Môi giới Bất động sản',
        'Tư vấn & Hỗ trợ KD',
        'Vang & Rượu mạnh',
        'Viễn thông cố định',
        'Viễn thông di động',
        'Vận tải Thủy',
        'Vận tải hành khách & Du lịch',
        'Vật liệu xây dựng & Nội thất',
        'Xe tải & Đóng tàu',
        'Xây dựng',
        'Điện tử tiêu dùng',
        'Đào tạo & Việc làm',
        'Đường sắt',
        'Đồ chơi',
        'Đồ gia dụng lâu bền',
        'Đồ gia dụng một lần',
        'Đồ uống & giải khát'
    ]

    industry = st.selectbox("Chọn ngành:", industries, index=industries.index("Xây dựng"))

    
    #========================
    # Nhập khoảng thời gian
    #========================
    default_start = "2018-01-01"
    default_end   = "2024-12-31"
    default_start_date = datetime.strptime(default_start, '%Y-%m-%d').date()
    default_end_date = datetime.strptime(default_end, '%Y-%m-%d').date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("[Choose start date]", value=default_start_date)
    with col2:
        end_date = st.date_input("[Choose end date]", value=default_end_date)

    today = datetime.today().date()

    if start_date and end_date:
        if end_date > today:
            st.error("Lỗi: The end date cannot be later than today.")
        else:
            if start_date <= end_date and (end_date - start_date) > timedelta(weeks=4):
                st.success(f"You have chosen the period from {start_date} to {end_date}")
            elif start_date == default_start_date and end_date == default_end_date:
                st.info(f"Default date range selected: {default_start} to {default_end}")
            else:
                st.error("Lỗi: The end date must be after the start date, and the period must be sufficiently long.")

    # Sử dụng giá trị ngày dưới dạng chuỗi
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    st.write(f"**Dữ liệu từ {start_date_str} đến {end_date_str}**")
    
    st.write("**Tải lên file CSV (tuỳ chọn):**")
    uploaded_file = st.file_uploader("Chọn file CSV (cấu trúc gồm cột [time, ticker, close])", type=['csv'])

    if st.button("Nhấn để bắt đầu tính toán"):
        st.write("**Bắt đầu lấy dữ liệu & xử lý...**")


        #============================
        # BƯỚC 1: LẤY DỮ LIỆU
        #============================
        if uploaded_file is not None:
            st.success("Đang sử dụng dữ liệu từ file CSV đã upload.")
            combined_df = pd.read_csv(uploaded_file)
            required_cols = {'time','ticker','close'}
            if not required_cols.issubset(combined_df.columns):
                st.error("File CSV thiếu cột bắt buộc. Cần có [time, ticker, close].")
                return
            combined_df['time'] = pd.to_datetime(combined_df['time'])
            combined_df.sort_values('time', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
        else:
            st.info("Không upload file CSV => Tải dữ liệu từ vnstock.")
            stock = Vnstock().stock(symbol='VN30F1M', source='VCI')
            list_icb = stock.listing.symbols_by_industries()
            list_ticker = list_icb[list_icb['icb_name4'] == industry]['symbol'].to_list()

            list_exchange = stock.listing.symbols_by_exchange()[['symbol','type','exchange']]
            df_filtered = list_exchange[
                list_exchange['symbol'].isin(list_ticker) &
                ((list_exchange['exchange'] == 'HSX') | (list_exchange['exchange'] == 'HNX'))
            ]
            list_ticker = df_filtered['symbol'].to_list()

            all_data = {}
            for ticker in list_ticker:
                df_ = fetch_stock_data(ticker, start_date_str, end_date_str)
                if df_ is not None and not df_.empty:
                    all_data[ticker] = df_
            if len(all_data) == 0:
                st.error("Không tải được dữ liệu cổ phiếu nào. Vui lòng thử lại hoặc upload CSV.")
                return
            combined_df = pd.concat(all_data.values(), axis=0).reset_index(drop=True)

        st.write("Các cột của combined_df:", combined_df.columns)
        # Chuẩn hóa tên cột: chuyển về chữ thường và loại bỏ khoảng trắng thừa
        combined_df.columns = combined_df.columns.str.lower().str.strip()

        #============================
        # BƯỚC 2: XỬ LÝ DỮ LIỆU
        #============================
        try:
            pivot_df = combined_df.pivot(index="time", columns="ticker", values="close")
        except KeyError as e:
            st.error(f"Lỗi khi pivot dữ liệu: {e}. Kiểm tra lại tên cột của DataFrame.")
            return

        pivot_df.sort_index(inplace=True)
        pivot_df.fillna(0, inplace=True)

        daily_returns = pivot_df.pct_change()
        mean_daily_returns = daily_returns.mean()
        std_daily_returns  = daily_returns.std()
        days_per_year   = 252
        annual_returns  = mean_daily_returns * days_per_year
        annual_volatility = std_daily_returns * np.sqrt(days_per_year)
        sharpe_ratio = annual_returns / annual_volatility

        df_sharpe = pd.DataFrame({
            'annual return': annual_returns,
            'annual volatility': annual_volatility,
            'sharpe ratio': sharpe_ratio
        }).sort_values(by='sharpe ratio', ascending=False)

        st.write("**Top 10 cổ phiếu theo Sharpe Ratio**")
        top_10 = df_sharpe.head(10)
        st.dataframe(top_10)

        top_10_symbols = top_10.index.tolist()
        pivot_top10_df = pivot_df[top_10_symbols]

        #============================
        # BƯỚC 3: TÁCH TRAIN / TEST
        #============================
        train_price = pivot_top10_df.loc[pivot_top10_df.index.year < 2024]
        test_price  = pivot_top10_df.loc[pivot_top10_df.index.year == 2024]

        train_price = train_price.reset_index(drop=True)
        test_price = test_price.reset_index(drop=True)

        #============================
        # BƯỚC 4: HUẤN LUYỆN MÔ HÌNH LSTM-GRU
        #============================
        X_train = train_price.values[np.newaxis, :, :]
        y_train = np.zeros((1, train_price.shape[1]))

        sharpe_model = SharpeLossModel(pd.DataFrame(train_price))
        model_lstm_gru = build_lstm_gru_model(train_price.shape[0], train_price.shape[1])
        model_lstm_gru.compile(optimizer=Adam(), loss=sharpe_model.sharpe_loss)

        st.write("**Bắt đầu huấn luyện mô hình...** (epochs=100, batch_size=32)")
        model_lstm_gru.fit(X_train, y_train, epochs=100, batch_size=32, shuffle=False, verbose=1)

        weights_lstm_gru = model_lstm_gru.predict(X_train)[0]
        results_LSTM_GRU = pd.DataFrame({'Asset': top_10_symbols, "Weight": weights_lstm_gru})

        st.write("**Phân bổ danh mục từ mô hình LSTM-GRU:**")
        # Chuyển đổi trọng số thành % và làm tròn
        results_LSTM_GRU['Weight (%)'] = (results_LSTM_GRU['Weight'] * 100).round(2).astype(str) + "%"
        st.dataframe(results_LSTM_GRU[['Asset', 'Weight (%)']].sort_values('Weight (%)', ascending=False))

        # Hiển thị tỷ trọng phân bổ dưới dạng biểu đồ cột
        fig, ax = plt.subplots(figsize=(12, 6))
        sorted_df = results_LSTM_GRU.sort_values('Weight', ascending=False)
        ax.bar(sorted_df['Asset'], sorted_df['Weight']*100, color='green')  # nhân 100 để hiển thị %
        ax.set_xlabel('Tài sản')
        ax.set_ylabel('Trọng số (%)')
        ax.set_title('Phân bổ tài sản (LSTM-GRU)')
        plt.xticks(rotation=0)
        st.pyplot(fig)

        # Hiển thị tỷ trọng phân bổ dưới dạng biểu đồ tròn (pie chart)
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(sorted_df['Weight']*100, labels=sorted_df['Asset'], autopct='%1.1f%%', startangle=90)
        ax_pie.set_title("Tỷ trọng phân bổ (LSTM-GRU) - Pie Chart")
        st.pyplot(fig_pie)

        st.success("Hoàn tất quá trình tính toán & trực quan.")

if __name__ == '__main__':
    main()
