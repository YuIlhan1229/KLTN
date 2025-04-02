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

# Vì Streamlit dùng cơ chế vẽ inline, ta import pyplot ở chế độ "inline"
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
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
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
# Chia layout thành 2 cột:
#  - col_logo: cột để đặt logo
#  - col_title: cột để đặt tiêu đề
# Tỷ lệ [1, 4] nghĩa là cột logo chiếm 1 phần, cột tiêu đề chiếm 4 phần.
col_logo, col_title = st.columns([1, 4])

with col_logo:
    # Hiển thị logo, có thể điều chỉnh 'width' để tăng/giảm kích thước ảnh
    st.image("Logo_HUB.png", width=400)

with col_title:
    # Tiêu đề được căn giữa với CSS, tiêu đề 1 và tiêu đề 2 độc lập, có kích thước chữ khác nhau
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="margin-bottom: 0px; color: #0B5394; font-size: 32px;">TRƯỜNG ĐẠI HỌC NGÂN HÀNG THÀNH PHỐ HỒ CHÍ MINH</h1>
            <h2 style="margin-top: 5px; color: #333; font-size: 36px;">Xây dựng danh mục đầu tư tối ưu bằng mô hình LSTM - GRU</h2>
        </div>
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
    except:
        return None

class SharpeLossModel:
    def __init__(self, data):
        # data shape (T, 10)
        self.data = tf.constant(data.values, dtype=tf.float32)

    def sharpe_loss(self, _, y_pred):
        """Mất mát = -Sharpe => mô hình cực đại hoá Sharpe."""
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
    Sau đó, hệ thống tự động tính Sharpe Ratio, chọn Top 10 cổ phiếu, huấn luyện LSTM-GRU.
    """)

    industry = st.selectbox("Chọn ngành:", ["Xây dựng"], index=0)
    
    #========================
    # Nhập khoảng thời gian
    #========================
    default_start = "2018-01-01"
    default_end   = "2024-12-31"
    # Chuyển đổi chuỗi sang date object
    default_start_date = datetime.strptime(default_start, '%Y-%m-%d').date()
    default_end_date = datetime.strptime(default_end, '%Y-%m-%d').date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(":red[Choose start date]", value=default_start_date)
    with col2:
        end_date = st.date_input(":red[Choose end date]", value=default_end_date)

    today = datetime.today().date()

    if start_date and end_date:
        if end_date > today:
            st.error("Lỗi: The end date cannot be later than today.")
        elif start_date <= end_date and (end_date - start_date) > timedelta(weeks=4):
            st.success(f"You have chosen the period from {start_date} to {end_date}")
        else:
            st.error("Lỗi: The end date must be after the start date, and the period must be sufficiently long.")
    
    # Nếu người dùng không thay đổi ngày (vẫn mặc định)
    if start_date == default_start_date and end_date == default_end_date:
        st.info(f"Default date range selected: {default_start} to {default_end}")

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

        st.write("**Cấu trúc dữ liệu:**")
        st.dataframe(combined_df)

        #============================
        # BƯỚC 2: XỬ LÝ DỮ LIỆU
        #============================
        pivot_df = combined_df.pivot(index="time", columns="ticker", values="close")
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
            'Annual Return': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio
        }).sort_values(by='Sharpe Ratio', ascending=False)

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
        st.dataframe(results_LSTM_GRU.sort_values('Weight', ascending=False))

        fig, ax = plt.subplots(figsize=(12, 6))
        sorted_df = results_LSTM_GRU.sort_values('Weight', ascending=False)
        ax.bar(sorted_df['Asset'], sorted_df['Weight'], color='green')
        ax.set_xlabel('Tài sản')
        ax.set_ylabel('Trọng số')
        ax.set_title('Phân bổ tài sản (LSTM-GRU)')
        plt.xticks(rotation=0)
        st.pyplot(fig)

        #============================
        # BƯỚC 5: TÍNH TOÁN VÀ SO SÁNH VỚI 2 PHƯƠNG PHÁP
        #============================
        st.write("**So sánh với 2 phương pháp: Phân bổ đồng đều & 80-20**")

        # Phân bổ đồng đều
        Allo_1 = pd.DataFrame({'Asset': top_10_symbols, 'Weight': [1/len(top_10_symbols)]*len(top_10_symbols)})

        # Chiến lược 80-20
        mcp = train_price.columns
        Allo_2_temp = train_price.sum().sort_values(ascending=False).reset_index()
        Allo_2_temp.columns = ['Asset','Er']
        top_count = int(0.2 * len(mcp))
        bottom_count = len(mcp) - top_count
        top_weights = [0.8 / top_count] * top_count
        bottom_weights = [0.2 / bottom_count] * bottom_count
        Allo_2_temp['Weight'] = top_weights + bottom_weights
        Allo_2 = Allo_2_temp[['Asset','Weight']]

        Er_lstm_gru, std_lstm_gru = port_char(results_LSTM_GRU, test_price)
        Er_1, std_1 = port_char(Allo_1, test_price)
        Er_2, std_2 = port_char(Allo_2, test_price)

        shr_lstm_gru = sharpe_port(results_LSTM_GRU, test_price)
        shr_1 = sharpe_port(Allo_1, test_price)
        shr_2 = sharpe_port(Allo_2, test_price)

        table_ = pd.DataFrame({
            'Expected_return': [Er_lstm_gru, Er_1, Er_2],
            'Standard_deviation': [std_lstm_gru, std_1, std_2],
            'Sharpe_ratio': [shr_lstm_gru, shr_1, shr_2]
        }, index=['LSTM_GRU','Phân bổ đều','80-20'])

        st.write("**Bảng so sánh danh mục trên Test set**")
        st.dataframe(table_.T)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        categories = table_.columns.values
        er_vals = table_['Expected_return'].values
        std_vals = table_['Standard_deviation'].values
        shr_vals = table_['Sharpe_ratio'].values

        x_ = np.arange(len(categories))
        w_ = 0.2

        ax3.bar(x_ - w_, er_vals, w_, label='Expected_return')
        ax3.bar(x_, std_vals, w_, label='Standard_deviation')
        ax3.bar(x_ + w_, shr_vals, w_, label='Sharpe_ratio', color='green')

        ax3.set_xticks(x_)
        ax3.set_xticklabels(categories)
        ax3.set_ylabel("Giá trị")
        ax3.legend()
        ax3.set_title("So sánh Er, Std_dev, Sharpe (Test set)")

        st.pyplot(fig3)
        st.success("Hoàn tất quá trình tính toán & trực quan (có tính năng upload CSV).")

if __name__ == '__main__':
    main()
