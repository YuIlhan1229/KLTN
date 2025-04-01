import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import squarify
import random

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from vnstock import Vnstock

# Vì Streamlit dùng cơ chế vẽ inline, ta import pyplot ở chế độ "inline" bằng matplotlib:
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
        data_normalized = self.data / (self.data[0] + K.epsilon())  # shape (T,10)
        portfolio_values = tf.reduce_sum(data_normalized * y_pred[0], axis=1)
        pvals_shift = portfolio_values[:-1]
        pvals_curr  = portfolio_values[1:]
        daily_ret = (pvals_curr - pvals_shift) / (pvals_shift + K.epsilon())

        mean_r = K.mean(daily_ret)
        std_r  = K.std(daily_ret) + K.epsilon()

        # Lãi suất phi rủi ro
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
    Tính (Er, std_dev) cho danh mục.
    - weights_df: DataFrame gồm ['Asset','Weight'].
    - returns_df: DataFrame gồm cột = tên Asset, giá trị = returns (theo ngày/tuần...).
    """
    # 1) Lấy Er
    Er_ = returns_df.mean().reset_index()
    Er_.columns = ['Asset','Er']
    
    # 2) Merge
    weights_merged = pd.merge(weights_df, Er_, on='Asset', how='left')
    weights_merged['Er'].fillna(0, inplace=True)

    # 3) Er danh mục
    portfolio_er = np.dot(weights_merged['Weight'], weights_merged['Er'])

    # 4) Ma trận hiệp phương sai
    cov_matrix = returns_df.cov()
    asset_order = weights_merged['Asset']
    cov_matrix = cov_matrix.loc[asset_order, asset_order]

    w = weights_merged['Weight'].values
    portfolio_std_dev = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))

    return portfolio_er, portfolio_std_dev

def sharpe_port(weights_df, returns_df, rf=0.016, freq=252):
    """
    Tính Sharpe Ratio trên cùng tần suất (daily/weekly...) với returns_df.
    """
    portfolio_er, portfolio_std_dev = port_char(weights_df, returns_df)
    rf_period = rf / freq
    sharpe_ratio_ = (portfolio_er - rf_period) / (portfolio_std_dev + 1e-12)
    return sharpe_ratio_

#========================
# 2) CODE STREAMLIT
#========================

def main():
    st.title("Phân tích Sharpe Ratio & Xây dựng Danh mục (Cho phép Upload CSV)")

    st.markdown("""
    Ứng dụng này có hai tùy chọn:
    1. Tải lên file CSV có dữ liệu 'time', 'ticker', 'close'.
    2. Tự động tải dữ liệu từ `vnstock` (nếu không upload).
    Sau đó, ta tính Sharpe Ratio, chọn Top 10 cổ phiếu, huấn luyện LSTM-GRU.
    """)

    industry = st.selectbox("Chọn ngành:", ["Xây dựng"], index=0)
    
    default_start = "2018-01-01"
    default_end   = "2024-12-31"

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("Ngày bắt đầu (YYYY-MM-DD):", value=default_start)
    with col2:
        end_date = st.text_input("Ngày kết thúc (YYYY-MM-DD):", value=default_end)

    # Tính năng Upload CSV
    st.write("**Tải lên file CSV (tuỳ chọn):**")
    uploaded_file = st.file_uploader("Chọn file CSV (cấu trúc gồm cột [time, ticker, close])", type=['csv'])

    if st.button("Tính toán"):
        st.write("**Bắt đầu lấy dữ liệu & xử lý...**")

        #============================
        # BƯỚC 1: LẤY DỮ LIỆU
        #============================

        if uploaded_file is not None:
            # Người dùng đã up file => ta dùng data từ file
            st.success("Đang sử dụng dữ liệu từ file CSV đã upload.")
            combined_df = pd.read_csv(uploaded_file)
            # Kiểm tra cột
            required_cols = {'time','ticker','close'}
            if not required_cols.issubset(combined_df.columns):
                st.error("File CSV thiếu cột bắt buộc. Cần có [time, ticker, close].")
                return
            # Convert time => datetime & sort
            combined_df['time'] = pd.to_datetime(combined_df['time'])
            combined_df.sort_values('time', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)

        else:
            # Không có file => ta lấy dữ liệu từ vnstock
            st.info("Không upload file CSV => Tải dữ liệu từ vnstock.")
            stock = Vnstock().stock(symbol='VN30F1M', source='VCI')
            list_icb = stock.listing.symbols_by_industries()
            list_ticker = list_icb[list_icb['icb_name4'] == industry]['symbol'].to_list()

            list_exchange = stock.listing.symbols_by_exchange()[['symbol','type','exchange']]
            df_filtered = list_exchange[
                list_exchange['symbol'].isin(list_ticker) &
                (
                    (list_exchange['exchange'] == 'HSX') | (list_exchange['exchange'] == 'HNX')
                )
            ]
            list_ticker = df_filtered['symbol'].to_list()

            all_data = {}
            for ticker in list_ticker:
                df_ = fetch_stock_data(ticker, start_date, end_date)
                if df_ is not None and not df_.empty:
                    all_data[ticker] = df_

            if len(all_data) == 0:
                st.error("Không tải được dữ liệu cổ phiếu nào. Vui lòng thử lại hoặc upload CSV.")
                return

            combined_df = pd.concat(all_data.values(), axis=0).reset_index(drop=True)

        st.write("**Dữ liệu sau khi ghép (combined_df):**")
        st.dataframe(combined_df.head(5))

        #============================
        # BƯỚC 2: XỬ LÝ DỮ LIỆU
        #============================

        # pivot_df: index = time, columns = ticker, values = close
        pivot_df = combined_df.pivot(index="time", columns="ticker", values="close")
        pivot_df.sort_index(inplace=True)
        pivot_df.fillna(0, inplace=True)

        # Tính daily_returns
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

        train_price = train_price.reset_index()
        train_price.drop(columns=['time'], inplace=True)

        test_price = test_price.reset_index()
        test_price.drop(columns=['time'], inplace=True)

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

        fig, ax = plt.subplots(figsize=(8, 4))
        sorted_df = results_LSTM_GRU.sort_values('Weight', ascending=False)
        ax.bar(sorted_df['Asset'], sorted_df['Weight'], color='green')
        ax.set_xlabel('Tài sản')
        ax.set_ylabel('Trọng số')
        ax.set_title('Phân bổ tài sản (LSTM-GRU)')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # Treemap
        fig2, ax2 = plt.subplots(figsize=(6,6))
        square_plot_test = pd.DataFrame({
            'Cổ phiếu': sorted_df['Asset'],
            'Tỷ trọng': sorted_df['Weight']
        })
        square_plot_test['Nhãn'] = square_plot_test['Cổ phiếu'] + '\n' + square_plot_test['Tỷ trọng'].apply(lambda x: f"{x*100:.2f}%")

        colors = ['#91DCEA', '#64CDCC', '#5FBB68', '#F9D23C', '#F9A729', '#FD6F30','#B0E0E6','#FFE4E1','#D8BFD8','#FFB6C1']
        squarify.plot(sizes=square_plot_test['Tỷ trọng'], label=square_plot_test['Nhãn'], color=colors,
                      alpha=.8, edgecolor='black', linewidth=2, text_kwargs={'fontsize':10})
        plt.axis('off')
        plt.title('Treemap phân bổ danh mục (LSTM-GRU)')
        st.pyplot(fig2)

        #============================
        # BƯỚC 5: TÍNH TOÁN, SO SÁNH VỚI 2 PHƯƠNG PHÁP
        #============================
        st.write("**So sánh với 2 phương pháp: Phân bổ đều & 80-20**")

        # Allo_1 (phân bổ đều)
        Allo_1 = results_LSTM_GRU[['Asset']].copy()
        Allo_1['Weight'] = 1 / 10

        # Allo_2 (80-20)
        mcp = train_price.columns  # cột = top_10_symbols
        Allo_2_temp = train_price.sum().sort_values(ascending=False).reset_index()
        Allo_2_temp.columns = ['Asset','Er']
        top_count = int(0.2 * len(mcp))
        bottom_count = len(mcp) - top_count

        top_weights = [0.8 / (0.2 * len(mcp))] * top_count
        bottom_weights = [0.2 / (0.8 * len(mcp))] * bottom_count

        Allo_2_temp['Weight'] = top_weights + bottom_weights
        Allo_2 = Allo_2_temp[['Asset','Weight']]

        # Tính Expected_return, Standard_deviation, Sharpe_ratio => test set
        Er_lstm_gru,  std_lstm_gru  = port_char(results_LSTM_GRU, test_price)
        Er_1,     std_1     = port_char(Allo_1,          test_price)
        Er_2,     std_2     = port_char(Allo_2,          test_price)

        shr_lstm_gru = sharpe_port(results_LSTM_GRU, test_price)
        shr_1    = sharpe_port(Allo_1,          test_price)
        shr_2    = sharpe_port(Allo_2,          test_price)

        table_ = pd.DataFrame({
            'Expected_return': [Er_lstm_gru, Er_1, Er_2],
            'Standard_deviation': [std_lstm_gru, std_1, std_2],
            'Sharpe_ratio': [shr_lstm_gru, shr_1, shr_2]
        }, index=['LSTM_GRU','Phân bổ đều','80-20'])

        st.write("**Bảng so sánh danh mục trên Test set**")
        st.dataframe(table_.T)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        categories = table_.columns.values
        er_vals    = table_['Expected_return'].values
        std_vals   = table_['Standard_deviation'].values
        shr_vals   = table_['Sharpe_ratio'].values

        x_ = np.arange(len(categories))
        w_ = 0.2

        ax3.bar(x_ - w_, er_vals,  w_, label='Expected_return')
        ax3.bar(x_,       std_vals, w_, label='Standard_deviation')
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
