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
    st.title("Phân tích Sharpe Ratio & Xây dựng Danh mục (Phiên bản đồng nhất code cũ)")

    st.markdown("""
    Ứng dụng này tính toán Sharpe Ratio cho các cổ phiếu ngành Xây dựng (lấy Top 10), 
    rồi huấn luyện mô hình LSTM-GRU để tối ưu danh mục. 
    Sau đó, so sánh với hai phương pháp phân bổ (Phân bổ đều, 80-20) 
    theo đúng trình tự giống code cũ, nhằm hạn chế sai lệch kết quả.
    """)

    default_start = "2018-01-01"
    default_end   = "2024-12-31"

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("Ngày bắt đầu (YYYY-MM-DD):", value=default_start)
    with col2:
        end_date = st.text_input("Ngày kết thúc (YYYY-MM-DD):", value=default_end)

    industry = st.selectbox("Chọn ngành:", ["Xây dựng"], index=0)

    if st.button("Tính toán"):
        st.write("**Bắt đầu tải dữ liệu...**")

        # 1) Lấy danh sách mã cổ phiếu
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
            st.error("Không tải được dữ liệu cổ phiếu nào. Vui lòng thử lại.")
            return

        # 2) Ghép dữ liệu
        combined_df = pd.concat(all_data.values(), axis=0)
        combined_df.reset_index(inplace=True)

        pivot_df = combined_df.pivot(index="time", columns="ticker", values="close")
        pivot_df.sort_index(inplace=True)
        pivot_df.fillna(0, inplace=True)  # giống code cũ => fillna=0

        # 3) Tính daily_returns (trước khi chia train/test)
        daily_returns = pivot_df.pct_change()
        # Tính mean, std
        mean_daily_returns = daily_returns.mean()
        std_daily_returns  = daily_returns.std()

        days_per_year      = 252
        annual_returns     = mean_daily_returns * days_per_year
        annual_volatility  = std_daily_returns * np.sqrt(days_per_year)
        sharpe_ratio       = annual_returns / annual_volatility

        df_sharpe = pd.DataFrame({
            'Annual Return': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio
        })

        df_sharpe.sort_values(by='Sharpe Ratio', ascending=False, inplace=True)

        # 4) Lấy Top 10
        top_10 = df_sharpe.head(10)
        top_10_symbols = top_10.index.tolist()
        pivot_top10_df = pivot_df[top_10_symbols]

        st.write("**Top 10 mã cổ phiếu & Sharpe**")
        st.dataframe(top_10)

        # 5) Tách train/test (vẫn theo code cũ: <2024 => train, ==2024 => test)
        train_price = pivot_top10_df.loc[pivot_top10_df.index.year < 2024]
        test_price  = pivot_top10_df.loc[pivot_top10_df.index.year == 2024]

        # Tương tự code cũ => Convert sang DataFrame, drop cột time
        train_price = train_price.reset_index()
        train_price.drop(columns=['time'], inplace=True)

        test_price = test_price.reset_index()
        test_price.drop(columns=['time'], inplace=True)

        # 6) Xây dựng mô hình LSTM-GRU như code cũ
        st.write("**Bắt đầu huấn luyện mô hình LSTM-GRU (giống code cũ)**")

        # Tạo X_train
        X_train = train_price.values[np.newaxis, :, :]  # shape(1, T, 10)
        y_train = np.zeros((1, train_price.shape[1]))   # shape(1, 10)

        sharpe_model = SharpeLossModel(pd.DataFrame(train_price))  # code cũ => dataFrame

        model_lstm_gru = build_lstm_gru_model(train_price.shape[0], train_price.shape[1])
        model_lstm_gru.compile(optimizer=Adam(), loss=sharpe_model.sharpe_loss)

        # Huấn luyện => epochs=100 (giống code cũ) hoặc tuỳ
        model_lstm_gru.fit(X_train, y_train, epochs=100, batch_size=32, shuffle=False, verbose=1)

        # Dự đoán ra weights
        weights_lstm_gru = model_lstm_gru.predict(X_train)[0]
        results_LSTM_GRU = pd.DataFrame({'Asset':top_10_symbols, "Weight":weights_lstm_gru})

        st.write("**Phân bổ danh mục từ LSTM-GRU**")
        st.dataframe(results_LSTM_GRU.sort_values('Weight', ascending=False))

        # Vẽ biểu đồ cột
        fig, ax = plt.subplots(figsize=(8, 4))
        sorted_df = results_LSTM_GRU.sort_values('Weight', ascending=False)
        ax.bar(sorted_df['Asset'], sorted_df['Weight'], color='green')
        ax.set_xlabel('Tài sản')
        ax.set_ylabel('Trọng số')
        ax.set_title('Phân bổ tài sản theo mô hình LSTM-GRU')
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
        plt.title('Biểu đồ phân bố tài sản (Treemap)')
        st.pyplot(fig2)

        # 7) Tạo daily_returns train & test (để so sánh Allo_1, Allo_2)
        #    Code cũ => daily_returns chung, ta tách 2 phần y hệt:
        daily_returns_top10 = daily_returns[top_10_symbols].copy()
        train_rets = daily_returns_top10.loc[daily_returns_top10.index.year < 2024]
        test_rets  = daily_returns_top10.loc[daily_returns_top10.index.year == 2024]

        # fillna(0)
        train_rets.fillna(0, inplace=True)
        test_rets.fillna(0, inplace=True)

        # Tính 2 phương pháp Allo_1, Allo_2 => tương tự code cũ
        st.write("**So sánh với 2 chiến lược: Phân bổ đều & 80-20**")

        Allo_1 = results_LSTM_GRU[['Asset']].copy()
        Allo_1['Weight'] = 1.0 / 10.0

        # Allo_2: Tính sum train_price, xếp hạng
        mcp = train_price.columns
        # Bỏ cột 'Asset' => cẩn thận: mcp = [time?? => tuỳ] => check:
        # Thường: train_price cột = top_10_symbols => time?
        # Tách 'time' ra hay check?
        # Ta debug: ta drop time column => ok, bây giờ mcp= list of top10 ticker

        Allo_2_temp = train_price.sum().sort_values(ascending=False).reset_index()
        Allo_2_temp.columns = ['Asset','Er']

        top_count = int(0.2 * len(mcp))  # top 20%
        bottom_count = len(mcp) - top_count

        top_weights = [0.8 / (0.2 * len(mcp))] * top_count
        bottom_weights = [0.2 / (0.8 * len(mcp))] * bottom_count

        Allo_2_temp['Weight'] = top_weights + bottom_weights
        Allo_2 = Allo_2_temp[['Asset','Weight']]

        # Tính Er, std_dev
        Er_lstm, std_lstm = port_char(results_LSTM_GRU, test_rets)
        Er_1, std_1 = port_char(Allo_1, test_rets)
        Er_2, std_2 = port_char(Allo_2, test_rets)

        shr_lstm = sharpe_port(results_LSTM_GRU, test_rets)
        shr_1    = sharpe_port(Allo_1, test_rets)
        shr_2    = sharpe_port(Allo_2, test_rets)

        table_ = pd.DataFrame({
            'Er': [Er_lstm, Er_1, Er_2],
            'Std_dev': [std_lstm, std_1, std_2],
            'Sharpe': [shr_lstm, shr_1, shr_2]
        }, index=['LSTM_GRU','Phân bổ đều','Phân bổ 80-20'])

        st.write("**Bảng so sánh 3 danh mục** (test set) :")
        st.dataframe(table_.T)

        # Biểu đồ so sánh
        fig3, ax3 = plt.subplots(figsize=(8, 4))

        categories = table_.index.values
        er_values = table_['Er'].values
        std_values = table_['Std_dev'].values
        sharpe_values = table_['Sharpe'].values

        x_corr = np.arange(len(categories))
        width = 0.2

        ax3.bar(x_corr - width, er_values, width, label='Er')
        ax3.bar(x_corr, std_values, width, label='Std dev')
        ax3.bar(x_corr + width, sharpe_values, width, label='Sharpe', color='green')
        ax3.set_xticks(x_corr)
        ax3.set_xticklabels(categories)
        ax3.set_ylabel("Giá trị")
        ax3.set_title("So sánh Er, Std_dev, Sharpe các danh mục (Test set)")
        ax3.legend()

        st.pyplot(fig3)

        st.success("Hoàn tất - Code Streamlit đã đồng bộ logic với code cũ!")

        #========================
        # Thêm phần in ra shape, head, tail để so sánh
        #========================
        with st.expander("So sánh data frames (Kiểm tra)"):
            st.write("**pivot_df shape:**", pivot_df.shape)
            st.write("**train_price shape:**", train_price.shape)
            st.write("**test_price shape:**", test_price.shape)
            st.write("**daily_returns_top10 shape:**", daily_returns_top10.shape)
            st.write("**train_rets shape:**", train_rets.shape)
            st.write("**test_rets shape:**", test_rets.shape)

            st.write("**train_price HEAD**")
            st.dataframe(train_price.head(3))
            st.write("**train_price TAIL**")
            st.dataframe(train_price.tail(3))

            st.write("**daily_returns_top10 HEAD**")
            st.dataframe(daily_returns_top10.head(3))

            st.write("**Kết quả LSTM_GRU**")
            st.dataframe(results_LSTM_GRU)

if __name__ == '__main__':
    main()
