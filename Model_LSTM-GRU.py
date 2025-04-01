import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import squarify
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from vnstock import Vnstock

# Vì Streamlit dùng cơ chế vẽ inline, ta import pyplot ở chế độ "inline" bằng matplotlib:
plt.switch_backend('Agg')

# Tắt cảnh báo
import warnings
warnings.filterwarnings('ignore')

#========================
# 1) ĐỊNH NGHĨA CÁC HÀM, MÔ HÌNH
#========================

def fetch_stock_data(ticker, start_date, end_date):
    """Tải dữ liệu giá đóng cửa, trả về DataFrame gồm cột 'close' và index = 'time'."""
    try:
        dt = Vnstock().stock(symbol=ticker, source='VCI').quote.history(start=start_date, end=end_date)
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
        # y_pred shape (1, 10)
        data_normalized = self.data / (self.data[0] + K.epsilon())  # shape (T,10)
        portfolio_values = tf.reduce_sum(data_normalized * y_pred[0], axis=1)
        pvals_shift = portfolio_values[:-1]
        pvals_curr  = portfolio_values[1:]
        daily_ret = (pvals_curr - pvals_shift) / (pvals_shift + K.epsilon())
        mean_r = K.mean(daily_ret)
        std_r  = K.std(daily_ret) + K.epsilon()
        rf = 0.016
        rf_period = rf / 252
        sharpe = (mean_r - rf_period) / std_r
        return -sharpe  # maximize sharpe => minimize -sharpe

def build_lstm_gru_model(timesteps, n_assets):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(timesteps, n_assets)))
    model.add(GRU(256, return_sequences=False))
    model.add(Dense(n_assets, activation='softmax'))  # softmax => trọng số dương, sum=1
    return model

def port_char(weights_df, returns_df):
    """
    Tính (Er, std_dev) cho danh mục.
    - weights_df: DataFrame gồm ['Asset','Weight'].
    - returns_df: DataFrame gồm các cột là tên Asset, giá trị là returns (theo ngày/tuần...).
    """
    Er = returns_df.mean().reset_index()
    Er.columns = ['Asset','Er']
    weights_merged = pd.merge(weights_df, Er, on='Asset', how='left')
    weights_merged['Er'].fillna(0, inplace=True)

    portfolio_er = np.dot(weights_merged['Weight'], weights_merged['Er'])

    cov_matrix = returns_df.cov()
    asset_order = weights_merged['Asset']
    cov_matrix = cov_matrix.loc[asset_order, asset_order]

    w = weights_merged['Weight'].values
    portfolio_std_dev = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))

    return portfolio_er, portfolio_std_dev

def sharpe_port(weights_df, returns_df, rf=0.016, freq=252):
    """
    Tính Sharpe Ratio cho danh mục (trên cùng tần suất với returns_df).
    - weights_df: DataFrame ['Asset','Weight']
    - returns_df: DataFrame cột là Asset, hàng là thời gian, giá trị = returns
    - rf: lãi suất phi rủi ro (mặc định 1.6%/năm)
    - freq: số kỳ trong 1 năm (daily ~252, weekly ~52, v.v.)
    """
    portfolio_er, portfolio_std_dev = port_char(weights_df, returns_df)
    rf_period = rf / freq
    sharpe_ratio = (portfolio_er - rf_period) / (portfolio_std_dev + 1e-12)
    return sharpe_ratio


#========================
# 2) CODE STREAMLIT
#========================

def main():
    st.title("Phân tích Sharpe Ratio và xây dựng danh mục đầu tư bằng LSTM-GRU")

    st.markdown("""
    Ứng dụng này minh họa cách tính toán Sharpe Ratio cho các cổ phiếu ngành Xây dựng, 
    chọn Top 10 theo Sharpe cao nhất, rồi dùng mô hình LSTM-GRU tối ưu danh mục. 
    Ngoài ra, ta so sánh với hai phương pháp phân bổ truyền thống (Allo_1, Allo_2).
    """)

    # Lựa chọn ngày bắt đầu, ngày kết thúc
    default_start = "2018-01-01"
    default_end   = "2024-12-31"

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("Ngày bắt đầu (YYYY-MM-DD):", value=default_start)
    with col2:
        end_date = st.text_input("Ngày kết thúc (YYYY-MM-DD):", value=default_end)

    # Hiện ta chỉ minh họa 1 ngành "Xây dựng"
    # Ta thêm 1 selectbox phòng khi mở rộng sau này
    industry = st.selectbox("Chọn ngành:", ["Xây dựng"], index=0)

    if st.button("Tính toán"):
        st.write("**Bắt đầu tải dữ liệu và xử lý...**")

        # 1) Lấy danh sách mã ngành Xây dựng
        stock = Vnstock().stock(symbol='VN30F1M', source='VCI')
        list_icb = stock.listing.symbols_by_industries()
        list_ticker = list_icb[list_icb['icb_name4']==industry]['symbol'].to_list()

        list_exchange = stock.listing.symbols_by_exchange()[['symbol','type','exchange']]
        df_filtered = list_exchange[
            list_exchange['symbol'].isin(list_ticker) &
            (
                (list_exchange['exchange'] == 'HSX') | (list_exchange['exchange'] == 'HNX')
            )
        ]
        list_ticker = df_filtered['symbol'].to_list()

        # 2) Tải dữ liệu cho tất cả các mã
        all_data = {}
        for ticker in list_ticker:
            df_ = fetch_stock_data(ticker, default_start, default_date)
            if df_ is not None and not df_.empty:
                all_data[ticker] = df_

        if len(all_data)==0:
            st.error("Không tải được dữ liệu cổ phiếu cho ngành này. Vui lòng thử lại.")
            return

        combined_df = pd.concat(all_data.values(), axis=0)
        combined_df.reset_index(inplace=True)

        pivot_df = combined_df.pivot(index="time", columns="ticker", values="close")
        pivot_df.sort_index(inplace=True)
        pivot_df.fillna(0, inplace=True)

        # Tính daily returns
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
        })

        # Sắp xếp theo Sharpe ratio giảm dần
        df_sharpe.sort_values(by='Sharpe Ratio', ascending=False, inplace=True)

        # Lấy Top 10
        top_10 = df_sharpe.head(10)
        top_10_symbols = top_10.index.tolist()

        # Lọc pivot_df chỉ gồm Top 10
        pivot_top10_df = pivot_df[top_10_symbols]

        # Tách train/test theo năm
        train_data = pivot_top10_df.loc[pivot_top10_df.index.year < 2024]
        test_data  = pivot_top10_df.loc[pivot_top10_df.index.year == 2024]

        train_data.rename_axis(None, axis='columns', inplace=True)
        train_data = train_data.reset_index()
        train_data.drop(columns=['time'], inplace=True)

        test_data.rename_axis(None, axis='columns', inplace=True)
        test_data = test_data.reset_index()
        test_data.drop(columns=['time'], inplace=True)

        #===================================
        # Bắt đầu mô hình LSTM-GRU tối ưu Sharpe
        st.write("**Huấn luyện mô hình LSTM-GRU...**")

        class SharpeLossModel:
            def __init__(self, data):
                self.data = tf.constant(data.values, dtype=tf.float32)
            def sharpe_loss(self, _, y_pred):
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
            model = Sequential()
            model.add(LSTM(256, return_sequences=True, input_shape=(timesteps, n_assets)))
            model.add(GRU(256, return_sequences=False))
            model.add(Dense(n_assets, activation='softmax'))
            return model

        X_train = train_data.values[np.newaxis, :, :]
        y_train = np.zeros((1, len(train_data.columns)))

        sharpe_model = SharpeLossModel(train_data)
        model_lstm_gru = build_lstm_gru_model(train_data.shape[0], train_data.shape[1])
        model_lstm_gru.compile(optimizer='adam', loss=sharpe_model.sharpe_loss)

        model_lstm_gru.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=False, verbose=0)
        weights_lstm_gru = model_lstm_gru.predict(X_train)[0]

        results_LSTM_GRU = pd.DataFrame({'Asset': top_10_symbols, "Weight": weights_lstm_gru})

        st.write("**Phân bổ danh mục từ mô hình LSTM-GRU:**")
        st.dataframe(results_LSTM_GRU.sort_values('Weight', ascending=False))

        # Biểu đồ cột
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_df = results_LSTM_GRU.sort_values('Weight', ascending=False)
        ax.bar(sorted_df['Asset'], sorted_df['Weight'], color='green')
        ax.set_xlabel('Tài sản')
        ax.set_ylabel('Trọng số')
        ax.set_title('Phân bổ tài sản theo mô hình LSTM-GRU')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # Treemap
        square_plot_test = pd.DataFrame({
            'Cổ phiếu': sorted_df['Asset'],
            'Tỷ trọng': sorted_df['Weight']
        })
        square_plot_test['Nhãn'] = square_plot_test['Cổ phiếu'] + '\n' + square_plot_test['Tỷ trọng'].apply(lambda x: f"{x*100:.2f}%")

        fig2, ax2 = plt.subplots(figsize=(10,10))
        colors = ['#91DCEA', '#64CDCC', '#5FBB68', '#F9D23C', '#F9A729', '#FD6F30','#B0E0E6','#FFE4E1','#D8BFD8','#FFB6C1']
        squarify.plot(sizes=square_plot_test['Tỷ trọng'], label=square_plot_test['Nhãn'], color=colors,
                      alpha=.8, edgecolor='black', linewidth=2, text_kwargs={'fontsize':10})
        plt.axis('off')
        plt.title('Biểu đồ phân bố tài sản của danh mục đầu tư (Treemap)')
        st.pyplot(fig2)

        #===================================
        # Tính toán, so sánh với Allo_1 & Allo_2
        st.write("**So sánh với phương pháp phân bổ truyền thống**")

        # Tính daily returns test
        test_data_ = pivot_top10_df.loc[pivot_top10_df.index.year == 2024]
        daily_returns_test = test_data_.pct_change()

        # Allo_1
        Allo_1 = results_LSTM_GRU[['Asset']].copy()
        Allo_1['Weight'] = 1/10

        # Allo_2
        mcp = train_data.columns
        Allo_2_temp = train_data.sum().sort_values(ascending=False).reset_index()
        Allo_2_temp.columns = ['Asset','Er']
        top_count = int(0.2 * len(mcp))
        bottom_count = len(mcp) - top_count
        top_weights = [0.8 / (0.2 * len(mcp))] * top_count
        bottom_weights = [0.2 / (0.8 * len(mcp))] * bottom_count
        Allo_2_temp['Weight'] = top_weights + bottom_weights
        Allo_2 = Allo_2_temp[['Asset','Weight']]

        # Convert daily_returns_test -> cột = Asset
        # daily_returns_test có cột = top_10_symbols
        daily_returns_test.columns.name = None
        daily_returns_test = daily_returns_test.reset_index(drop=True)

        # Tạo DataFrame test_data_rets: cột = [Asset], row = time index
        test_data_rets = pd.DataFrame(daily_returns_test, columns=top_10_symbols)
        test_data_rets.fillna(0, inplace=True)

        # Tính Er, Std_dev
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

        Er_lstm, std_lstm = port_char(results_LSTM_GRU, test_data_rets)
        Er_1, std_1 = port_char(Allo_1, test_data_rets)
        Er_2, std_2 = port_char(Allo_2, test_data_rets)

        shr_lstm = sharpe_port(results_LSTM_GRU, test_data_rets)
        shr_1 = sharpe_port(Allo_1, test_data_rets)
        shr_2 = sharpe_port(Allo_2, test_data_rets)

        table_ = pd.DataFrame({
            'Er':[Er_lstm, Er_1, Er_2],
            'Std_dev':[std_lstm, std_1, std_2],
            'Sharpe':[shr_lstm, shr_1, shr_2]
        })
        table_ = table_.T
        table_ = table_.rename(columns={0:'LSTM_GRU',1:'Allo_1',2:'Allo_2'})
        st.write("**Bảng so sánh 3 danh mục:**")
        st.dataframe(table_)

        # Biểu đồ so sánh
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        categories = table_.columns.values
        er_values = table_.loc['Er'].values
        std_dev_values = table_.loc['Std_dev'].values
        sharpe_values = table_.loc['Sharpe'].values

        x_corr = np.arange(len(categories))
        width = 0.2

        ax3.bar(x_corr - width, er_values, width, label='Er')
        ax3.bar(x_corr, std_dev_values, width, label='Std dev')
        ax3.bar(x_corr + width, sharpe_values, width, label='Sharpe', color='green')
        ax3.set_xticks(x_corr)
        ax3.set_xticklabels(categories)
        ax3.set_ylabel("Giá trị")
        ax3.set_title("Biểu đồ Er - Std_dev - Sharpe của các danh mục")
        ax3.legend()

        st.pyplot(fig3)

        st.success("Hoàn tất quá trình tính toán và trực quan!")

if __name__ == '__main__':
    main()
