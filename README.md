# KLTN
# 📊 Ứng dụng Tối ưu danh mục đầu tư bằng LSTM-GRU trên thị trường chứng khoán Việt Nam

Đây là một ứng dụng web được xây dựng bằng **Streamlit**, áp dụng **Deep Learning (LSTM + GRU)** để tối ưu hóa danh mục đầu tư trên thị trường chứng khoán Việt Nam.

## 🚀 Tính năng chính

- Cho phép người dùng chọn **ngành** trong thị trường chứng khoán Việt Nam
- Lấy dữ liệu từ API **vnstock** hoặc cho phép upload file CSV tùy chỉnh
- Tính toán các chỉ số tài chính cơ bản: **Tỷ suất sinh lời**, **Độ biến động**, **Sharpe Ratio**
- Tự động lọc **Top 10 cổ phiếu** có Sharpe Ratio cao nhất
- Huấn luyện mô hình LSTM-GRU để tìm phân bổ tài sản tối ưu
- Hiển thị kết quả danh mục đầu tư trực quan

## 📁 Cấu trúc dữ liệu

File CSV (nếu upload) cần có 3 cột sau:

- `time`: ngày giao dịch (YYYY-MM-DD)
- `ticker`: mã cổ phiếu
- `close`: giá đóng cửa

## 🧠 Mô hình học sâu sử dụng

- LSTM Layer
- GRU Layer
- Dense Softmax (để tạo phân bổ danh mục)
- Custom Loss Function tối ưu **Sharpe Ratio**

## 🛠️ Cài đặt

1. Clone repo hoặc tải mã nguồn về:
    ```bash
    git clone <link-repo>
    cd <tên-thư-mục>
    ```

2. Tạo môi trường và cài đặt thư viện:
    ```bash
    pip install -r requirements.txt
    ```

3. Chạy ứng dụng:
    ```bash
    streamlit run app.py
    ```

## 🖼️ Giao diện

- Ứng dụng có sử dụng ảnh nền và logo (các file `background.png`, `Logo_HUB.png`).
- Đảm bảo chúng được đặt đúng thư mục cùng file `.py`.

## 📚 Yêu cầu

- Python 3.8+
- Kết nối Internet (nếu dùng dữ liệu trực tiếp từ `vnstock`)

## 📌 Ghi chú

- Mã nguồn dành cho mục đích **nghiên cứu học thuật**.
- Không nên sử dụng trực tiếp để đầu tư tài chính thực tế nếu chưa được kiểm định kỹ.

---

✅ Được phát triển bởi sinh viên tại Trường Đại học Ngân hàng TP.HCM.


