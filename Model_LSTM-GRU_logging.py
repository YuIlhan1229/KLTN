# ==============================
# PHẦN IMPORT & CẤU HÌNH LOG
# ==============================
import logging
import time

# Ghi log ra file app_log.txt
logging.basicConfig(
    filename='app_log.txt',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==============================
# CODE GỐC TỪ NGƯỜI DÙNG (BỔ SUNG LOG SAU)
# ==============================

# Phần mã đã rút gọn trong đoạn trước (từ phần `main()` trở xuống) sẽ được sửa ở bước sau.
# Để đảm bảo tính rõ ràng, mình sẽ gộp và dán tiếp phần hoàn chỉnh đã chèn logging ở bước sau.

def main():
    start_time = time.time()
    logging.info("🚀 Người dùng bắt đầu chạy ứng dụng.")

    st.markdown("""
    ### 🧪 Hướng dẫn sử dụng
    Ứng dụng này có hai tùy chọn dữ liệu đầu vào:
    1. **Tải lên file CSV** có chứa dữ liệu gồm các cột `'time'`, `'ticker'`, `'close'`.
    2. **Hoặc** để hệ thống **tự động tải dữ liệu** từ `vnstock` nếu bạn không upload file.
    👉 **Lưu ý:**  
    Nếu bạn chỉ muốn **trải nghiệm nhanh ứng dụng**, **KHÔNG cần tải lên gì cả**, chỉ cần **nhấn nút "Nhấn để bắt đầu tính toán"**.  
    Hệ thống sẽ sử dụng **mặc định ngành "Xây dựng"** và **khoảng thời gian từ 01/01/2018 đến 31/12/2024**.  
    """)

    # ... (các dòng giữ nguyên từ mã gốc như danh sách ngành, lựa chọn thời gian, v.v.)

    if st.button("Nhấn để bắt đầu tính toán"):
        logging.info("🟢 Người dùng nhấn nút bắt đầu tính toán.")
        st.write("**Bắt đầu lấy dữ liệu & xử lý...**")

        # Sau khi tải cổ phiếu từ vnstock
        logging.info(f"🔍 Ngành: {industry}, thời gian: {start_date_str} → {end_date_str}")
        logging.info(f"✅ Tải thành công {len(all_data)} cổ phiếu.")

        # Trước khi huấn luyện
        train_start = time.time()
        logging.info("🧠 Bắt đầu huấn luyện mô hình LSTM-GRU.")

        # Sau khi huấn luyện
        train_end = time.time()
        logging.info(f"✅ Mô hình huấn luyện hoàn tất trong {train_end - train_start:.2f} giây.")

        # Kết thúc toàn bộ xử lý
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
