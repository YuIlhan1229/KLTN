import streamlit as st
import pandas as pd

def main():
    st.title("My Streamlit App")
    
    st.write("Đây là ứng dụng demo đầu tiên!")

    # Ví dụ đọc một DataFrame (nếu bạn có file .csv)
    # data = pd.read_csv('sample_data.csv')
    # st.dataframe(data)

    # Hoặc làm một ví dụ nho nhỏ
    name = st.text_input("Nhập tên của bạn:")
    if st.button("Xác nhận"):
        st.write(f"Xin chào, {name}!")

if __name__ == "__main__":
    main()
