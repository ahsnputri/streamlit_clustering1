import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from io import BytesIO

st.set_page_config(page_title="Clustering Produk dengan CRUD", layout="wide")

# Inisialisasi session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Tambah Data", "Hapus Data", "Clustering"])
st.session_state.page = page

st.title("Aplikasi Clustering Data Produk dengan CRUD dan Multi-Halaman")

# Upload file hanya di halaman beranda
if page == "Beranda":
    st.header("Beranda - Upload File Excel")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.data = df.copy()
            st.success("File berhasil dibaca dan data dimuat.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    if not st.session_state.data.empty:
        st.subheader("Data Saat Ini")
        st.dataframe(st.session_state.data)

elif page == "Tambah Data":
    st.header("Tambah Data Baru")
    if not st.session_state.data.empty:
        with st.form("form_tambah"):
            produk = st.text_input("Nama Produk")
            tipe = st.text_input("Tipe Bahan Baku")
            harga = st.number_input("Harga Rata-Rata Bahan Baku")
            stok = st.number_input("Rata-Rata Stok Bahan Baku")
            jual = st.number_input("Rata-Rata Jumlah Penjualan Produk")
            simpan = st.form_submit_button("Simpan Data")

            if simpan:
                new_row = pd.DataFrame([[produk, tipe, harga, stok, jual]], columns=[
                    'Product', 'Tipe Bahan Baku',
                    'Harga Rata-Rata Bahan Baku',
                    'Rata-Rata Stok Bahan Baku',
                    'Rata-Rata Jumlah Penjualan Produk'
                ])
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                st.success("Data berhasil ditambahkan!")
        st.subheader("Data Saat Ini")
        st.dataframe(st.session_state.data)
    else:
        st.info("Silakan upload file terlebih dahulu di halaman Beranda.")

elif page == "Hapus Data":
    st.header("Hapus Data")
    if not st.session_state.data.empty:
        st.dataframe(st.session_state.data)
        idx_to_delete = st.number_input("Pilih index baris yang ingin dihapus", min_value=0, max_value=len(st.session_state.data)-1, step=1)
        if st.button("Hapus Baris"):
            st.session_state.data = st.session_state.data.drop(index=idx_to_delete).reset_index(drop=True)
            st.success(f"Baris ke-{idx_to_delete} berhasil dihapus.")
    else:
        st.info("Tidak ada data. Silakan upload file terlebih dahulu.")

elif page == "Clustering":
    st.header("Clustering Data Produk")
    if not st.session_state.data.empty:
        df = st.session_state.data.copy()

        kolom_numerik = ['Harga Rata-Rata Bahan Baku', 'Rata-Rata Stok Bahan Baku', 'Rata-Rata Jumlah Penjualan Produk']

        if df.isnull().values.any():
            df.fillna(df.median(numeric_only=True), inplace=True)

        if df.duplicated().any():
            df.drop_duplicates(inplace=True)

        for kolom in kolom_numerik:
            df[kolom] = df[kolom].abs()

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[kolom_numerik])
        df_scaled = pd.DataFrame(scaled_features, columns=kolom_numerik)

        k = st.slider("Jumlah Cluster (k)", min_value=2, max_value=10, value=3)

        model = KMeans(n_clusters=k, random_state=42)
        cluster_labels = model.fit_predict(scaled_features)

        df_result = df.copy()
        df_result['Cluster'] = cluster_labels

        st.subheader("Hasil Clustering")
        st.dataframe(df_result)

        plot_df = df_scaled.copy()
        plot_df['Cluster'] = cluster_labels

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            x=plot_df['Harga Rata-Rata Bahan Baku'],
            y=plot_df['Rata-Rata Jumlah Penjualan Produk'],
            hue=plot_df['Cluster'],
            palette='tab10',
            ax=ax
        )
        plt.title(f'Visualisasi Clustering (k={k})')
        plt.xlabel('Harga Rata-Rata Bahan Baku (Scaled)')
        plt.ylabel('Rata-Rata Jumlah Penjualan Produk (Scaled)')
        st.pyplot(fig)

        score = silhouette_score(scaled_features, cluster_labels)
        st.success(f"Silhouette Score untuk k={k}: {score:.4f}")

                # Unduh hasil clustering
        # buffer = BytesIO()
        # with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        #     df_result.to_excel(writer, index=False, sheet_name='Clustered Data')
        # st.download_button(
        #     label="Download Hasil Clustering",
        #     data=buffer.getvalue(),
        #     file_name="hasil_clustering.xlsx",
        #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        # )
    else:
        st.info("Tidak ada data untuk clustering. Silakan upload file terlebih dahulu.")

