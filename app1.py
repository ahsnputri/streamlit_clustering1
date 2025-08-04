import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from io import BytesIO
import base64

# Fungsi untuk login
def login_page():
    st.title("Login Aplikasi Clustering Produk")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.page = "Beranda"
            st.experimental_rerun()
        else:
            st.error("Username atau password salah")

# Konfigurasi halaman
st.set_page_config(page_title="Clustering Produk dengan CRUD", layout="wide")

# Inisialisasi session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
    st.stop()

# Load data default
@st.cache_data
def load_default_data():
    try:
        return pd.read_excel("dataset1.xlsx")
    except:
        # Fallback jika file tidak ditemukan
        return pd.DataFrame({
            'Product': ['Produk A', 'Produk B', 'Produk C'],
            'Tipe Bahan Baku': ['Tipe 1', 'Tipe 2', 'Tipe 1'],
            'Harga Rata-Rata Bahan Baku': [10000, 15000, 12000],
            'Rata-Rata Stok Bahan Baku': [50, 30, 45],
            'Rata-Rata Jumlah Penjualan Produk': [200, 150, 180]
        })

if "data" not in st.session_state:
    st.session_state.data = load_default_data()

# Style untuk tab
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Navigasi dengan tab
tab1, tab2, tab3, tab4 = st.tabs(["Beranda", "Stok", "Clustering", "Laporan"])

with tab1:  # Beranda
    st.header("Dashboard Produk")
    
    if not st.session_state.data.empty:
        df = st.session_state.data.copy()
        
        # Visualisasi clustering default (k=4)
        kolom_numerik = ['Harga Rata-Rata Bahan Baku', 'Rata-Rata Stok Bahan Baku', 'Rata-Rata Jumlah Penjualan Produk']
        
        # Preprocessing
        df_clean = df.dropna().copy()
        for kolom in kolom_numerik:
            df_clean[kolom] = df_clean[kolom].abs()
            
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_clean[kolom_numerik])
        
        # Clustering dengan k=4
        model = KMeans(n_clusters=4, random_state=42)
        cluster_labels = model.fit_predict(scaled_features)
        df_clean['Cluster'] = cluster_labels
        
        # Visualisasi
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=scaled_features[:, 0],
            y=scaled_features[:, 2],
            hue=cluster_labels,
            palette='tab10',
            ax=ax
        )
        plt.title('Visualisasi Clustering Produk (k=4)')
        plt.xlabel('Harga Rata-Rata Bahan Baku (Scaled)')
        plt.ylabel('Rata-Rata Jumlah Penjualan Produk (Scaled)')
        st.pyplot(fig)
        
        # 5 produk dengan stok paling sedikit
        st.subheader("5 Produk dengan Stok Paling Sedikit")
        stok_terendah = df.sort_values('Rata-Rata Stok Bahan Baku').head(5)
        st.dataframe(stok_terendah)

with tab2:  # Stok
    st.header("Manajemen Stok Produk")
    
    if not st.session_state.data.empty:
        # Tombol tambah data di kanan atas
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➕ Tambah Data", use_container_width=True):
                st.session_state.show_add_form = True
        
        if getattr(st.session_state, 'show_add_form', False):
            with st.form("form_tambah"):
                st.subheader("Tambah Data Baru")
                produk = st.text_input("Nama Produk")
                tipe = st.text_input("Tipe Bahan Baku")
                harga = st.number_input("Harga Rata-Rata Bahan Baku", min_value=0)
                stok = st.number_input("Rata-Rata Stok Bahan Baku", min_value=0)
                jual = st.number_input("Rata-Rata Jumlah Penjualan Produk", min_value=0)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Simpan"):
                        new_row = pd.DataFrame([[produk, tipe, harga, stok, jual]], columns=[
                            'Product', 'Tipe Bahan Baku',
                            'Harga Rata-Rata Bahan Baku',
                            'Rata-Rata Stok Bahan Baku',
                            'Rata-Rata Jumlah Penjualan Produk'
                        ])
                        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                        st.session_state.show_add_form = False
                        st.experimental_rerun()
                with col2:
                    if st.form_submit_button("Batal"):
                        st.session_state.show_add_form = False
                        st.experimental_rerun()
        
        # Tampilkan data dengan opsi edit/hapus
        st.subheader("Data Produk")
        edited_df = st.data_editor(
            st.session_state.data,
            num_rows="dynamic",
            column_config={
                "Product": st.column_config.TextColumn("Nama Produk", width="medium", required=True),
                "Tipe Bahan Baku": st.column_config.TextColumn("Tipe Bahan", width="medium"),
                "Harga Rata-Rata Bahan Baku": st.column_config.NumberColumn("Harga", format="Rp %d", width="small"),
                "Rata-Rata Stok Bahan Baku": st.column_config.NumberColumn("Stok", width="small"),
                "Rata-Rata Jumlah Penjualan Produk": st.column_config.NumberColumn("Penjualan", width="small")
            },
            key="data_editor"
        )
        
        if st.button("Simpan Perubahan"):
            st.session_state.data = edited_df
            st.success("Perubahan berhasil disimpan!")

with tab3:  # Clustering
    st.header("Clustering Data Produk")
    
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"], key="cluster_upload")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.cluster_data = df.copy()
            st.success("File berhasil dibaca dan data dimuat.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    
    if getattr(st.session_state, 'cluster_data', None) is not None:
        df = st.session_state.cluster_data.copy()
        
        kolom_numerik = ['Harga Rata-Rata Bahan Baku', 'Rata-Rata Stok Bahan Baku', 'Rata-Rata Jumlah Penjualan Produk']
        
        # Preprocessing
        df_clean = df.dropna().copy()
        for kolom in kolom_numerik:
            df_clean[kolom] = df_clean[kolom].abs()
        
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_clean[kolom_numerik])
        
        k = st.select_slider("Pilih jumlah cluster (k)", options=[3, 4, 5], value=4)
        
        if st.button("Proses Clustering"):
            model = KMeans(n_clusters=k, random_state=42)
            cluster_labels = model.fit_predict(scaled_features)
            
            df_result = df_clean.copy()
            df_result['Cluster'] = cluster_labels
            
            st.subheader("Hasil Clustering")
            st.dataframe(df_result)
            
            # Visualisasi
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x=scaled_features[:, 0],
                y=scaled_features[:, 2],
                hue=cluster_labels,
                palette='tab10',
                ax=ax
            )
            plt.title(f'Visualisasi Clustering (k={k})')
            plt.xlabel('Harga Rata-Rata Bahan Baku (Scaled)')
            plt.ylabel('Rata-Rata Jumlah Penjualan Produk (Scaled)')
            st.pyplot(fig)
            
            score = silhouette_score(scaled_features, cluster_labels)
            st.success(f"Silhouette Score untuk k={k}: {score:.4f}")

with tab4:  # Laporan
    st.header("Laporan Data Produk")
    
    if not st.session_state.data.empty:
        df = st.session_state.data.copy()
        
        # Statistik
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Produk", len(df))
        with col2:
            st.metric("Total Stok", f"{df['Rata-Rata Stok Bahan Baku'].sum():,}")
        with col3:
            st.metric("Total Penjualan", f"{df['Rata-Rata Jumlah Penjualan Produk'].sum():,}")
        
        # Tombol download di kanan atas
        st.download_button(
            label="📥 Download Laporan",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="laporan_produk.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Tabel data lengkap
        st.subheader("Data Lengkap Produk")
        st.dataframe(df)
