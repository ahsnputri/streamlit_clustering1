import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import base64
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(
    page_title="Clustering Produk",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded"
)

# CSS Kustom
st.markdown("""
<style>
    /* Warna utama */
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
        --accent: #4fc3f7;
        --background: #f8f9fa;
        --card: #ffffff;
    }
    
    /* Style untuk login */
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: var(--card);
    }
    
    .login-input {
        width: 100%;
        margin-bottom: 1rem;
    }
    
    /* Style untuk header */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    /* Style untuk card */
    .card {
        background-color: var(--card);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Style untuk tombol */
    .stButton>button {
        border-radius: 6px;
        border: 1px solid var(--primary);
        background-color: var(--primary);
        color: white;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
        color: white;
    }
    
    /* Style untuk tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e9ecef;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        color: #495057;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white !important;
    }
    
    /* Style untuk tabel */
    .dataframe {
        width: 100%;
    }
    
    /* Style untuk visualisasi */
    .stPlotContainer {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk login
def login_page():
    st.markdown(
        """
        <div style='display: flex; justify-content: center; align-items: center; height: 80vh;'>
            <div class='login-container'>
                <h2 style='text-align: center; color: var(--primary); margin-bottom: 2rem;'>Login Aplikasi</h2>
                <form>
                    <div class='login-input'>
                        <label style='display: block; margin-bottom: 0.5rem; color: var(--secondary);'>Username</label>
                        <input type='text' style='width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #ced4da;' />
                    </div>
                    <div class='login-input'>
                        <label style='display: block; margin-bottom: 0.5rem; color: var(--secondary);'>Password</label>
                        <input type='password' style='width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #ced4da;' />
                    </div>
                    <div style='text-align: right; margin-bottom: 1.5rem;'>
                        <a href='#' style='color: var(--secondary); text-decoration: none; font-size: 0.8rem;'>Lupa Password?</a>
                    </div>
                    <button type='submit' style='width: 100%; padding: 0.5rem; background-color: var(--primary); color: white; border: none; border-radius: 4px; cursor: pointer;'>Login</button>
                </form>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Logika login sederhana
    if st.button("Login (Demo)", key="demo_login"):
        st.session_state.logged_in = True
        st.session_state.page = "Beranda"
        st.rerun()

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
            'Product': ['Produk A', 'Produk B', 'Produk C', 'Produk D', 'Produk E'],
            'Tipe Bahan Baku': ['Tipe 1', 'Tipe 2', 'Tipe 1', 'Tipe 3', 'Tipe 2'],
            'Harga Rata-Rata Bahan Baku': [10000, 15000, 12000, 18000, 9000],
            'Rata-Rata Stok Bahan Baku': [50, 30, 45, 20, 60],
            'Rata-Rata Jumlah Penjualan Produk': [200, 150, 180, 220, 170]
        })

if "data" not in st.session_state:
    st.session_state.data = load_default_data()

# Header dengan profil dan logout
def show_header(current_page):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"<h1 style='color: var(--primary);'>{current_page}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
            <div style='display: flex; align-items: center; justify-content: flex-end; gap: 1rem;'>
                <div style='text-align: right;'>
                    <p style='margin: 0; font-weight: 500;'>Admin</p>
                    <p style='margin: 0; font-size: 0.8rem; color: #6c757d;'>Administrator</p>
                </div>
                <div style='width: 40px; height: 40px; border-radius: 50%; background-color: var(--accent); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>A</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.rerun()

# Navigasi dengan tab
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Beranda", "📦 Stok", "📊 Clustering", "📑 Laporan"])

with tab1:  # Beranda
    show_header("Dashboard Produk")
    
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
        st.markdown("<div class='card'><h3>Visualisasi Clustering Produk</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            x=scaled_features[:, 0],
            y=scaled_features[:, 2],
            hue=cluster_labels,
            palette='tab10',
            ax=ax
        )
        plt.title('Visualisasi Clustering Produk (k=4)', pad=20)
        plt.xlabel('Harga Rata-Rata Bahan Baku (Scaled)')
        plt.ylabel('Rata-Rata Jumlah Penjualan Produk (Scaled)')
        st.pyplot(fig)
        
        # 5 produk dengan stok paling sedikit
        st.markdown("<div class='card'><h3>5 Produk dengan Stok Paling Sedikit</h3></div>", unsafe_allow_html=True)
        stok_terendah = df.sort_values('Rata-Rata Stok Bahan Baku').head(5)[['Product', 'Rata-Rata Stok Bahan Baku']]
        st.dataframe(stok_terendah, use_container_width=True)

with tab2:  # Stok
    show_header("Manajemen Stok Produk")
    
    if not st.session_state.data.empty:
        # Tombol tambah data
        if st.button("➕ Tambah Produk", key="add_product"):
            st.session_state.show_add_form = True
        
        if getattr(st.session_state, 'show_add_form', False):
            with st.form("form_tambah"):
                st.markdown("<div class='card'><h3>Tambah Produk Baru</h3></div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    produk = st.text_input("Nama Produk")
                    tipe = st.text_input("Tipe Bahan Baku")
                    harga = st.number_input("Harga Rata-Rata Bahan Baku", min_value=0)
                with col2:
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
                        st.success("Produk berhasil ditambahkan!")
                        st.rerun()
                with col2:
                    if st.form_submit_button("Batal"):
                        st.session_state.show_add_form = False
                        st.rerun()
        
        # Tampilkan data dengan opsi edit/hapus
        st.markdown("<div class='card'><h3>Daftar Produk</h3></div>", unsafe_allow_html=True)
        
        # Buat salinan data untuk editing
        edited_data = st.session_state.data.copy()
        
        # Tampilkan tabel dengan kolom aksi
        edited_data['Aksi'] = "✏️ | 🗑️"
        st.dataframe(edited_data, use_container_width=True)
        
        # Form edit (akan muncul ketika tombol edit diklik)
        if st.button("Edit Data", key="edit_btn"):
            st.session_state.show_edit_form = True
        
        if getattr(st.session_state, 'show_edit_form', False):
            with st.form("form_edit"):
                st.markdown("<div class='card'><h3>Edit Produk</h3></div>", unsafe_allow_html=True)
                selected_index = st.number_input("Pilih nomor baris yang akan diedit", 
                                              min_value=0, 
                                              max_value=len(st.session_state.data)-1,
                                              step=1)
                
                if selected_index >= 0 and selected_index < len(st.session_state.data):
                    selected_row = st.session_state.data.iloc[selected_index]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        produk = st.text_input("Nama Produk", value=selected_row['Product'])
                        tipe = st.text_input("Tipe Bahan Baku", value=selected_row['Tipe Bahan Baku'])
                        harga = st.number_input("Harga Rata-Rata Bahan Baku", 
                                             min_value=0,
                                             value=int(selected_row['Harga Rata-Rata Bahan Baku']))
                    with col2:
                        stok = st.number_input("Rata-Rata Stok Bahan Baku", 
                                             min_value=0,
                                             value=int(selected_row['Rata-Rata Stok Bahan Baku']))
                        jual = st.number_input("Rata-Rata Jumlah Penjualan Produk", 
                                             min_value=0,
                                             value=int(selected_row['Rata-Rata Jumlah Penjualan Produk']))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Simpan Perubahan"):
                            st.session_state.data.at[selected_index, 'Product'] = produk
                            st.session_state.data.at[selected_index, 'Tipe Bahan Baku'] = tipe
                            st.session_state.data.at[selected_index, 'Harga Rata-Rata Bahan Baku'] = harga
                            st.session_state.data.at[selected_index, 'Rata-Rata Stok Bahan Baku'] = stok
                            st.session_state.data.at[selected_index, 'Rata-Rata Jumlah Penjualan Produk'] = jual
                            st.session_state.show_edit_form = False
                            st.success("Perubahan berhasil disimpan!")
                            st.rerun()
                    with col2:
                        if st.form_submit_button("Batal"):
                            st.session_state.show_edit_form = False
                            st.rerun()

with tab3:  # Clustering
    show_header("Clustering Produk")
    
    st.markdown("<div class='card'><h3>Upload Data Produk</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.cluster_data = df.copy()
            st.success("File berhasil diupload dan data dimuat.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    
    if getattr(st.session_state, 'cluster_data', None) is not None:
        df = st.session_state.cluster_data.copy()
        
        st.markdown("<div class='card'><h3>Pengaturan Clustering</h3></div>", unsafe_allow_html=True)
        kolom_numerik = ['Harga Rata-Rata Bahan Baku', 'Rata-Rata Stok Bahan Baku', 'Rata-Rata Jumlah Penjualan Produk']
        
        # Preprocessing
        df_clean = df.dropna().copy()
        for kolom in kolom_numerik:
            df_clean[kolom] = df_clean[kolom].abs()
        
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_clean[kolom_numerik])
        
        k = st.select_slider("Pilih jumlah cluster (k)", options=[3, 4, 5], value=4)
        
        if st.button("Proses Clustering", key="process_cluster"):
            model = KMeans(n_clusters=k, random_state=42)
            cluster_labels = model.fit_predict(scaled_features)
            
            df_result = df_clean.copy()
            df_result['Cluster'] = cluster_labels
            
            st.markdown("<div class='card'><h3>Hasil Clustering</h3></div>", unsafe_allow_html=True)
            st.dataframe(df_result, use_container_width=True)
            
            # Visualisasi
            st.markdown("<div class='card'><h3>Visualisasi Cluster</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                x=scaled_features[:, 0],
                y=scaled_features[:, 2],
                hue=cluster_labels,
                palette='tab10',
                ax=ax
            )
            plt.title(f'Visualisasi Clustering (k={k})', pad=20)
            plt.xlabel('Harga Rata-Rata Bahan Baku (Scaled)')
            plt.ylabel('Rata-Rata Jumlah Penjualan Produk (Scaled)')
            st.pyplot(fig)
            
            score = silhouette_score(scaled_features, cluster_labels)
            st.success(f"Silhouette Score untuk k={k}: {score:.4f}")

with tab4:  # Laporan
    show_header("Laporan Produk")
    
    if not st.session_state.data.empty:
        df = st.session_state.data.copy()
        
        # Statistik
        st.markdown("<div class='card'><h3>Statistik Produk</h3></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Produk", len(df), help="Jumlah total produk yang terdaftar")
        with col2:
            st.metric("Total Stok", f"{df['Rata-Rata Stok Bahan Baku'].sum():,}", help="Total stok semua produk")
        with col3:
            st.metric("Total Penjualan", f"{df['Rata-Rata Jumlah Penjualan Produk'].sum():,}", help="Total penjualan semua produk")
        
        # Tombol download
        st.markdown("<div class='card'><h3>Download Laporan</h3></div>", unsafe_allow_html=True)
        
        # Fungsi untuk download Excel
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Laporan')
            processed_data = output.getvalue()
            return processed_data
        
        excel_data = to_excel(df)
        st.download_button(
            label="📥 Download Laporan (Excel)",
            data=excel_data,
            file_name="laporan_produk.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # Tabel data lengkap
        st.markdown("<div class='card'><h3>Data Lengkap Produk</h3></div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
