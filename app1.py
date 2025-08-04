import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
    .login-box {
        max-width: 400px;
        margin: 5rem auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    
    /* Pastikan semua tombol bisa diklik */
    .stButton>button {
        position: relative;
        z-index: 1;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Style untuk header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    /* Style untuk card */
    .card {
        background-color: var(--card);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Style untuk visualisasi */
    .visualization-container {
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Style untuk tombol action */
    .action-btn {
        background: none;
        border: none;
        cursor: pointer;
        font-size: 1.1rem;
        margin: 0 3px;
        padding: 5px;
    }
    
    .action-btn:hover {
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# ========== FUNGSI LOGIN ==========
def login_page():
    st.markdown("""
    <div class="login-box">
        <h2 style="text-align: center; color: var(--primary); margin-bottom: 1.5rem;">Login Aplikasi</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Input fields sebenarnya
    username = st.text_input("Username", key="login_username", label_visibility="collapsed")
    password = st.text_input("Password", type="password", key="login_password", label_visibility="collapsed")
    
    # Tombol login
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Login", key="login_btn", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.page = "Beranda"
                st.rerun()
            else:
                st.error("Username atau password salah")

# ========== INISIALISASI SESSION STATE ==========
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Beranda"

if "data" not in st.session_state:
    try:
        st.session_state.data = pd.read_excel("dataset1.xlsx")
    except:
        st.session_state.data = pd.DataFrame({
            'Product': ['Produk A', 'Produk B', 'Produk C', 'Produk D', 'Produk E'],
            'Tipe Bahan Baku': ['Tipe 1', 'Tipe 2', 'Tipe 1', 'Tipe 3', 'Tipe 2'],
            'Harga Rata-Rata Bahan Baku': [10000, 15000, 12000, 18000, 9000],
            'Rata-Rata Stok Bahan Baku': [50, 30, 45, 20, 60],
            'Rata-Rata Jumlah Penjualan Produk': [200, 150, 180, 220, 170]
        })

if "show_form" not in st.session_state:
    st.session_state.show_form = False

if "edit_index" not in st.session_state:
    st.session_state.edit_index = None

# ========== HALAMAN LOGIN ==========
if not st.session_state.logged_in:
    login_page()
    st.stop()

# ========== KOMPONEN HEADER ==========
def show_header(title):
    col1, col2 = st.columns([4,1])
    with col1:
        st.title(title)
    with col2:
        # Dropdown profil
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 10px;">
            <div style="text-align: right;">
                <p style="margin: 0; font-weight: 500;">Admin</p>
                <p style="margin: 0; font-size: 0.8rem; color: #6c757d;">Administrator</p>
            </div>
            <div style="position: relative;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background-color: var(--accent); 
                            display: flex; align-items: center; justify-content: center; color: white; 
                            font-weight: bold; cursor: pointer;" onclick="document.getElementById('profile-dropdown').style.display='block'">
                    A
                </div>
                <div id="profile-dropdown" style="display: none; position: absolute; right: 0; background-color: white; 
                                                min-width: 120px; box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.1); 
                                                z-index: 1; border-radius: 4px; padding: 0.5rem 0;">
                    <a href="#" onclick="handleLogout()" style="color: var(--secondary); text-decoration: none; 
                                                            display: block; padding: 0.5rem 1rem;">Logout</a>
                </div>
            </div>
        </div>
        <script>
            function handleLogout() {
                window.streamlitApi.runMethod('logout', '');
            }
            
            // Tutup dropdown saat klik di luar
            window.onclick = function(event) {
                if (!event.target.matches('div[onclick*="profile-dropdown"]')) {
                    document.getElementById('profile-dropdown').style.display = 'none';
                }
            }
        </script>
        """, unsafe_allow_html=True)

# Handle logout
if st.session_state.get('logout'):
    st.session_state.logged_in = False
    st.rerun()

# ========== HALAMAN BERANDA ==========
def beranda_page():
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
        st.markdown("<div class='visualization-container'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            x=scaled_features[:, 0],
            y=scaled_features[:, 2],
            hue=cluster_labels,
            palette='tab10',
            ax=ax
        )
        plt.title('Visualisasi Clustering Produk (k=4)', pad=10)
        plt.xlabel('Harga Rata-Rata Bahan Baku (Scaled)')
        plt.ylabel('Rata-Rata Jumlah Penjualan Produk (Scaled)')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 5 produk dengan stok paling sedikit
        st.markdown("<div class='card'><h3>5 Produk dengan Stok Paling Sedikit</h3></div>", unsafe_allow_html=True)
        stok_terendah = df.sort_values('Rata-Rata Stok Bahan Baku').head(5)[['Product', 'Rata-Rata Stok Bahan Baku']]
        st.dataframe(stok_terendah, use_container_width=True)

# ========== HALAMAN STOK ==========
def stok_page():
    show_header("Manajemen Stok Produk")
    
    if not st.session_state.data.empty:
        # Tombol tambah data
        if st.button("➕ Tambah Produk", key="tambah_produk"):
            st.session_state.show_form = True
            st.session_state.edit_index = None
            st.rerun()
        
        # Form tambah/edit data
        if st.session_state.show_form:
            with st.form("produk_form"):
                st.markdown("<div class='card'><h3>Form Produk</h3></div>", unsafe_allow_html=True)
                
                # Data yang akan diedit (jika ada)
                if st.session_state.edit_index is not None:
                    produk_data = st.session_state.data.iloc[st.session_state.edit_index]
                else:
                    produk_data = None
                
                col1, col2 = st.columns(2)
                with col1:
                    produk = st.text_input("Nama Produk", 
                                         value=produk_data['Product'] if produk_data is not None else "")
                    tipe = st.text_input("Tipe Bahan Baku", 
                                       value=produk_data['Tipe Bahan Baku'] if produk_data is not None else "")
                with col2:
                    harga = st.number_input("Harga Rata-Rata Bahan Baku", 
                                         min_value=0,
                                         value=int(produk_data['Harga Rata-Rata Bahan Baku']) if produk_data is not None else 0)
                    stok = st.number_input("Rata-Rata Stok Bahan Baku", 
                                        min_value=0,
                                        value=int(produk_data['Rata-Rata Stok Bahan Baku']) if produk_data is not None else 0)
                
                jual = st.number_input("Rata-Rata Jumlah Penjualan Produk", 
                                     min_value=0,
                                     value=int(produk_data['Rata-Rata Jumlah Penjualan Produk']) if produk_data is not None else 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Simpan"):
                        new_row = {
                            'Product': produk,
                            'Tipe Bahan Baku': tipe,
                            'Harga Rata-Rata Bahan Baku': harga,
                            'Rata-Rata Stok Bahan Baku': stok,
                            'Rata-Rata Jumlah Penjualan Produk': jual
                        }
                        
                        if st.session_state.edit_index is not None:
                            # Update data yang ada
                            st.session_state.data.loc[st.session_state.edit_index] = new_row
                        else:
                            # Tambah data baru
                            st.session_state.data = st.session_state.data.append(new_row, ignore_index=True)
                        
                        st.session_state.show_form = False
                        st.session_state.edit_index = None
                        st.rerun()
                
                with col2:
                    if st.form_submit_button("Batal"):
                        st.session_state.show_form = False
                        st.session_state.edit_index = None
                        st.rerun()
        
        # Tampilkan data dengan action buttons
        st.markdown("<div class='card'><h3>Daftar Produk</h3></div>", unsafe_allow_html=True)
        
        # Buat DataFrame untuk ditampilkan
        display_df = st.session_state.data.copy()
        
        # Tambahkan kolom Action
        display_df['Action'] = ""
        
        # Tampilkan tabel
        st.dataframe(display_df, use_container_width=True)
        
        # Tambahkan tombol action untuk setiap baris
        for idx in range(len(st.session_state.data)):
            cols = st.columns([1,1,1,1,1,1,1,1,1,1])
            with cols[-2]:
                if st.button("✏️", key=f"edit_{idx}"):
                    st.session_state.show_form = True
                    st.session_state.edit_index = idx
                    st.rerun()
            with cols[-1]:
                if st.button("🗑️", key=f"delete_{idx}"):
                    st.session_state.data = st.session_state.data.drop(index=idx).reset_index(drop=True)
                    st.rerun()

# ========== HALAMAN CLUSTERING ==========
def clustering_page():
    show_header("Clustering Produk")
    
    st.markdown("<div class='card'><h3>Upload Data Produk</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed", key="cluster_upload")
    
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
        
        k = st.select_slider("Pilih jumlah cluster (k)", options=[3, 4, 5], value=4, key="cluster_k")
        
        if st.button("Proses Clustering", key="process_cluster"):
            model = KMeans(n_clusters=k, random_state=42)
            cluster_labels = model.fit_predict(scaled_features)
            
            df_result = df_clean.copy()
            df_result['Cluster'] = cluster_labels
            
            st.markdown("<div class='card'><h3>Hasil Clustering</h3></div>", unsafe_allow_html=True)
            st.dataframe(df_result, use_container_width=True)
            
            # Visualisasi
            st.markdown("<div class='card'><h3>Visualisasi Cluster</h3></div>", unsafe_allow_html=True)
            st.markdown("<div class='visualization-container'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x=scaled_features[:, 0],
                y=scaled_features[:, 2],
                hue=cluster_labels,
                palette='tab10',
                ax=ax
            )
            plt.title(f'Visualisasi Clustering (k={k})', pad=10)
            plt.xlabel('Harga Rata-Rata Bahan Baku (Scaled)')
            plt.ylabel('Rata-Rata Jumlah Penjualan Produk (Scaled)')
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
            score = silhouette_score(scaled_features, cluster_labels)
            st.success(f"Silhouette Score untuk k={k}: {score:.4f}")

# ========== HALAMAN LAPORAN ==========
def laporan_page():
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
            use_container_width=True,
            key="download_report"
        )
        
        # Tabel data lengkap
        st.markdown("<div class='card'><h3>Data Lengkap Produk</h3></div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

# ========== NAVIGASI HALAMAN ==========
pages = {
    "Beranda": beranda_page,
    "Stok": stok_page,
    "Clustering": clustering_page,
    "Laporan": laporan_page
}

# Sidebar navigation
st.sidebar.title("Navigasi")
selected_page = st.sidebar.radio("Pilih Halaman", list(pages.keys()), index=list(pages.keys()).index(st.session_state.page))

# Update session state
st.session_state.page = selected_page

# Tampilkan halaman yang dipilih
pages[selected_page]()
