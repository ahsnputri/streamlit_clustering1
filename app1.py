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
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .tabs-container {
        flex-grow: 1;
    }
    
    .profile-container {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* Style untuk card */
    .card {
        background-color: var(--card);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Style untuk modal */
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }
    
    .modal-content {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        width: 50%;
        max-width: 600px;
    }
    
    /* Style untuk tabel action */
    .action-btn {
        background: none;
        border: none;
        cursor: pointer;
        font-size: 1.2rem;
        margin: 0 5px;
    }
    
    /* Style untuk visualisasi */
    .visualization-container {
        max-width: 700px;
        margin: 0 auto;
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
                        <input type='text' id='username' style='width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #ced4da;' />
                    </div>
                    <div class='login-input'>
                        <label style='display: block; margin-bottom: 0.5rem; color: var(--secondary);'>Password</label>
                        <input type='password' id='password' style='width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #ced4da;' />
                    </div>
                    <div style='text-align: right; margin-bottom: 1.5rem;'>
                        <a href='#' style='color: var(--secondary); text-decoration: none; font-size: 0.8rem;'>Lupa Password?</a>
                    </div>
                    <button type='button' onclick='handleLogin()' style='width: 100%; padding: 0.5rem; background-color: var(--primary); color: white; border: none; border-radius: 4px; cursor: pointer;'>Login</button>
                </form>
            </div>
        </div>
        <script>
            function handleLogin() {
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                window.streamlitApi.runMethod('login', {username, password});
            }
        </script>
        """,
        unsafe_allow_html=True
    )

# Inisialisasi session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False
    st.session_state.modal_type = ""
    st.session_state.edit_index = None

if "data" not in st.session_state:
    # Load data default
    try:
        st.session_state.data = pd.read_excel("dataset1.xlsx")
    except:
        # Fallback jika file tidak ditemukan
        st.session_state.data = pd.DataFrame({
            'Product': ['Produk A', 'Produk B', 'Produk C', 'Produk D', 'Produk E'],
            'Tipe Bahan Baku': ['Tipe 1', 'Tipe 2', 'Tipe 1', 'Tipe 3', 'Tipe 2'],
            'Harga Rata-Rata Bahan Baku': [10000, 15000, 12000, 18000, 9000],
            'Rata-Rata Stok Bahan Baku': [50, 30, 45, 20, 60],
            'Rata-Rata Jumlah Penjualan Produk': [200, 150, 180, 220, 170]
        })

# Handle login dari JavaScript
if st.session_state.get('login'):
    username = st.session_state.login['username']
    password = st.session_state.login['password']
    
    if username == "admin" and password == "admin123":
        st.session_state.logged_in = True
        st.session_state.page = "Beranda"
        st.rerun()
    else:
        st.error("Username atau password salah")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# Header dengan profil dan tab
def show_header(current_page):
    st.markdown("""
    <div class='header-container'>
        <div class='tabs-container'>
            <div data-testid='stTabs' style='width: 100%;'>
                <div role='tablist'>
                    <button role='tab' aria-selected='true'>🏠 Beranda</button>
                    <button role='tab'>📦 Stok</button>
                    <button role='tab'>📊 Clustering</button>
                    <button role='tab'>📑 Laporan</button>
                </div>
            </div>
        </div>
        <div class='profile-container'>
            <div style='text-align: right;'>
                <p style='margin: 0; font-weight: 500;'>Admin</p>
                <p style='margin: 0; font-size: 0.8rem; color: #6c757d;'>Administrator</p>
            </div>
            <div style='width: 40px; height: 40px; border-radius: 50%; background-color: var(--accent); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; cursor: pointer;' onclick='handleLogout()'>A</div>
        </div>
    </div>
    <script>
        function handleLogout() {
            if (confirm('Apakah Anda yakin ingin logout?')) {
                window.streamlitApi.runMethod('logout', '');
            }
        }
    </script>
    """, unsafe_allow_html=True)

# Callback untuk logout
if st.session_state.get('logout'):
    st.session_state.logged_in = False
    st.rerun()

# Modal untuk form
def show_modal():
    if st.session_state.show_modal:
        st.markdown("""
        <div class='modal'>
            <div class='modal-content'>
        """, unsafe_allow_html=True)
        
        if st.session_state.modal_type == "add":
            st.markdown("<h3>Tambah Produk Baru</h3>", unsafe_allow_html=True)
            
            with st.form("form_tambah"):
                col1, col2 = st.columns(2)
                with col1:
                    produk = st.text_input("Nama Produk", key="modal_produk")
                    tipe = st.text_input("Tipe Bahan Baku", key="modal_tipe")
                with col2:
                    harga = st.number_input("Harga Rata-Rata Bahan Baku", min_value=0, key="modal_harga")
                    stok = st.number_input("Rata-Rata Stok Bahan Baku", min_value=0, key="modal_stok")
                
                jual = st.number_input("Rata-Rata Jumlah Penjualan Produk", min_value=0, key="modal_jual")
                
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
                        st.session_state.show_modal = False
                        st.rerun()
                with col2:
                    if st.form_submit_button("Batal"):
                        st.session_state.show_modal = False
                        st.rerun()
        
        elif st.session_state.modal_type == "edit":
            st.markdown("<h3>Edit Produk</h3>", unsafe_allow_html=True)
            
            selected_row = st.session_state.data.iloc[st.session_state.edit_index]
            
            with st.form("form_edit"):
                col1, col2 = st.columns(2)
                with col1:
                    produk = st.text_input("Nama Produk", value=selected_row['Product'], key="modal_edit_produk")
                    tipe = st.text_input("Tipe Bahan Baku", value=selected_row['Tipe Bahan Baku'], key="modal_edit_tipe")
                with col2:
                    harga = st.number_input("Harga Rata-Rata Bahan Baku", 
                                         min_value=0,
                                         value=int(selected_row['Harga Rata-Rata Bahan Baku']),
                                         key="modal_edit_harga")
                    stok = st.number_input("Rata-Rata Stok Bahan Baku", 
                                         min_value=0,
                                         value=int(selected_row['Rata-Rata Stok Bahan Baku']),
                                         key="modal_edit_stok")
                
                jual = st.number_input("Rata-Rata Jumlah Penjualan Produk", 
                                     min_value=0,
                                     value=int(selected_row['Rata-Rata Jumlah Penjualan Produk']),
                                     key="modal_edit_jual")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Simpan Perubahan"):
                        st.session_state.data.at[st.session_state.edit_index, 'Product'] = produk
                        st.session_state.data.at[st.session_state.edit_index, 'Tipe Bahan Baku'] = tipe
                        st.session_state.data.at[st.session_state.edit_index, 'Harga Rata-Rata Bahan Baku'] = harga
                        st.session_state.data.at[st.session_state.edit_index, 'Rata-Rata Stok Bahan Baku'] = stok
                        st.session_state.data.at[st.session_state.edit_index, 'Rata-Rata Jumlah Penjualan Produk'] = jual
                        st.session_state.show_modal = False
                        st.rerun()
                with col2:
                    if st.form_submit_button("Batal"):
                        st.session_state.show_modal = False
                        st.rerun()
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

# Halaman Beranda
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
        fig, ax = plt.subplots(figsize=(5, 3.5))
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

# Halaman Stok
def stok_page():
    show_header("Manajemen Stok Produk")
    
    if not st.session_state.data.empty:
        # Tombol tambah data
        if st.button("➕ Tambah Produk", key="add_product"):
            st.session_state.show_modal = True
            st.session_state.modal_type = "add"
            st.rerun()
        
        # Tampilkan data dengan action buttons
        st.markdown("<div class='card'><h3>Daftar Produk</h3></div>", unsafe_allow_html=True)
        
        # Buat salinan data untuk ditampilkan
        display_df = st.session_state.data.copy()
        
        # Tambahkan kolom action
        display_df['Action'] = ["✏️ 🗑️"] * len(display_df)
        
        # Tampilkan tabel
        st.dataframe(display_df, use_container_width=True)
        
        # Handle action buttons
        cols = st.columns(5)
        for i in range(len(st.session_state.data)):
            if cols[i % 5].button(f"Edit {i}", key=f"edit_{i}", label_visibility="hidden"):
                st.session_state.show_modal = True
                st.session_state.modal_type = "edit"
                st.session_state.edit_index = i
                st.rerun()
            
            if cols[i % 5].button(f"Delete {i}", key=f"delete_{i}", label_visibility="hidden"):
                st.session_state.data = st.session_state.data.drop(index=i).reset_index(drop=True)
                st.rerun()

# Halaman Clustering
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

# Halaman Laporan
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

# Routing halaman
if st.session_state.get('page') == "Beranda":
    beranda_page()
elif st.session_state.get('page') == "Stok":
    stok_page()
elif st.session_state.get('page') == "Clustering":
    clustering_page()
elif st.session_state.get('page') == "Laporan":
    laporan_page()
else:
    st.session_state.page = "Beranda"
    beranda_page()

# Tampilkan modal jika diperlukan
show_modal()
