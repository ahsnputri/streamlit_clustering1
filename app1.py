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
    initial_sidebar_state="collapsed"
)

# CSS Kustom
st.markdown("""
<style>
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
        --accent: #4fc3f7;
    }
    
    /* Login Container */
    .login-box {
        max-width: 320px;
        margin: 100px auto;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    
    /* Header Styles */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .tabs-container {
        display: flex;
        gap: 1rem;
    }
    
    .tab-btn {
        padding: 0.5rem 1rem;
        background: none;
        border: none;
        cursor: pointer;
        font-size: 1rem;
        color: #555;
    }
    
    .tab-btn:hover {
        color: var(--primary);
    }
    
    .tab-btn.active {
        color: var(--primary);
        font-weight: 600;
        border-bottom: 2px solid var(--primary);
    }
    
    /* Profile Dropdown */
    .profile-wrapper {
        position: relative;
        display: inline-block;
    }
    
    .profile-content {
        display: none;
        position: absolute;
        right: 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 4px;
        min-width: 120px;
        z-index: 1;
    }
    
    .profile-wrapper:hover .profile-content {
        display: block;
    }
    
    .dropdown-item {
        padding: 0.5rem 1rem;
        display: block;
        color: #333;
        text-decoration: none;
    }
    
    .dropdown-item:hover {
        background-color: #f5f5f5;
        color: var(--primary);
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Table Actions */
    .action-buttons {
        display: flex;
        gap: 0.5rem;
    }
    
    /* Utility Classes */
    .hidden {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ========== FUNGSI UTAMA ==========

def main():
    # Inisialisasi session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = "Beranda"
    if 'data' not in st.session_state:
        try:
            st.session_state.data = pd.read_excel("dataset1.xlsx")
        except:
            st.session_state.data = pd.DataFrame({
                'Product': ['Produk A', 'Produk B', 'Produk C'],
                'Tipe Bahan Baku': ['Tipe 1', 'Tipe 2', 'Tipe 1'],
                'Harga Rata-Rata Bahan Baku': [10000, 15000, 12000],
                'Rata-Rata Stok Bahan Baku': [50, 30, 45],
                'Rata-Rata Jumlah Penjualan Produk': [200, 150, 180]
            })
    if 'show_form' not in st.session_state:
        st.session_state.show_form = False
    if 'edit_index' not in st.session_state:
        st.session_state.edit_index = None

    # Halaman Login
    if not st.session_state.logged_in:
        show_login()
        return

    # Komponen Header
    show_header()

    # Routing Halaman
    if st.session_state.page == "Beranda":
        show_beranda()
    elif st.session_state.page == "Stok":
        show_stok()
    elif st.session_state.page == "Clustering":
        show_clustering()
    elif st.session_state.page == "Laporan":
        show_laporan()

# ========== KOMPONEN TAMPILAN ==========

def show_login():
    """Menampilkan halaman login"""
    st.markdown("""
    <div class="login-box">
        <h2 style="text-align: center; color: var(--primary); margin-bottom: 1.5rem;">Login</h2>
        <form onsubmit="return false;">
            <div style="margin-bottom: 1rem;">
                <label style="display: block; margin-bottom: 0.5rem; color: #555;">Username</label>
                <input type="text" id="username" style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;">
            </div>
            <div style="margin-bottom: 1.5rem;">
                <label style="display: block; margin-bottom: 0.5rem; color: #555;">Password</label>
                <input type="password" id="password" style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;">
            </div>
            <button onclick="handleLogin()" style="width: 100%; padding: 0.5rem; background-color: var(--primary); color: white; border: none; border-radius: 4px; cursor: pointer;">Login</button>
        </form>
    </div>
    <script>
        function handleLogin() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            window.streamlitApi.runMethod('handle_login', {username, password});
        }
    </script>
    """, unsafe_allow_html=True)

    # Handle login
    if st.session_state.get('handle_login'):
        username = st.session_state.handle_login['username']
        password = st.session_state.handle_login['password']
        
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.page = "Beranda"
            st.rerun()
        else:
            st.error("Username atau password salah")

def show_header():
    """Menampilkan header dengan navigasi dan profil"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Tab Navigasi
        tabs = st.container()
        with tabs:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("🏠 Beranda", key="tab_beranda"):
                    st.session_state.page = "Beranda"
                    st.rerun()
            with col2:
                if st.button("📦 Stok", key="tab_stok"):
                    st.session_state.page = "Stok"
                    st.rerun()
            with col3:
                if st.button("📊 Clustering", key="tab_clustering"):
                    st.session_state.page = "Clustering"
                    st.rerun()
            with col4:
                if st.button("📑 Laporan", key="tab_laporan"):
                    st.session_state.page = "Laporan"
                    st.rerun()
    
    with col2:
        # Profile Dropdown
        st.markdown("""
        <div class="profile-wrapper">
            <div style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                <div style="text-align: right;">
                    <p style="margin: 0; font-weight: 500;">Admin</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #777;">Administrator</p>
                </div>
                <div style="width: 36px; height: 36px; border-radius: 50%; background-color: var(--accent); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">A</div>
            </div>
            <div class="profile-content">
                <a href="#" class="dropdown-item" onclick="window.streamlitApi.runMethod('logout', '');">Logout</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Handle logout
    if st.session_state.get('logout'):
        st.session_state.logged_in = False
        st.rerun()

def show_beranda():
    """Menampilkan halaman beranda"""
    st.markdown("<div class='card'><h2>Dashboard Produk</h2></div>", unsafe_allow_html=True)
    
    if not st.session_state.data.empty:
        df = st.session_state.data.copy()
        
        # Visualisasi clustering
        kolom_numerik = ['Harga Rata-Rata Bahan Baku', 'Rata-Rata Stok Bahan Baku', 'Rata-Rata Jumlah Penjualan Produk']
        df_clean = df.dropna().copy()
        
        for kolom in kolom_numerik:
            df_clean[kolom] = df_clean[kolom].abs()
            
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_clean[kolom_numerik])
        
        model = KMeans(n_clusters=4, random_state=42)
        cluster_labels = model.fit_predict(scaled_features)
        
        # Visualisasi
        st.markdown("<div class='card'><h3>Visualisasi Clustering</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            x=scaled_features[:, 0],
            y=scaled_features[:, 2],
            hue=cluster_labels,
            palette='tab10',
            ax=ax
        )
        plt.title('Visualisasi Clustering (k=4)')
        plt.xlabel('Harga Rata-Rata Bahan Baku')
        plt.ylabel('Rata-Rata Penjualan Produk')
        st.pyplot(fig)
        
        # 5 produk stok terendah
        st.markdown("<div class='card'><h3>5 Produk Stok Terendah</h3></div>", unsafe_allow_html=True)
        stok_terendah = df.sort_values('Rata-Rata Stok Bahan Baku').head(5)[['Product', 'Rata-Rata Stok Bahan Baku']]
        st.dataframe(stok_terendah, use_container_width=True)

def show_stok():
    """Menampilkan halaman manajemen stok"""
    st.markdown("<div class='card'><h2>Manajemen Stok Produk</h2></div>", unsafe_allow_html=True)
    
    if not st.session_state.data.empty:
        # Tombol Tambah Data
        if st.button("➕ Tambah Produk", key="btn_tambah"):
            st.session_state.show_form = True
            st.session_state.edit_index = None
            st.rerun()
        
        # Form Tambah/Edit Data
        if st.session_state.show_form:
            with st.form("form_produk"):
                st.markdown("<div class='card'><h3>Form Produk</h3></div>", unsafe_allow_html=True)
                
                # Data yang akan diedit (jika ada)
                if st.session_state.edit_index is not None:
                    produk_data = st.session_state.data.iloc[st.session_state.edit_index]
                else:
                    produk_data = None
                
                col1, col2 = st.columns(2)
                with col1:
                    produk = st.text_input("Nama Produk", value=produk_data['Product'] if produk_data is not None else "")
                    tipe = st.text_input("Tipe Bahan Baku", value=produk_data['Tipe Bahan Baku'] if produk_data is not None else "")
                with col2:
                    harga = st.number_input("Harga Rata-Rata", min_value=0, value=int(produk_data['Harga Rata-Rata Bahan Baku']) if produk_data is not None else 0)
                    stok = st.number_input("Rata-Rata Stok", min_value=0, value=int(produk_data['Rata-Rata Stok Bahan Baku']) if produk_data is not None else 0)
                
                jual = st.number_input("Rata-Rata Penjualan", min_value=0, value=int(produk_data['Rata-Rata Jumlah Penjualan Produk']) if produk_data is not None else 0)
                
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
                            st.session_state.data.loc[st.session_state.edit_index] = new_row
                        else:
                            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
                        
                        st.session_state.show_form = False
                        st.rerun()
                with col2:
                    if st.form_submit_button("Batal"):
                        st.session_state.show_form = False
                        st.rerun()
        
        # Tabel Data
        st.markdown("<div class='card'><h3>Daftar Produk</h3></div>", unsafe_allow_html=True)
        
        # Buat salinan data untuk ditampilkan
        display_df = st.session_state.data.copy()
        
        # Tampilkan tabel dengan kolom action
        st.dataframe(
            display_df,
            column_config={
                "Action": st.column_config.Column(
                    "Aksi",
                    help="Edit atau hapus produk"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Tambahkan tombol aksi untuk setiap baris
        for idx in range(len(st.session_state.data)):
            cols = st.columns([1,1,1,1,1,1,1,1,1,1])  # 10 kolom
            
            # Tombol Edit
            with cols[-2]:
                if st.button("✏️", key=f"edit_{idx}"):
                    st.session_state.show_form = True
                    st.session_state.edit_index = idx
                    st.rerun()
            
            # Tombol Hapus
            with cols[-1]:
                if st.button("🗑️", key=f"delete_{idx}"):
                    st.session_state.data = st.session_state.data.drop(index=idx).reset_index(drop=True)
                    st.rerun()

def show_clustering():
    """Menampilkan halaman clustering"""
    st.markdown("<div class='card'><h2>Clustering Produk</h2></div>", unsafe_allow_html=True)
    
    # Upload file
    uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"], key="file_uploader")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.cluster_data = df
            st.success("File berhasil diupload!")
        except Exception as e:
            st.error(f"Error: {e}")
    
    if hasattr(st.session_state, 'cluster_data') and st.session_state.cluster_data is not None:
        df = st.session_state.cluster_data.copy()
        
        # Preprocessing
        kolom_numerik = ['Harga Rata-Rata Bahan Baku', 'Rata-Rata Stok Bahan Baku', 'Rata-Rata Jumlah Penjualan Produk']
        df_clean = df.dropna().copy()
        
        for kolom in kolom_numerik:
            df_clean[kolom] = df_clean[kolom].abs()
        
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_clean[kolom_numerik])
        
        # Pilih jumlah cluster
        k = st.slider("Jumlah Cluster (k)", 2, 5, 3, key="cluster_k")
        
        if st.button("Proses Clustering", key="btn_cluster"):
            model = KMeans(n_clusters=k, random_state=42)
            cluster_labels = model.fit_predict(scaled_features)
            
            df_result = df_clean.copy()
            df_result['Cluster'] = cluster_labels
            
            # Tampilkan hasil
            st.markdown("<div class='card'><h3>Hasil Clustering</h3></div>", unsafe_allow_html=True)
            st.dataframe(df_result, use_container_width=True)
            
            # Visualisasi
            st.markdown("<div class='card'><h3>Visualisasi Cluster</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x=scaled_features[:, 0],
                y=scaled_features[:, 2],
                hue=cluster_labels,
                palette='tab10',
                ax=ax
            )
            plt.title(f'Visualisasi Clustering (k={k})')
            plt.xlabel('Harga Rata-Rata Bahan Baku')
            plt.ylabel('Rata-Rata Penjualan Produk')
            st.pyplot(fig)
            
            # Hitung silhouette score
            score = silhouette_score(scaled_features, cluster_labels)
            st.success(f"Silhouette Score: {score:.4f}")

def show_laporan():
    """Menampilkan halaman laporan"""
    st.markdown("<div class='card'><h2>Laporan Produk</h2></div>", unsafe_allow_html=True)
    
    if not st.session_state.data.empty:
        df = st.session_state.data.copy()
        
        # Statistik
        st.markdown("<div class='card'><h3>Statistik</h3></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Produk", len(df))
        with col2:
            st.metric("Total Stok", f"{df['Rata-Rata Stok Bahan Baku'].sum():,}")
        with col3:
            st.metric("Total Penjualan", f"{df['Rata-Rata Jumlah Penjualan Produk'].sum():,}")
        
        # Tombol Download
        st.markdown("<div class='card'><h3>Download Laporan</h3></div>", unsafe_allow_html=True)
        
        # Fungsi untuk konversi ke Excel
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Laporan')
            processed_data = output.getvalue()
            return processed_data
        
        excel_data = to_excel(df)
        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name="laporan_produk.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # Tabel Data Lengkap
        st.markdown("<div class='card'><h3>Data Lengkap</h3></div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
