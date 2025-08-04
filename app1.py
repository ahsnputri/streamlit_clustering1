import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Aplikasi Clustering dengan CRUD", layout="centered")

st.title("Aplikasi Clustering dengan CRUD")

# Simpan data di session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["fitur1", "fitur2"])

# Upload CSV (opsional)
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)

df = st.session_state.data

st.subheader("Data Saat Ini")
st.dataframe(df)

# Create (Tambah Data Baru)
st.subheader("Tambah Data Baru")
with st.form("form_tambah"):
    fitur1 = st.number_input("Nilai fitur 1", value=0.0)
    fitur2 = st.number_input("Nilai fitur 2", value=0.0)
    tambah = st.form_submit_button("Tambah")
    if tambah:
        new_row = pd.DataFrame([[fitur1, fitur2]], columns=["fitur1", "fitur2"])
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        st.success("Data berhasil ditambahkan!")

# Delete
st.subheader("Hapus Data")
hapus_index = st.number_input("Index baris yang ingin dihapus", min_value=0, max_value=len(df)-1 if len(df)>0 else 0, step=1)
if st.button("Hapus"):
    if len(df) > 0:
        st.session_state.data = st.session_state.data.drop(index=hapus_index).reset_index(drop=True)
        st.success(f"Data baris ke-{hapus_index} berhasil dihapus.")

# Clustering
if len(st.session_state.data) >= 2:
    st.subheader("Clustering")
    k = st.slider("Jumlah Cluster (k)", 2, min(10, len(df)), 2)
    
    model = KMeans(n_clusters=k)
    df['cluster'] = model.fit_predict(df[['fitur1', 'fitur2']])
    st.write("Hasil Clustering:")
    st.dataframe(df)

    # Visualisasi hasil clustering
    st.subheader("Visualisasi Clustering")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['fitur1'], df['fitur2'], c=df['cluster'], cmap='viridis')
    ax.set_xlabel("fitur1")
    ax.set_ylabel("fitur2")
    ax.set_title("Hasil Clustering")
    st.pyplot(fig)
