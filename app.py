import streamlit as st
import simpy
import random
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ============================
# KONFIGURASI SIMULASI
# ============================
@dataclass
class Config:
    """Konfigurasi parameter simulasi piket kantin"""
    # Parameter dasar
    NUM_PETUGAS: int = 7
    NUM_MEJA: int = 60
    MAHASISWA_PER_MEJA: int = 3
    TOTAL_OMPRENG: int = 180
    
    # Alokasi petugas per stage
    NUM_PETUGAS_LAUK: int = 2
    NUM_PETUGAS_ANGKUT: int = 2
    NUM_PETUGAS_NASI: int = 3
    
    # Distribusi waktu (dalam menit)
    MIN_LAUK_TIME: float = 0.5
    MAX_LAUK_TIME: float = 1.0
    
    MIN_ANGKUT_TIME: float = 0.33
    MAX_ANGKUT_TIME: float = 1.0
    
    MIN_NASI_TIME: float = 0.5
    MAX_NASI_TIME: float = 1.0
    
    # Kapasitas angkut per trip
    MIN_OMPRENG_PER_TRIP: int = 4
    MAX_OMPRENG_PER_TRIP: int = 7
    
    # Waktu mulai
    START_HOUR: int = 7
    START_MINUTE: int = 0
    
    # Seed untuk reproduktibilitas
    RANDOM_SEED: int = 42

# ============================
# MODEL SIMULASI
# ============================
class PiketKantinDES:
    """Model Discrete Event Simulation untuk piket kantin IT Del"""
    
    def __init__(self, config: Config):
        self.config = config
        self.env = simpy.Environment()
        
        # Resources: Petugas untuk 3 stage
        self.petugas_lauk = simpy.Resource(self.env, capacity=config.NUM_PETUGAS_LAUK)
        self.petugas_angkut = simpy.Resource(self.env, capacity=config.NUM_PETUGAS_ANGKUT)
        self.petugas_nasi = simpy.Resource(self.env, capacity=config.NUM_PETUGAS_NASI)
        
        # Buffer/Antrian antar stage
        self.buffer_lauk_selesai = simpy.Store(self.env)
        self.buffer_angkut_selesai = simpy.Store(self.env)
        
        # Statistik
        self.statistics = {
            'ompreng_data': [],
            'stage_times': {'lauk': [], 'angkut': [], 'nasi': []},
            'utilisasi_petugas': {'lauk': [], 'angkut': [], 'nasi': []}
        }
        
        # Waktu mulai simulasi
        self.start_time = datetime(2024, 1, 1, config.START_HOUR, config.START_MINUTE)
        
        # Set random seed
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
    
    def waktu_ke_jam(self, waktu_simulasi: float) -> datetime:
        """Konversi waktu simulasi (menit) ke datetime"""
        return self.start_time + timedelta(minutes=waktu_simulasi)
    
    def generate_lauk_time(self) -> float:
        """Generate waktu isi lauk (30-60 detik) dalam menit"""
        return random.uniform(self.config.MIN_LAUK_TIME, self.config.MAX_LAUK_TIME)
    
    def generate_angkut_time(self) -> float:
        """Generate waktu angkut (20-60 detik) dalam menit"""
        return random.uniform(self.config.MIN_ANGKUT_TIME, self.config.MAX_ANGKUT_TIME)
    
    def generate_nasi_time(self) -> float:
        """Generate waktu isi nasi (30-60 detik) dalam menit"""
        return random.uniform(self.config.MIN_NASI_TIME, self.config.MAX_NASI_TIME)
    
    def generate_ompreng_per_trip(self) -> int:
        """Generate jumlah ompreng per trip angkut (4-7)"""
        return random.randint(self.config.MIN_OMPRENG_PER_TRIP, self.config.MAX_OMPRENG_PER_TRIP)
    
    def proses_stage_lauk(self, ompreng_id: int):
        """Stage 1: Mengisi lauk ke ompreng"""
        waktu_mulai = self.env.now
        
        with self.petugas_lauk.request() as request:
            yield request
            
            # Catat utilisasi
            self.statistics['utilisasi_petugas']['lauk'].append({
                'time': self.env.now,
                'in_use': self.petugas_lauk.count
            })
            
            # Proses isi lauk
            lauk_time = self.generate_lauk_time()
            yield self.env.timeout(lauk_time)
        
        waktu_selesai = self.env.now
        
        # Simpan ke buffer stage berikutnya
        yield self.buffer_lauk_selesai.put({
            'ompreng_id': ompreng_id,
            'waktu_selesai_lauk': waktu_selesai,
            'waktu_lauk': lauk_time
        })
        
        self.statistics['stage_times']['lauk'].append(lauk_time)
    
    def proses_stage_angkut(self):
        """Stage 2: Mengangkat ompreng ke meja (batch processing)"""
        ompreng_batch = []
        trip_count = 0
        
        while len(ompreng_batch) < self.config.TOTAL_OMPRENG:
            # Kumpulkan ompreng dari buffer lauk
            item = yield self.buffer_lauk_selesai.get()
            ompreng_batch.append(item)
            
            # Cek apakah sudah cukup untuk satu trip atau ini ompreng terakhir
            ompreng_per_trip = self.generate_ompreng_per_trip()
            sisa_ompreng = self.config.TOTAL_OMPRENG - len(ompreng_batch)
            
            if len(ompreng_batch) >= ompreng_per_trip or sisa_ompreng <= 0:
                trip_count += 1
                # Proses angkut batch ini
                with self.petugas_angkut.request() as request:
                    yield request
                    
                    # Catat utilisasi
                    self.statistics['utilisasi_petugas']['angkut'].append({
                        'time': self.env.now,
                        'in_use': self.petugas_angkut.count,
                        'batch_size': len(ompreng_batch),
                        'trip': trip_count
                    })
                    
                    # Proses angkut
                    angkut_time = self.generate_angkut_time()
                    yield self.env.timeout(angkut_time)
                
                # Kirim ke buffer stage berikutnya
                for item in ompreng_batch:
                    yield self.buffer_angkut_selesai.put({
                        'ompreng_id': item['ompreng_id'],
                        'waktu_selesai_angkut': self.env.now,
                        'waktu_angkut': angkut_time / len(ompreng_batch),
                        'waktu_lauk': item['waktu_lauk'],
                        'waktu_selesai_lauk': item['waktu_selesai_lauk']
                    })
                
                self.statistics['stage_times']['angkut'].append(angkut_time)
                ompreng_batch = []
    
    def proses_stage_nasi(self, ompreng_id: int, data_sebelumnya: dict):
        """Stage 3: Menambahkan nasi ke ompreng"""
        waktu_mulai = self.env.now
        
        with self.petugas_nasi.request() as request:
            yield request
            
            # Catat utilisasi
            self.statistics['utilisasi_petugas']['nasi'].append({
                'time': self.env.now,
                'in_use': self.petugas_nasi.count
            })
            
            # Proses isi nasi
            nasi_time = self.generate_nasi_time()
            yield self.env.timeout(nasi_time)
        
        waktu_selesai = self.env.now
        
        # Simpan data lengkap ompreng
        self.statistics['ompreng_data'].append({
            'id': ompreng_id,
            'waktu_lauk': data_sebelumnya['waktu_lauk'],
            'waktu_angkut': data_sebelumnya['waktu_angkut'],
            'waktu_nasi': nasi_time,
            'total_waktu': data_sebelumnya['waktu_lauk'] + data_sebelumnya['waktu_angkut'] + nasi_time,
            'waktu_selesai': waktu_selesai,
            'jam_selesai': self.waktu_ke_jam(waktu_selesai)
        })
        
        self.statistics['stage_times']['nasi'].append(nasi_time)
    
    def proses_kedatangan_ompreng(self):
        """Generate proses untuk semua ompreng"""
        # Start stage 1 untuk semua ompreng
        for i in range(self.config.TOTAL_OMPRENG):
            self.env.process(self.proses_stage_lauk(i))
        
        # Start stage 2 (batch processing)
        self.env.process(self.proses_stage_angkut())
        
        # Start stage 3 untuk semua ompreng dari buffer angkut
        processed_count = 0
        while processed_count < self.config.TOTAL_OMPRENG:
            try:
                data = yield self.buffer_angkut_selesai.get()
                self.env.process(self.proses_stage_nasi(
                    data['ompreng_id'], 
                    data
                ))
                processed_count += 1
            except:
                break
    
    def run_simulation(self):
        """Jalankan simulasi"""
        self.env.process(self.proses_kedatangan_ompreng())
        self.env.run()
        return self.analyze_results()
    
    def analyze_results(self):
        """Analisis hasil simulasi"""
        if not self.statistics['ompreng_data']:
            return None, None
        
        df = pd.DataFrame(self.statistics['ompreng_data'])
        
        results = {
            'total_ompreng': len(df),
            'waktu_selesai_terakhir': df['waktu_selesai'].max(),
            'jam_selesai_terakhir': self.waktu_ke_jam(df['waktu_selesai'].max()),
            'durasi_total_menit': df['waktu_selesai'].max(),
            
            # Statistik per stage
            'avg_lauk_time': np.mean(self.statistics['stage_times']['lauk']),
            'avg_angkut_time': np.mean(self.statistics['stage_times']['angkut']),
            'avg_nasi_time': np.mean(self.statistics['stage_times']['nasi']),
            
            'total_lauk_time': sum(self.statistics['stage_times']['lauk']),
            'total_angkut_time': sum(self.statistics['stage_times']['angkut']),
            'total_nasi_time': sum(self.statistics['stage_times']['nasi']),
            
            # Utilisasi petugas
            'utilisasi_petugas': self.calculate_utilization(df),
            
            # Jumlah trip angkut
            'total_trip_angkut': len(self.statistics['stage_times']['angkut'])
        }
        
        return results, df
    
    def calculate_utilization(self, df):
        """Hitung utilisasi petugas per stage"""
        total_time = df['waktu_selesai'].max()
        
        utilization = {}
        
        # Utilisasi stage lauk
        if self.statistics['stage_times']['lauk']:
            total_lauk = sum(self.statistics['stage_times']['lauk'])
            utilization['lauk'] = (total_lauk / (total_time * self.config.NUM_PETUGAS_LAUK)) * 100
        
        # Utilisasi stage angkut
        if self.statistics['stage_times']['angkut']:
            total_angkut = sum(self.statistics['stage_times']['angkut'])
            utilization['angkut'] = (total_angkut / (total_time * self.config.NUM_PETUGAS_ANGKUT)) * 100
        
        # Utilisasi stage nasi
        if self.statistics['stage_times']['nasi']:
            total_nasi = sum(self.statistics['stage_times']['nasi'])
            utilization['nasi'] = (total_nasi / (total_time * self.config.NUM_PETUGAS_NASI)) * 100
        
        return utilization

# ============================
# FUNGSI VISUALISASI PLOTLY
# ============================
def create_stage_time_distribution(df):
    """Histogram distribusi waktu per stage - seperti di Jupyter Notebook"""
    fig = px.histogram(
        df, 
        x='waktu_lauk',
        nbins=20,
        title='',
        labels={'waktu_lauk': 'Waktu (menit)', 'count': 'Frekuensi'},
        color_discrete_sequence=['#1f77b4'],
        opacity=0.7
    )
    
    # Tambah histogram untuk stage lain
    fig.add_trace(px.histogram(df, x='waktu_angkut', nbins=20, opacity=0.7,
                              color_discrete_sequence=['#ff7f0e']).data[0])
    fig.add_trace(px.histogram(df, x='waktu_nasi', nbins=20, opacity=0.7,
                              color_discrete_sequence=['#2ca02c']).data[0])
    
    fig.update_layout(
        title='📊 Distribusi Waktu per Stage',
        xaxis_title="Waktu (menit)",
        yaxis_title="Frekuensi",
        showlegend=True,
        legend_title='Stage',
        hovermode="x unified",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    # Update legend labels
    fig.data[0].name = 'Stage 1 (Lauk)'
    fig.data[1].name = 'Stage 2 (Angkut)'
    fig.data[2].name = 'Stage 3 (Nasi)'
    
    return fig

def create_timeline_chart(df):
    """Timeline scatter plot penyelesaian ompreng - seperti di Jupyter Notebook"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['waktu_selesai'],
        mode='markers',
        name='Waktu Selesai',
        marker=dict(size=6, color='green', opacity=0.6),
        hovertemplate='Ompreng ID: %{x}<br>Waktu: %{y:.1f} menit<extra></extra>'
    ))
    
    fig.update_layout(
        title='',
        xaxis_title="ID Ompreng",
        yaxis_title="Waktu Selesai (menit)",
        hovermode="closest",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_utilization_bar_chart(results):
    """Bar chart utilisasi petugas per stage - seperti di Jupyter Notebook"""
    stages = list(results['utilisasi_petugas'].keys())
    utils = list(results['utilisasi_petugas'].values())
    stage_names = ['Stage 1\n(Lauk)', 'Stage 2\n(Angkut)', 'Stage 3\n(Nasi)']
    
    fig = px.bar(
        x=stage_names,
        y=utils,
        title='',
        labels={'x': 'Stage', 'y': 'Utilisasi (%)'},
        color=utils,
        color_continuous_scale='Viridis',
        opacity=0.8
    )
    
    # Tambah garis target 80%
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="Target 80%", annotation_position="top right")
    
    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Utilisasi (%)",
        coloraxis_showscale=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis_range=[0, 100]
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_total_time_breakdown(results):
    """Bar chart total waktu per stage - seperti di Jupyter Notebook"""
    total_times = [results['total_lauk_time'], results['total_angkut_time'], results['total_nasi_time']]
    stage_names = ['Lauk', 'Angkut', 'Nasi']
    
    fig = px.bar(
        x=stage_names,
        y=total_times,
        title='',
        labels={'x': 'Stage', 'y': 'Total Waktu (menit)'},
        color=total_times,
        color_continuous_scale='Viridis',
        opacity=0.8
    )
    
    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Total Waktu (menit)",
        coloraxis_showscale=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    return fig

# ============================
# APLIKASI STREAMLIT
# ============================
def main():
    st.set_page_config(
        page_title="Simulasi Piket Kantin IT Del",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS untuk tampilan yang lebih gelap
    st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        .stAlert {
            background-color: #1e293b;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar untuk input parameter
    with st.sidebar:
        st.subheader("⚙️ Parameter Simulasi")
        
        num_meja = st.number_input(
            "Jumlah Meja", 
            min_value=10, 
            max_value=100, 
            value=60,
            step=10,
            help="Total meja yang harus dilayani"
        )
        
        mahasiswa_per_meja = st.number_input(
            "Mahasiswa per Meja", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Jumlah mahasiswa per meja"
        )
        
        st.markdown("---")
        
        st.subheader("👥 Alokasi Petugas")
        
        total_petugas = st.number_input(
            "Total Petugas", 
            min_value=3, 
            max_value=15, 
            value=7,
            help="Total petugas piket"
        )
        
        petugas_lauk = st.slider(
            "Petugas Stage 1 (Lauk)", 
            1, 5, 2,
            help="Petugas untuk mengisi lauk"
        )
        
        petugas_angkut = st.slider(
            "Petugas Stage 2 (Angkut)", 
            1, 5, 2,
            help="Petugas untuk mengangkat ompreng"
        )
        
        petugas_nasi = st.slider(
            "Petugas Stage 3 (Nasi)", 
            1, 5, 3,
            help="Petugas untuk mengisi nasi"
        )
        
        # Validasi total petugas
        if petugas_lauk + petugas_angkut + petugas_nasi != total_petugas:
            st.warning(f"⚠️ Total petugas ({petugas_lauk + petugas_angkut + petugas_nasi}) tidak sesuai dengan setting ({total_petugas})")
        
        st.markdown("---")
        
        st.subheader("⏱️ Parameter Waktu Layanan")
        
        min_lauk = st.slider(
            "Lauk Minimum (menit)",
            min_value=0.3,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        
        max_lauk = st.slider(
            "Lauk Maksimum (menit)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1
        )
        
        min_angkut = st.slider(
            "Angkut Minimum (menit)",
            min_value=0.2,
            max_value=1.0,
            value=0.33,
            step=0.1
        )
        
        max_angkut = st.slider(
            "Angkut Maksimum (menit)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
        
        min_nasi = st.slider(
            "Nasi Minimum (menit)",
            min_value=0.3,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        
        max_nasi = st.slider(
            "Nasi Maksimum (menit)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1
        )
        
        st.markdown("---")
        
        st.subheader("🕐 Waktu Mulai")
        
        start_hour = st.slider(
            "Jam Mulai", 
            0, 23, 7
        )
        
        start_minute = st.slider(
            "Menit Mulai", 
            0, 59, 0
        )
        
        st.markdown("---")
        
        run_simulation = st.button(
            "🚀 Jalankan Simulasi",
            type="primary",
            use_container_width=True
        )
        
        reset_params = st.button(
            "🔄 Reset Parameter",
            use_container_width=True
        )
        
        if reset_params:
            st.rerun()
    
    # Header utama
    st.title("🍽️ Simulasi Sistem Piket Kantin IT Del")
    st.markdown("""
    **Simulasi Discrete Event System (DES)** untuk analisis kinerja sistem piket kantin 
    dengan 3 stage proses: Lauk → Angkut → Nasi
    """)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
        st.session_state.df = None
        st.session_state.model = None
    
    # Jika tombol di-klik, jalankan simulasi
    if run_simulation:
        with st.spinner("Menjalankan simulasi..."):
            total_ompreng = num_meja * mahasiswa_per_meja
            
            # Setup konfigurasi
            config = Config(
                NUM_PETUGAS=total_petugas,
                NUM_MEJA=num_meja,
                MAHASISWA_PER_MEJA=mahasiswa_per_meja,
                TOTAL_OMPRENG=total_ompreng,
                NUM_PETUGAS_LAUK=petugas_lauk,
                NUM_PETUGAS_ANGKUT=petugas_angkut,
                NUM_PETUGAS_NASI=petugas_nasi,
                MIN_LAUK_TIME=min_lauk,
                MAX_LAUK_TIME=max_lauk,
                MIN_ANGKUT_TIME=min_angkut,
                MAX_ANGKUT_TIME=max_angkut,
                MIN_NASI_TIME=min_nasi,
                MAX_NASI_TIME=max_nasi,
                START_HOUR=start_hour,
                START_MINUTE=start_minute
            )
            
            # Jalankan simulasi
            model = PiketKantinDES(config)
            results, df = model.run_simulation()
            
            if results:
                st.session_state.results = results
                st.session_state.df = df
                st.session_state.model = model
                st.success(f"✅ Simulasi selesai! {total_ompreng} ompreng diproses.")
            else:
                st.error("❌ Gagal menjalankan simulasi!")
    
    # Tampilan default sebelum simulasi dijalankan
    if st.session_state.results is None:
        st.info("""
        ### 🚀 Mulai Simulasi
        
        **Langkah-langkah:**
        1. Atur parameter simulasi di sidebar kiri
        2. Klik tombol **"Jalankan Simulasi"** 
        3. Tunggu proses simulasi selesai
        4. Lihat hasil dan visualisasi
        
        **Parameter default:**
        - Total Petugas: 7 orang
        - Jumlah Meja: 60 meja
        - Mahasiswa/Meja: 3 orang
        - Total Ompreng: 180
        - Waktu Mulai: 07:00
        """)
        
        # Preview chart kosong
        st.markdown("---")
        st.subheader("🎯 Preview Visualisasi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("📊 **Distribusi Waktu per Stage**")
            st.info("Chart akan muncul setelah simulasi dijalankan")
        
        with col2:
            st.write("📈 **Timeline Penyelesaian**")
            st.info("Chart akan muncul setelah simulasi dijalankan")
    else:
        # Tampilkan summary metrics
        results = st.session_state.results
        df = st.session_state.df
        model = st.session_state.model
        
        # Metrics utama
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "⏰ Jam Selesai Terakhir",
                results['jam_selesai_terakhir'].strftime('%H:%M')
            )
        
        with col2:
            st.metric(
                "⏱️ Durasi Total",
                f"{results['durasi_total_menit']:.2f} menit"
            )
        
        with col3:
            st.metric(
                "🍱 Total Ompreng",
                f"{results['total_ompreng']}"
            )
        
        with col4:
            avg_util = np.mean(list(results['utilisasi_petugas'].values()))
            st.metric(
                "📈 Utilisasi Rata-rata",
                f"{avg_util:.1f}%"
            )
        
        # Tampilkan detail hasil
        with st.expander("📋 Detail Hasil Simulasi", expanded=False):
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Statistik Waktu per Stage")
                st.write(f"**Stage 1 (Lauk):** {results['avg_lauk_time']:.2f} menit (rata-rata)")
                st.write(f"**Stage 2 (Angkut):** {results['avg_angkut_time']:.2f} menit (rata-rata)")
                st.write(f"**Stage 3 (Nasi):** {results['avg_nasi_time']:.2f} menit (rata-rata)")
                
                st.subheader("Total Waktu per Stage")
                st.write(f"**Stage 1 (Lauk):** {results['total_lauk_time']:.2f} menit")
                st.write(f"**Stage 2 (Angkut):** {results['total_angkut_time']:.2f} menit")
                st.write(f"**Stage 3 (Nasi):** {results['total_nasi_time']:.2f} menit")
            
            with col_right:
                st.subheader("Utilisasi Petugas")
                stage_names = {'lauk': 'Stage 1 (Lauk)', 'angkut': 'Stage 2 (Angkut)', 'nasi': 'Stage 3 (Nasi)'}
                for stage, util in results['utilisasi_petugas'].items():
                    st.write(f"**{stage_names[stage]}:** {util:.1f}%")
                
                st.subheader("Parameter Simulasi")
                st.write(f"**Jumlah Meja:** {num_meja}")
                st.write(f"**Mahasiswa/Meja:** {mahasiswa_per_meja}")
                st.write(f"**Total Ompreng:** {num_meja * mahasiswa_per_meja}")
                st.write(f"**Total Petugas:** {total_petugas}")
                st.write(f"**Waktu Mulai:** {start_hour:02d}:{start_minute:02d}")
        
        # VISUALISASI
        st.markdown("---")
        st.header("📊 Visualisasi Hasil")
        
        # Baris 1: Distribusi waktu per stage dan timeline
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("📊 **Distribusi Waktu per Stage**")
            fig_stage = create_stage_time_distribution(df)
            st.plotly_chart(fig_stage, use_container_width=True)
        
        with col2:
            st.markdown("📈 **Timeline Penyelesaian Ompreng**")
            fig_timeline = create_timeline_chart(df)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Baris 2: Utilisasi dan total waktu
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("👥 **Utilisasi Petugas per Stage**")
            fig_util = create_utilization_bar_chart(results)
            st.plotly_chart(fig_util, use_container_width=True)
        
        with col4:
            st.markdown("⏱️ **Total Waktu per Stage**")
            fig_breakdown = create_total_time_breakdown(results)
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Tampilkan data tabel
        st.markdown("---")
        st.subheader("📄 Data Hasil Simulasi")
        
        with st.expander("Lihat Data", expanded=False):
            st.dataframe(
                df.sort_values('id'),
                column_config={
                    "id": st.column_config.NumberColumn("ID Ompreng"),
                    "waktu_lauk": st.column_config.NumberColumn("Waktu Lauk", format="%.2f"),
                    "waktu_angkut": st.column_config.NumberColumn("Waktu Angkut", format="%.2f"),
                    "waktu_nasi": st.column_config.NumberColumn("Waktu Nasi", format="%.2f"),
                    "total_waktu": st.column_config.NumberColumn("Total Waktu", format="%.2f"),
                    "jam_selesai": st.column_config.DatetimeColumn("Waktu Selesai"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Tombol download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data CSV",
                data=csv,
                file_name=f"simulasi_piket_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Rekomendasi
        st.markdown("---")
        st.subheader("💡 Rekomendasi Berdasarkan Hasil")
        
        avg_util = np.mean(list(results['utilisasi_petugas'].values()))
        max_util_stage = max(results['utilisasi_petugas'], key=results['utilisasi_petugas'].get)
        
        if avg_util > 80:
            st.warning("⚠️ **Utilisasi tinggi!** Pertimbangkan untuk menambah petugas agar tidak terjadi bottleneck.")
        elif avg_util < 50:
            st.info("ℹ️ **Utilisasi rendah.** Sistem memiliki kapasitas berlebih, bisa mengurangi petugas.")
        else:
            st.success("✅ **Utilisasi optimal.** Sistem berjalan dengan baik.")
        
        stage_names = {'lauk': 'Stage 1 (Lauk)', 'angkut': 'Stage 2 (Angkut)', 'nasi': 'Stage 3 (Nasi)'}
        st.write(f"**Stage dengan utilisasi tertinggi:** {stage_names[max_util_stage]} ({results['utilisasi_petugas'][max_util_stage]:.1f}%)")
        st.write(f"**Rekomendasi:** Fokus optimasi pada {stage_names[max_util_stage]} untuk meningkatkan kinerja sistem.")
    
    # Footer
    st.markdown("---")
    st.caption(
        f"**MODSIM: Discrete Event System (DES)** | "
        f"Studi Kasus 2.1 - Sistem Piket Kantin IT Del | "
        f"Terakhir diupdate: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )

if __name__ == "__main__":
    main()