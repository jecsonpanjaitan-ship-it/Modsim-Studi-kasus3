import streamlit as st
import simpy
import random
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# ============================
# KONFIGURASI SIMULASI
# ============================
@dataclass
class Config:
    """Konfigurasi parameter simulasi piket kantin IT Del"""
    NUM_PETUGAS: int = 7
    NUM_MEJA: int = 60
    MAHASISWA_PER_MEJA: int = 3
    TOTAL_OMPRENG: int = 180
    
    NUM_PETUGAS_LAUK: int = 2
    NUM_PETUGAS_ANGKUT: int = 2
    NUM_PETUGAS_NASI: int = 3
    
    MIN_LAUK_TIME: float = 0.5
    MAX_LAUK_TIME: float = 1.0
    
    MIN_ANGKUT_TIME: float = 0.33
    MAX_ANGKUT_TIME: float = 1.0
    
    MIN_NASI_TIME: float = 0.5
    MAX_NASI_TIME: float = 1.0
    
    MIN_OMPRENG_PER_TRIP: int = 4
    MAX_OMPRENG_PER_TRIP: int = 7
    
    START_HOUR: int = 7
    START_MINUTE: int = 0
    RANDOM_SEED: int = 42

# ============================
# MODEL SIMULASI
# ============================
class PiketKantinDES:
    """Model Discrete Event Simulation untuk piket kantin IT Del"""
    
    def __init__(self, config):
        self.config = config
        self.env = simpy.Environment()
        
        self.petugas_lauk = simpy.Resource(self.env, capacity=config.NUM_PETUGAS_LAUK)
        self.petugas_angkut = simpy.Resource(self.env, capacity=config.NUM_PETUGAS_ANGKUT)
        self.petugas_nasi = simpy.Resource(self.env, capacity=config.NUM_PETUGAS_NASI)
        
        self.buffer_lauk_selesai = simpy.Store(self.env)
        self.buffer_angkut_selesai = simpy.Store(self.env)
        
        self.statistics = {
            'ompreng_data': [],
            'stage_times': {'lauk': [], 'angkut': [], 'nasi': []},
            'utilisasi_petugas': {'lauk': [], 'angkut': [], 'nasi': []}
        }
        
        self.start_time = datetime(2024, 1, 1, config.START_HOUR, config.START_MINUTE)
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
    
    def waktu_ke_jam(self, waktu_simulasi):
        return self.start_time + timedelta(minutes=waktu_simulasi)
    
    def generate_lauk_time(self):
        return random.uniform(self.config.MIN_LAUK_TIME, self.config.MAX_LAUK_TIME)
    
    def generate_angkut_time(self):
        return random.uniform(self.config.MIN_ANGKUT_TIME, self.config.MAX_ANGKUT_TIME)
    
    def generate_nasi_time(self):
        return random.uniform(self.config.MIN_NASI_TIME, self.config.MAX_NASI_TIME)
    
    def generate_ompreng_per_trip(self):
        return random.randint(self.config.MIN_OMPRENG_PER_TRIP, self.config.MAX_OMPRENG_PER_TRIP)
    
    def proses_stage_lauk(self, ompreng_id):
        with self.petugas_lauk.request() as request:
            yield request
            self.statistics['utilisasi_petugas']['lauk'].append({
                'time': self.env.now,
                'in_use': self.petugas_lauk.count
            })
            lauk_time = self.generate_lauk_time()
            yield self.env.timeout(lauk_time)
        
        yield self.buffer_lauk_selesai.put({
            'ompreng_id': ompreng_id,
            'waktu_lauk': lauk_time
        })
        self.statistics['stage_times']['lauk'].append(lauk_time)
    
    def proses_stage_angkut(self):
        ompreng_batch = []
        trip_count = 0
        ompreng_selesai_lauk = 0
        
        while ompreng_selesai_lauk < self.config.TOTAL_OMPRENG or len(ompreng_batch) > 0:
            if len(self.buffer_lauk_selesai.items) > 0:
                item = yield self.buffer_lauk_selesai.get()
                ompreng_batch.append(item)
                ompreng_selesai_lauk += 1
            else:
                yield self.env.timeout(0.01)
                continue
            
            ompreng_per_trip = self.generate_ompreng_per_trip()
            
            if len(ompreng_batch) >= ompreng_per_trip or ompreng_selesai_lauk >= self.config.TOTAL_OMPRENG:
                trip_count += 1
                
                with self.petugas_angkut.request() as request:
                    yield request
                    self.statistics['utilisasi_petugas']['angkut'].append({
                        'time': self.env.now,
                        'in_use': self.petugas_angkut.count,
                        'batch_size': len(ompreng_batch),
                        'trip': trip_count
                    })
                    angkut_time = self.generate_angkut_time()
                    yield self.env.timeout(angkut_time)
                
                for item in ompreng_batch:
                    yield self.buffer_angkut_selesai.put({
                        'ompreng_id': item['ompreng_id'],
                        'waktu_angkut': angkut_time / len(ompreng_batch),
                        'waktu_lauk': item['waktu_lauk']
                    })
                
                self.statistics['stage_times']['angkut'].append(angkut_time)
                ompreng_batch = []
    
    def proses_stage_nasi(self, ompreng_id, data_sebelumnya):
        with self.petugas_nasi.request() as request:
            yield request
            self.statistics['utilisasi_petugas']['nasi'].append({
                'time': self.env.now,
                'in_use': self.petugas_nasi.count
            })
            nasi_time = self.generate_nasi_time()
            yield self.env.timeout(nasi_time)
        
        self.statistics['ompreng_data'].append({
            'id': ompreng_id,
            'waktu_lauk': data_sebelumnya['waktu_lauk'],
            'waktu_angkut': data_sebelumnya['waktu_angkut'],
            'waktu_nasi': nasi_time,
            'total_waktu': data_sebelumnya['waktu_lauk'] + data_sebelumnya['waktu_angkut'] + nasi_time,
            'waktu_selesai': self.env.now,
            'jam_selesai': self.waktu_ke_jam(self.env.now)
        })
        self.statistics['stage_times']['nasi'].append(nasi_time)
    
    def proses_kedatangan_ompreng(self):
        for i in range(self.config.TOTAL_OMPRENG):
            self.env.process(self.proses_stage_lauk(i))
        self.env.process(self.proses_stage_angkut())
        
        ompreng_diproses = 0
        while ompreng_diproses < self.config.TOTAL_OMPRENG:
            if len(self.buffer_angkut_selesai.items) > 0:
                data = yield self.buffer_angkut_selesai.get()
                self.env.process(self.proses_stage_nasi(data['ompreng_id'], data))
                ompreng_diproses += 1
            else:
                yield self.env.timeout(0.01)
    
    def run_simulation(self):
        self.env.process(self.proses_kedatangan_ompreng())
        self.env.run()
        return self.analyze_results()
    
    def analyze_results(self):
        if not self.statistics['ompreng_data']:
            return None, None
        
        df = pd.DataFrame(self.statistics['ompreng_data'])
        
        results = {
            'total_ompreng': len(df),
            'waktu_selesai_terakhir': df['waktu_selesai'].max(),
            'jam_selesai_terakhir': self.waktu_ke_jam(df['waktu_selesai'].max()),
            'durasi_total_menit': df['waktu_selesai'].max(),
            'avg_lauk_time': np.mean(self.statistics['stage_times']['lauk']),
            'avg_angkut_time': np.mean(self.statistics['stage_times']['angkut']),
            'avg_nasi_time': np.mean(self.statistics['stage_times']['nasi']),
            'total_lauk_time': sum(self.statistics['stage_times']['lauk']),
            'total_angkut_time': sum(self.statistics['stage_times']['angkut']),
            'total_nasi_time': sum(self.statistics['stage_times']['nasi']),
            'utilisasi_petugas': self.calculate_utilization(df),
            'total_trip_angkut': len(self.statistics['stage_times']['angkut'])
        }
        
        return results, df
    
    def calculate_utilization(self, df):
        total_time = df['waktu_selesai'].max()
        utilization = {}
        
        if self.statistics['stage_times']['lauk']:
            total_lauk = sum(self.statistics['stage_times']['lauk'])
            utilization['lauk'] = (total_lauk / (total_time * self.config.NUM_PETUGAS_LAUK)) * 100
        
        if self.statistics['stage_times']['angkut']:
            total_angkut = sum(self.statistics['stage_times']['angkut'])
            utilization['angkut'] = (total_angkut / (total_time * self.config.NUM_PETUGAS_ANGKUT)) * 100
        
        if self.statistics['stage_times']['nasi']:
            total_nasi = sum(self.statistics['stage_times']['nasi'])
            utilization['nasi'] = (total_nasi / (total_time * self.config.NUM_PETUGAS_NASI)) * 100
        
        return utilization

# ============================
# FUNGSI VISUALISASI PLOTLY
# ============================
def create_stage_time_distribution(df):
    fig = px.histogram(df, x='waktu_lauk', nbins=20, title='Distribusi Waktu per Stage',
        labels={'waktu_lauk': 'Waktu (menit)', 'count': 'Frekuensi'},
        color_discrete_sequence=['#3B82F6'], opacity=0.7)
    fig.add_trace(px.histogram(df, x='waktu_angkut', nbins=20, opacity=0.7,
        color_discrete_sequence=['#F97316']).data[0])
    fig.add_trace(px.histogram(df, x='waktu_nasi', nbins=20, opacity=0.7,
        color_discrete_sequence=['#10B981']).data[0])
    
    fig.update_layout(xaxis_title="Waktu (menit)", yaxis_title="Frekuensi",
        showlegend=True, legend_title='Stage', hovermode="x unified",
        height=400, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)')
    fig.data[0].name = 'Stage 1 (Lauk)'
    fig.data[1].name = 'Stage 2 (Angkut)'
    fig.data[2].name = 'Stage 3 (Nasi)'
    return fig

def create_timeline_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['waktu_selesai'], mode='markers',
        name='Waktu Selesai',
        marker=dict(size=6, color='#10B981', opacity=0.6)))
    fig.update_layout(title='Timeline Penyelesaian Ompreng',
        xaxis_title="ID Ompreng", yaxis_title="Waktu Selesai (menit)",
        height=400, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_utilization_bar_chart(results):
    stages = list(results['utilisasi_petugas'].keys())
    utils = list(results['utilisasi_petugas'].values())
    stage_names = ['Stage 1\n(Lauk)', 'Stage 2\n(Angkut)', 'Stage 3\n(Nasi)']
    
    fig = px.bar(x=stage_names, y=utils, title='Utilisasi Petugas per Stage',
        labels={'x': 'Stage', 'y': 'Utilisasi (%)'},
        color=utils, color_continuous_scale='Viridis', opacity=0.8)
    fig.add_hline(y=80, line_dash="dash", line_color="#EF4444",
        annotation_text="Target 80%")
    
    fig.update_layout(height=400, yaxis_range=[0, 100],
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_total_time_breakdown(results):
    total_times = [results['total_lauk_time'], results['total_angkut_time'], results['total_nasi_time']]
    stage_names = ['Lauk', 'Angkut', 'Nasi']
    
    fig = px.bar(x=stage_names, y=total_times, title='Total Waktu per Stage',
        labels={'x': 'Stage', 'y': 'Total Waktu (menit)'},
        color=total_times, color_continuous_scale='Viridis', opacity=0.8)
    
    fig.update_layout(height=400,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_efficiency_gauge(results):
    avg_util = np.mean(list(results['utilisasi_petugas'].values()))
    
    if avg_util >= 80:
        color = "#10B981"
        status = "Optimal"
    elif avg_util >= 60:
        color = "#F97316"
        status = "Cukup"
    else:
        color = "#EF4444"
        status = "Perlu Perbaikan"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_util,
        title={'text': f"Efisiensi Sistem<br><span style='font-size:14px'>{status}</span>"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [60, 80], 'color': 'rgba(249, 115, 22, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# ============================
# APLIKASI STREAMLIT
# ============================
st.set_page_config(
    page_title="Simulasi Piket Kantin IT Del - ifs25026",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0E1117 0%, #1E293B 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0E1117 100%);
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #10B981;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #94A3B8;
    }
    .stButton>button {
        background: linear-gradient(90deg, #EF4444 0%, #F97316 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #475569;
    }
    .main-header {
        font-size: 32px;
        font-weight: bold;
        background: linear-gradient(90deg, #EF4444, #F97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #94A3B8;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #64748B;
        font-size: 12px;
        border-top: 1px solid #334155;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.markdown("## 🍽️")
with col_title:
    st.markdown('<p class="main-header">Simulasi Piket Kantin IT Del</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discrete Event System (DES) - Studi Kasus 2.1 | ifs25026</p>', unsafe_allow_html=True)
st.markdown("---")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Parameter Simulasi")
    
    st.subheader("📊 Parameter Dasar")
    num_meja = st.number_input("Jumlah Meja", min_value=10, max_value=100, value=60, step=10)
    mahasiswa_per_meja = st.number_input("Mahasiswa per Meja", min_value=1, max_value=5, value=3)
    total_ompreng = num_meja * mahasiswa_per_meja
    st.info(f"🍱 **Total Ompreng: {total_ompreng}**")
    
    st.subheader("👥 Alokasi Petugas")
    total_petugas = st.number_input("Total Petugas", min_value=3, max_value=15, value=7)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        petugas_lauk = st.number_input("Lauk", min_value=1, max_value=5, value=2)
    with col2:
        petugas_angkut = st.number_input("Angkut", min_value=1, max_value=5, value=2)
    with col3:
        petugas_nasi = st.number_input("Nasi", min_value=1, max_value=5, value=3)
    
    if petugas_lauk + petugas_angkut + petugas_nasi != total_petugas:
        st.warning(f"⚠️ Total ({petugas_lauk + petugas_angkut + petugas_nasi}) ≠ {total_petugas}")
    
    st.subheader("⏱️ Waktu Layanan (menit)")
    min_lauk = st.slider("Lauk Min", 0.3, 2.0, 0.5, 0.1)
    max_lauk = st.slider("Lauk Max", 0.5, 3.0, 1.0, 0.1)
    min_angkut = st.slider("Angkut Min", 0.2, 1.0, 0.33, 0.1)
    max_angkut = st.slider("Angkut Max", 0.5, 2.0, 1.0, 0.1)
    min_nasi = st.slider("Nasi Min", 0.3, 2.0, 0.5, 0.1)
    max_nasi = st.slider("Nasi Max", 0.5, 3.0, 1.0, 0.1)
    
    st.subheader("🕐 Waktu Mulai")
    start_hour = st.slider("Jam", 0, 23, 7)
    start_minute = st.slider("Menit", 0, 59, 0)
    
    st.markdown("---")
    run_simulation = st.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)

# MAIN CONTENT
if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.df = None
    st.session_state.model = None

if run_simulation:
    with st.spinner("Menjalankan simulasi..."):
        config = Config(
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
        
        model = PiketKantinDES(config)
        results, df = model.run_simulation()
        
        if results:
            st.session_state.results = results
            st.session_state.df = df
            st.session_state.model = model
            st.success(f"✅ Simulasi selesai! **{results['total_ompreng']}** ompreng diproses.")
        else:
            st.error("❌ Simulasi gagal!")

# Tampilkan hasil jika ada
if st.session_state.results is not None:
    results = st.session_state.results
    df = st.session_state.df
    model = st.session_state.model
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏰ Jam Selesai", results['jam_selesai_terakhir'].strftime('%H:%M'))
    with col2:
        st.metric("⏱️ Durasi Total", f"{results['durasi_total_menit']:.2f} menit")
    with col3:
        st.metric("🍱 Total Ompreng", f"{results['total_ompreng']}")
    with col4:
        avg_util = np.mean(list(results['utilisasi_petugas'].values()))
        st.metric("📈 Utilisasi Rata-rata", f"{avg_util:.1f}%")
    
    st.markdown("---")
    st.subheader("📊 Visualisasi Hasil")
    
    # Visualisasi dengan Plotly
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribusi Waktu per Stage**")
        fig_stage = create_stage_time_distribution(df)
        st.plotly_chart(fig_stage, use_container_width=True)
    with col2:
        st.markdown("**Timeline Penyelesaian**")
        fig_timeline = create_timeline_chart(df)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Utilisasi Petugas**")
        fig_util = create_utilization_bar_chart(results)
        st.plotly_chart(fig_util, use_container_width=True)
    with col4:
        st.markdown("**Total Waktu per Stage**")
        fig_breakdown = create_total_time_breakdown(results)
        st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Gauge Efficiency
    col5, col6 = st.columns([2, 1])
    with col5:
        st.markdown("**📄 Data Hasil Simulasi**")
        with st.expander("Lihat Data"):
            st.dataframe(df.sort_values('id'), use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download CSV", data=csv,
                file_name=f"simulasi_piket_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv")
    with col6:
        st.markdown("**Efisiensi Sistem**")
        fig_gauge = create_efficiency_gauge(results)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Rekomendasi
    st.markdown("---")
    st.subheader("💡 Rekomendasi")
    avg_util = np.mean(list(results['utilisasi_petugas'].values()))
    max_util_stage = max(results['utilisasi_petugas'], key=results['utilisasi_petugas'].get)
    
    if avg_util > 80:
        st.warning("⚠️ **Utilisasi tinggi!** Pertimbangkan menambah petugas.")
    elif avg_util < 50:
        st.info("ℹ️ **Utilisasi rendah.** Sistem memiliki kapasitas berlebih.")
    else:
        st.success("✅ **Utilisasi optimal.** Sistem berjalan dengan baik.")
    
    stage_names = {'lauk': 'Stage 1 (Lauk)', 'angkut': 'Stage 2 (Angkut)', 'nasi': 'Stage 3 (Nasi)'}
    st.write(f"**Stage dengan utilisasi tertinggi:** {stage_names[max_util_stage]} ({results['utilisasi_petugas'][max_util_stage]:.1f}%)")

else:
    st.info("👈 Klik tombol **Jalankan Simulasi** di sidebar untuk memulai")

# FOOTER
st.markdown("---")
st.caption(f"**MODSIM: Discrete Event System** | ifs25026 | Terakhir diupdate: {datetime.now().strftime('%d/%m/%Y %H:%M')}")