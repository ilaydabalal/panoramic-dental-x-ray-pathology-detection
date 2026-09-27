import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2
from datetime import datetime
from fpdf import FPDF
import plotly.graph_objects as go

st.set_page_config(page_title="Dental AI Professional", layout="wide", initial_sidebar_state="expanded")

for f in ["kayitlar", "raporlar"]:
    if not os.path.exists(f): os.makedirs(f)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

ACILIYET_SISTEMI = {
    'Periapical lesion': {'sev': 1, 'tag': '🚨 ACİL', 'clr': '#FF4444', 'bg': '#2D1B1B'},
    'Caries': {'sev': 2, 'tag': '⚠️ ÖNCELİKLİ', 'clr': '#FF9500', 'bg': '#2D2419'},
    'Root Canal Treatment': {'sev': 3, 'tag': '🟡 TAKİP', 'clr': '#FFD700', 'bg': '#2D2B19'},
    'Retained root': {'sev': 3, 'tag': '🟡 TAKİP', 'clr': '#FFD700', 'bg': '#2D2B19'},
    'Filling': {'sev': 4, 'tag': '🟢 RUTİN', 'clr': '#00D9FF', 'bg': '#19262D'},
    'Crown': {'sev': 4, 'tag': '🟢 RUTİN', 'clr': '#00D9FF', 'bg': '#19262D'},
    'Implant': {'sev': 4, 'tag': '🟢 RUTİN', 'clr': '#00D9FF', 'bg': '#19262D'},
    'impacted tooth': {'sev': 4, 'tag': '🟢 RUTİN', 'clr': '#00D9FF', 'bg': '#19262D'},
    'Missing teeth': {'sev': 5, 'tag': '⚪ BİLGİ', 'clr': '#8B96A0', 'bg': '#1E2329'},
    'Mandibular Canal': {'sev': 5, 'tag': '⚪ BİLGİ', 'clr': '#8B96A0', 'bg': '#1E2329'}
}

st.markdown("""
    <style>
    /* Genel Dark Theme */
    .main { background-color: #0D1117; padding: 0.5rem !important; }
    .stApp { background-color: #0D1117; }
    .block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { background-color: #161B22; }
    
    /* Kompakt İstatistik Kartları */
    .stat-mini {
        background: #161B22;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid #30363D;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: all 0.2s ease;
    }
    .stat-mini:hover {
        border-color: #58A6FF;
        transform: translateY(-2px);
    }
    
    .stat-icon {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    
    .stat-content {
        flex: 1;
        min-width: 0;
    }
    
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 2px;
    }
    
    .stat-label {
        color: #8B96A0;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar Özel Stil */
    .sidebar .stTextInput input {
        background-color: #0D1117 !important;
        color: #C9D1D9 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
        height: 40px !important;
    }
    
    .sidebar .stTextInput label {
        color: #C9D1D9 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    .sidebar .stSlider label {
        color: #C9D1D9 !important;
        font-size: 0.85rem !important;
    }
    
    /* Mini Triyaj Kartları */
    .mini-triage {
        background: #161B22;
        padding: 8px 10px;
        border-radius: 6px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 3px solid;
        font-size: 0.8rem;
        transition: all 0.2s ease;
    }
    .mini-triage:hover {
        transform: translateX(3px);
    }
    
    /* Butonlar */
    .stButton>button {
        border-radius: 8px !important;
        height: 2.8em !important;
        font-weight: 600;
        font-size: 0.9rem !important;
        border: 1px solid #30363D;
        background: #161B22;
        color: #C9D1D9;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #1F6FEB;
        border-color: #1F6FEB;
        color: white;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #161B22;
        border: 2px dashed #30363D;
        border-radius: 8px;
        padding: 20px;
    }
    
    /* Checkbox */
    .stCheckbox {
        background-color: #161B22;
        padding: 5px 10px;
        border-radius: 5px;
        margin-bottom: 4px;
        border: 1px solid #30363D;
    }
    .stCheckbox label { font-size: 0.85rem !important; }
    
    /* Başlıklar */
    h1, h2, h3 { color: #C9D1D9 !important; }
    h1 { font-size: 1.8rem !important; margin-bottom: 1rem !important; }
    h3 { font-size: 1rem !important; margin: 0.8rem 0 0.5rem 0 !important; }
    
    /* Grafik */
    .plotly-graph-div { height: 280px !important; }
    
    /* Divider */
    hr { 
        border-color: #30363D !important; 
        margin: 1rem 0 !important;
    }
    
    /* Warning/Info */
    .stWarning, .stInfo {
        background-color: #161B22;
        border-left: 4px solid #FF9500;
        padding: 12px;
        border-radius: 6px;
    }
    
    /* Image Container */
    .stImage { border-radius: 8px; overflow: hidden; }
    
    /* Sidebar Header */
    .sidebar-header {
        color: #58A6FF;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 10px 0;
        border-bottom: 2px solid #30363D;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DETAYLI PDF FONKSİYONU ---
def create_report(h_ad, h_id, dr_ad, img_path, bulgular, stats):
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font('ArialTR', '', 'C:\\Windows\\Fonts\\arial.ttf', uni=True)
        pdf.add_font('ArialTR', 'B', 'C:\\Windows\\Fonts\\arialbd.ttf', uni=True)
        pdf.add_font('ArialTR', 'I', 'C:\\Windows\\Fonts\\ariali.ttf', uni=True)
        f_name = 'ArialTR'
    except:
        f_name = 'Helvetica'
    
    # BAŞLIK
    pdf.set_font(f_name, 'B', 22)
    pdf.set_text_color(31, 97, 141)
    pdf.cell(190, 12, "DENTAL AI KLINIK ANALIZ RAPORU", ln=True, align='C')
    pdf.set_font(f_name, '', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(190, 6, "Yapay Zeka Destekli Panoramik Rontgen Analizi", ln=True, align='C')
    pdf.ln(5)
    
    # HASTA BİLGİLERİ KUTUSU
    pdf.set_fill_color(240, 248, 255)
    pdf.set_draw_color(31, 97, 141)
    pdf.rect(10, pdf.get_y(), 190, 25, 'D')
    
    pdf.set_font(f_name, 'B', 11)
    pdf.set_text_color(0, 0, 0)
    y_start = pdf.get_y() + 5
    pdf.set_xy(15, y_start)
    pdf.cell(90, 6, f"Hasta Ad Soyad: {h_ad}", ln=False)
    pdf.cell(90, 6, f"Rapor Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.set_x(15)
    pdf.cell(90, 6, f"TC / Protokol No: {h_id}", ln=False)
    pdf.cell(90, 6, f"Sorumlu Hekim: {dr_ad}", ln=True)
    pdf.ln(10)
    
    # İSTATİSTİK ÖZETİ
    pdf.set_font(f_name, 'B', 14)
    pdf.set_text_color(31, 97, 141)
    pdf.cell(190, 8, "ANALIZ OZETI", ln=True)
    pdf.ln(2)
    
    # İstatistik Kutuları
    box_width = 45
    box_height = 20
    x_start = 10
    y_pos = pdf.get_y()
    
    stats_data = [
        ("Toplam Bulgu", str(stats['total']), (70, 130, 180)),
        ("Acil Durum", str(stats['acil']), (220, 53, 69)),
        ("Oncelikli", str(stats['oncelik']), (230, 126, 34)),
        ("Ort. Guven", f"%{stats['avg_conf']:.0f}", (41, 128, 185))
    ]
    
    for idx, (label, value, color) in enumerate(stats_data):
        x_pos = x_start + (idx * (box_width + 2.5))
        
        pdf.set_fill_color(*color)
        pdf.set_draw_color(*color)
        pdf.rect(x_pos, y_pos, box_width, box_height, 'DF')
        
        pdf.set_font(f_name, 'B', 16)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(x_pos, y_pos + 4)
        pdf.cell(box_width, 6, value, align='C')
        
        pdf.set_font(f_name, '', 8)
        pdf.set_xy(x_pos, y_pos + 12)
        pdf.cell(box_width, 5, label, align='C')
    
    pdf.ln(25)
    
    # GÖRÜNTÜ
    pdf.set_font(f_name, 'B', 14)
    pdf.set_text_color(31, 97, 141)
    pdf.cell(190, 8, "AI ANALIZ GORUNTUSU", ln=True)
    pdf.ln(3)
    
    pdf.image(img_path, x=15, w=175)
    pdf.ln(5)
    
    # DETAYLI BULGULAR
    pdf.set_font(f_name, 'B', 14)
    pdf.set_text_color(31, 97, 141)
    pdf.cell(190, 8, "DETAYLI KLINIK BULGULAR", ln=True)
    pdf.ln(3)
    
    # Tablo Başlıkları
    pdf.set_fill_color(31, 97, 141)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(f_name, 'B', 10)
    pdf.cell(12, 8, "No", 1, 0, 'C', True)
    pdf.cell(50, 8, "Aciliyet", 1, 0, 'C', True)
    pdf.cell(85, 8, "Bulgu Adi", 1, 0, 'C', True)
    pdf.cell(43, 8, "Guven Skoru", 1, 1, 'C', True)
    
    # Bulgular
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(f_name, '', 9)
    
    for idx, b in enumerate(bulgular, 1):
        # Satır renklendirme
        if idx % 2 == 0:
            pdf.set_fill_color(245, 245, 245)
        else:
            pdf.set_fill_color(255, 255, 255)
        
        # Aciliyet rengine göre özel renk
        if b['sev'] == 1:
            pdf.set_text_color(220, 53, 69)
        elif b['sev'] == 2:
            pdf.set_text_color(230, 126, 34)
        elif b['sev'] == 3:
            pdf.set_text_color(241, 196, 15)
        else:
            pdf.set_text_color(0, 0, 0)
        
        clean_tag = b['tag'].replace('🚨','').replace('⚠️','').replace('🟡','').replace('🟢','').replace('⚪','').strip()
        
        pdf.cell(12, 7, str(idx), 1, 0, 'C', True)
        pdf.set_font(f_name, 'B', 9)
        pdf.cell(50, 7, clean_tag, 1, 0, 'L', True)
        pdf.set_font(f_name, '', 9)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(85, 7, b['name'], 1, 0, 'L', True)
        
        # Güven skoru için renk
        conf_percent = b['conf'] * 100
        if conf_percent >= 80:
            pdf.set_text_color(39, 174, 96)
        elif conf_percent >= 60:
            pdf.set_text_color(230, 126, 34)
        else:
            pdf.set_text_color(220, 53, 69)
        
        pdf.set_font(f_name, 'B', 9)
        pdf.cell(43, 7, f"%{conf_percent:.1f}", 1, 1, 'C', True)
        pdf.set_text_color(0, 0, 0)
    
    pdf.ln(5)
    
    # NOTLAR BÖLÜMÜ
    pdf.set_font(f_name, 'B', 12)
    pdf.set_text_color(31, 97, 141)
    pdf.cell(190, 8, "KLINIK NOTLAR VE ONERILER", ln=True)
    pdf.ln(2)
    
    pdf.set_font(f_name, '', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(190, 5, 
        "Bu rapor yapay zeka destekli Dental AI sistemi tarafindan olusturulmustur. "
        "Tespit edilen bulgular klinik muayene ile dogrulanmali ve kesin tani icin "
        "uzman hekim degerlendirmesi gereklidir. Acil ve oncelikli bulgular icin "
        "en kisa surede tedavi planlamasi onerilir.")
    
    pdf.ln(5)
    
    # ALT BİLGİ
    pdf.set_font(f_name, 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(190, 5, f"Rapor Olusturma Zamani: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 1, 'C')
    pdf.cell(190, 5, "Dental AI v2.0 - Yapay Zeka Destekli Dental Goruntuleme Sistemi", 0, 1, 'C')
    
    path = f"raporlar/{h_ad.replace(' ','_')}_{h_id}.pdf"
    pdf.output(path)
    return path

# --- KOMPAKT GRAFİK ---
def create_compact_chart(triage_list):
    sev_counts = {}
    for t in triage_list:
        tag = t['tag'].split()[1] if len(t['tag'].split()) > 1 else t['tag']
        sev_counts[tag] = sev_counts.get(tag, 0) + 1
    
    colors_map = {
        'ACİL': '#FF4444',
        'ÖNCELİKLİ': '#FF9500',
        'TAKİP': '#FFD700',
        'RUTİN': '#00D9FF',
        'BİLGİ': '#8B96A0'
    }
    
    fig = go.Figure(data=[go.Bar(
        x=list(sev_counts.keys()),
        y=list(sev_counts.values()),
        marker_color=[colors_map.get(k, '#8B96A0') for k in sev_counts.keys()],
        text=list(sev_counts.values()),
        textposition='outside',
    )])
    
    fig.update_layout(
        plot_bgcolor='#0D1117',
        paper_bgcolor='#161B22',
        font=dict(color='#C9D1D9', size=11),
        showlegend=False,
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor='#30363D', title=''),
        yaxis=dict(gridcolor='#30363D', title='')
    )
    
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='sidebar-header'>Hasta & Hekim Bilgileri</div>", unsafe_allow_html=True)
    
    dr_ad = st.text_input("Hekim Adı", "Dt. ", key="hekim")
    h_ad = st.text_input("Hasta Ad Soyad", key="hasta_ad")
    h_id = st.text_input("TC / Protokol No", key="hasta_tc")
    
    st.divider()
    
    st.markdown("<div class='sidebar-header'>Analiz Ayarları</div>", unsafe_allow_html=True)
    conf_val = st.slider("Tespit Hassasiyeti", 0.10, 1.0, 0.40, 0.05, 
                         help="Düşük: Daha fazla bulgu | Yüksek: Daha kesin sonuç")
    
    filter_box = st.empty()
    
    st.divider()
    st.markdown("<p style='color: #8B96A0; font-size: 0.8rem; text-align: center;'>Dental AI v2.0 • © 2024</p>", unsafe_allow_html=True)

# --- ANA ARAYÜZ ---
st.markdown("<h1 style='text-align: center; color: #58A6FF;'>Dental AI Karar Destek Sistemi</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Panoramik Röntgen Görüntüsü Yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file and h_ad and h_id:
    input_img = Image.open(uploaded_file)
    
    # Görüntü değiştiğinde tüm sınıfları bul
    if "active_file" not in st.session_state or st.session_state.active_file != uploaded_file.name:
        base_res = model.predict(source=input_img, conf=0.10, agnostic_nms=True, verbose=False)
        st.session_state.all_found = sorted(list(set([model.names[int(b.cls[0])] for b in base_res[0].boxes])))
        st.session_state.active_file = uploaded_file.name

    # Filtre Seçimi
    selected = []
    with filter_box.container():
        st.markdown("**Bulgu Filtreleme**")
        for c in st.session_state.get('all_found', []):
            if st.checkbox(c, value=True, key=f"f_{c}"):
                selected.append(c)

    # Analiz
    sel_ids = [i for i, n in model.names.items() if n in selected]
    results = model.predict(source=input_img, conf=conf_val, classes=sel_ids if sel_ids else [999], verbose=False)
    out_img = results[0].plot()
    
    # Verileri hazırla
    triage_list = []
    for b in results[0].boxes:
        n = model.names[int(b.cls[0])]
        i = ACILIYET_SISTEMI.get(n, {'sev': 9, 'tag': 'BİLGİ', 'clr': '#8B96A0', 'bg': '#1E2329'})
        triage_list.append({
            'name': n, 
            'conf': float(b.conf[0]), 
            'sev': i['sev'], 
            'tag': i['tag'], 
            'clr': i['clr'], 
            'bg': i['bg']
        })
    triage_list = sorted(triage_list, key=lambda x: x['sev'])
    
    # İstatistikler
    acil = len([t for t in triage_list if t['sev'] == 1])
    oncelik = len([t for t in triage_list if t['sev'] == 2])
    takip = len([t for t in triage_list if t['sev'] == 3])
    avg_conf = sum([t['conf'] for t in triage_list]) / len(triage_list) * 100 if triage_list else 0
    
    # KOMPAKT İSTATİSTİK KARTLARI (4 sütun)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stat-mini">
                <div class="stat-icon" style="background: #1F2D3D;">
                    📊
                </div>
                <div class="stat-content">
                    <div class="stat-value" style="color: #58A6FF;">{len(triage_list)}</div>
                    <div class="stat-label">Toplam Bulgu</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-mini">
                <div class="stat-icon" style="background: #2D1B1B;">
                    🚨
                </div>
                <div class="stat-content">
                    <div class="stat-value" style="color: #FF4444;">{acil}</div>
                    <div class="stat-label">Acil Durum</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-mini">
                <div class="stat-icon" style="background: #2D2419;">
                    ⚠️
                </div>
                <div class="stat-content">
                    <div class="stat-value" style="color: #FF9500;">{oncelik}</div>
                    <div class="stat-label">Öncelikli</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="stat-mini">
                <div class="stat-icon" style="background: #1F2D3D;">
                    🎯
                </div>
                <div class="stat-content">
                    <div class="stat-value" style="color: #00D9FF;">%{avg_conf:.0f}</div>
                    <div class="stat-label">Ort. Güven</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    # GÖRÜNTÜLER + GRAFİK (3 sütun)
    st.markdown("### Analiz Sonuçları")
    img1, img2, chart_col = st.columns([5, 5, 4])
    
    with img1:
        st.image(input_img, caption="📷 Orijinal Görüntü", use_container_width=True)
    
    with img2:
        st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), caption="AI Analiz", use_container_width=True)
    
    with chart_col:
        if triage_list:
            st.plotly_chart(create_compact_chart(triage_list), use_container_width=True)

    # BUTONLAR
    st.markdown("### İşlemler")
    btn1, btn2, btn3 = st.columns(3)
    t_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = f"{h_ad.replace(' ','_')}_{h_id}_{t_tag}.png"
    
    with btn1:
        if st.button("Analizi Kaydet", use_container_width=True):
            cv2.imwrite(f"kayitlar/{save_name}", out_img)
            st.success("Kaydedildi!")

    with btn2:
        if st.button("PDF Raporu Oluştur", use_container_width=True):
            cv2.imwrite("temp.png", out_img)
            stats_dict = {
                'total': len(triage_list),
                'acil': acil,
                'oncelik': oncelik,
                'avg_conf': avg_conf
            }
            pdf_path = create_report(h_ad, h_id, dr_ad, "temp.png", triage_list, stats_dict)
            st.success("PDF raporu oluşturuldu!")
            with open(pdf_path, "rb") as f:
                st.download_button("📥 PDF İndir", f, file_name=os.path.basename(pdf_path), use_container_width=True)

    with btn3:
        if st.button("Yeni Analiz", use_container_width=True):
            st.rerun()

    # DETAYLI TRİYAJ LİSTESİ
    if triage_list:
        st.markdown("### Detaylı Bulgular")
        for idx, t in enumerate(triage_list, 1):
            st.markdown(f"""
                <div class="mini-triage" style="border-left-color: {t['clr']}; background-color: {t['bg']};">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="background: {t['clr']}22; color: {t['clr']}; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.75rem;">
                            {idx}
                        </div>
                        <div>
                            <span style="color: {t['clr']}; font-weight: 600; font-size: 0.75rem;">{t['tag']}</span>
                            <span style="color: #C9D1D9; margin-left: 8px;">{t['name']}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: {t['clr']}; font-size: 1.1rem; font-weight: 700;">%{t['conf']*100:.1f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

elif not uploaded_file:
    st.info("Lütfen yukarıdan bir panoramik röntgen görüntüsü yükleyin")
else:
    st.warning("Lütfen sol panelden hasta bilgilerini doldurun")