# YOLO-Based Multi-Class Dental Pathology Detection & Clinical Decision Support System

[English](#english) | [Türkçe](#türkçe)

---

## <a name="english"></a>🇬🇧 English

### About the Project
This repository contains the source code and implementation of my **Computer Engineering Graduation Project (Selçuk University, 2026)**. The system is an advanced **Clinical Decision Support System (CDSS)** designed to automate the analysis of panoramic dental radiographs (OPG) and minimize human-related diagnostic errors using **YOLOv11**.

### Key Features & Tech Stack
- **AI Architecture:** YOLOv11-nano fine-tuned on a robust dataset of 13,814 original images (expanded to 41,442 via data augmentation).
- **Detected Classes (10 Core Categories):** Caries, Implant, Periapical Lesion, Filling, Crown, Mandibular Canal, Missing Teeth, Retained Root, Primary Teeth, and Impacted Tooth.
- **Interactive UI:** A fully functional web application built with **Streamlit** (`app.py`), allowing clinicians to upload OPG images, filter findings, and adjust confidence thresholds dynamically.
- **Clinical Triage & Reporting:** Automated severity-based prioritization (triage) and one-click **PDF Report Generation** using `FPDF` and `Plotly` analytics.
- **Training Infrastructure:** Optimized and trained using Google Colab Pro on NVIDIA A100 GPUs.

---

## <a name="türkçe"></a>🇹🇷 Türkçe

### Proje Hakkında
Bu repo, **Selçuk Üniversitesi Bilgisayar Mühendisliği Bölümü Lisans Bitirme Projesi** olarak geliştirilen, panoramik diş röntgenlerinde (OPG) çok sınıflı patoloji ve yapı tespiti yapan yapay zeka destekli bir **Klinik Karar Destek Sistemi (CDSS)** projesidir.

### Temel Özellikler ve Teknoloji Yığını
- **Yapay Zeka Mimarisi:** 13.814 orijinal (veri artırma ile 41.442'ye çıkarılan) görüntü üzerinde eğitilen **YOLOv11** nesne tespiti modeli.
- **Hedeflenen Sınıflar:** Çürük, İmplant, Periapikal Lezyon, Dolgu, Kuron, Mandibular Kanal, Eksik Diş, Kök Kalıntısı, Süt Dişi ve Gömülü Diş.
- **İnteraktif Arayüz:** Hekimlerin röntgen yükleyip eşik değerlerini ayarlayabileceği **Streamlit** tabanlı web arayüzü (`app.py`).
- **Triyaj ve Raporlama:** Bulguların tıbbi aciliyetine göre sınıflandırılması ve `FPDF` kütüphanesiyle otomatik kurumsal **PDF Raporu** üretebilme altyapısı.
- **Donanım ve Eğitim:** Google Colab Pro ve NVIDIA A100 GPU altyapısı kullanılarak optimize edilmiştir.
