# UAVid Remote Sensing AI — Deskripsi Proyek Lengkap

## 1. Gambaran Umum

Proyek ini adalah sistem **analisis citra penginderaan jauh (remote sensing)** berbasis AI yang menggabungkan dua teknologi utama:

1. **Semantic Segmentation** menggunakan model SegFormer-B0
2. **Retrieval-Augmented Generation (RAG)** menggunakan ChromaDB + Gemini 2.0 Flash

Dataset yang digunakan adalah **Modified UAVid Dataset** — dataset segmentasi semantik resolusi tinggi dari citra UAV (Unmanned Aerial Vehicle / drone) yang berfokus pada pemandangan jalan perkotaan.

---

## 2. Dataset: Modified UAVid

### Asal-usul
Dataset UAVid dikembangkan oleh Ye Lyu et al. (2020) dari University of Twente dan dipublikasikan dalam jurnal *ISPRS Journal of Photogrammetry and Remote Sensing*. Dataset ini mencakup:
- **300 gambar berlabel** resolusi 4K (ultra-high definition)
- **42 sequence** (seq1–seq42) dari sudut pandang oblique (miring)
- Diambil dari drone terbang rendah di lingkungan perkotaan

### Struktur Folder
```
modified_uavid_dataset/
├── train_data/
│   ├── Images/     (gambar RGB asli)
│   └── Labels/     (label segmentasi PNG berwarna)
├── val_data/
│   ├── Images/
│   └── Labels/
└── test_data/
    └── Images/     (tanpa label — hanya untuk inferensi)
```

### 8 Kelas Segmentasi
| Kelas | Warna RGB | Deskripsi |
|---|---|---|
| Background clutter | (0, 0, 0) | Semua objek yang tidak terklasifikasi |
| Building | (128, 0, 0) | Bangunan, gedung, struktur buatan |
| Road | (128, 64, 128) | Jalan, trotoar, area paved |
| Tree | (0, 128, 0) | Pohon, kanopi |
| Low vegetation | (128, 128, 0) | Rumput, semak, vegetasi rendah |
| Moving car | (64, 0, 128) | Kendaraan yang sedang bergerak |
| Static car | (192, 0, 192) | Kendaraan yang parkir/diam |
| Human | (64, 64, 0) | Pejalan kaki, manusia |

### Tantangan Utama Dataset
- **Large-scale variation** — objek terlihat sangat kecil karena ketinggian drone
- **Moving object recognition** — kendaraan bergerak sulit dibedakan dari statis
- **Temporal consistency** — konsistensi antar frame dalam satu sequence

---

## 3. Metode dan Arsitektur Sistem

### 3.1 Arsitektur Keseluruhan

```
Input (Gambar UAV)
        |
        v
[Preprocessing]
  Resize ke 512x512
  Normalize ImageNet stats
        |
        v
[SegFormer-B0 Model]
  Mix Transformer Encoder
  Lightweight MLP Decoder
        |
        v
[Logits → Upsample → ArgMax]
  Output: pred_mask (H x W, nilai 0-7)
        |
        +--> [Color Mask] → visualisasi
        +--> [Coverage Stats] → persentase tiap kelas
        +--> [Insight Text] → deskripsi natural language
                                    |
                                    v
                          [Vector Embedding]
                          all-MiniLM-L6-v2
                                    |
                                    v
                          [ChromaDB Vector Store]
                                    |
                          [RAG Query Pipeline]
                          User question → retrieve top-k → Gemini 2.0 Flash
                                    |
                                    v
                          [Streamlit UI Answer]
```

### 3.2 Model Segmentasi: SegFormer-B0

**SegFormer** adalah arsitektur transformer untuk segmentasi semantik yang diperkenalkan oleh Xie et al. (2021). Varian **B0** adalah yang paling ringan dan cocok untuk CPU.

**Kenapa SegFormer-B0?**
- Tidak memerlukan GPU — bisa jalan di CPU biasa
- Parameter paling sedikit di keluarga SegFormer (~3.7M params)
- Desain tanpa positional encoding yang kaku → generalisasi lebih baik
- Inference time ~1-3 detik per gambar di CPU

**Komponen utama:**
- **Mix Transformer (MiT) Encoder**: Hierarchical transformer yang mengekstrak fitur multi-skala
- **Lightweight MLP Decoder**: Decoder sederhana yang menggabungkan fitur dari berbagai stage encoder
- Input: gambar RGB dinormalisasi (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- Output: logits (1, 8, H/4, W/4) → di-upsample ke ukuran asli

**Catatan penting**: Model di-load dengan bobot pretrained MiT-B0 dari HuggingFace (`nvidia/mit-b0`), kemudian head diganti untuk 8 kelas UAVid. Untuk hasil terbaik, model perlu di-fine-tune dengan data UAVid. Dalam proyek ini, model digunakan untuk inference langsung (zero-shot) dan evaluasi metrik dihitung dari prediksinya.

### 3.3 Evaluasi Metrik

Tiga metrik standar segmentasi semantik digunakan:

**1. Pixel Accuracy**
```
PA = (jumlah pixel yang benar diklasifikasikan) / (total pixel)
```
Metrik paling sederhana — berapa persen pixel yang prediksinya benar.

**2. Mean Intersection over Union (mIoU)**
```
IoU_kelas_c = TP_c / (TP_c + FP_c + FN_c)
mIoU = rata-rata IoU dari semua kelas
```
Metrik utama yang digunakan di paper UAVid. Mengukur overlap antara area prediksi dan ground truth per kelas.

**3. Frequency-Weighted IoU (FW-IoU)**
```
FW-IoU = Σ (frekuensi_kelas_c × IoU_kelas_c)
```
Versi mIoU yang memberi bobot lebih ke kelas yang lebih sering muncul (seperti Background, Building, Road).

### 3.4 Pipeline RAG (Retrieval-Augmented Generation)

RAG adalah teknik yang menggabungkan pencarian informasi (retrieval) dengan generasi teks (LLM). Dalam proyek ini:

**Step 1: Document Creation**
Setiap gambar dianalisis dan diubah menjadi dokumen teks berisi:
- Nama gambar dan split
- Insight segmentasi (kelas dominan, persentase tiap kelas, tipe scene)
- Statistik detail semua kelas

**Step 2: Embedding**
Dokumen teks diubah menjadi vektor numerik (384 dimensi) menggunakan model `all-MiniLM-L6-v2` dari sentence-transformers. Model ini sangat ringan dan efisien di CPU.

**Step 3: Vector Store**
Semua vektor disimpan di **ChromaDB** — database vektor yang berjalan lokal. Menggunakan cosine similarity untuk pencarian.

**Step 4: Query Pipeline**
Saat user mengajukan pertanyaan:
1. Pertanyaan di-embed menjadi vektor
2. ChromaDB mencari top-k dokumen paling relevan (cosine similarity)
3. Dokumen-dokumen tersebut dijadikan konteks
4. Konteks + pertanyaan dikirim ke **Gemini 2.0 Flash** via API
5. Gemini menghasilkan jawaban berdasarkan konteks

---

## 4. Komponen Streamlit (5 Halaman)

### Halaman 1: Live Inference
Inti dari aplikasi — tiga cara input:
- **Upload Image**: Upload gambar dari komputer (PNG, JPG, JPEG)
- **Camera**: Ambil foto langsung dari kamera device
- **Dataset Sample**: Pilih gambar dari dataset UAVid yang sudah ada

Output yang ditampilkan:
- Gambar asli, color mask, dan overlay blend
- Bar chart coverage tiap kelas
- Radar chart coverage
- 4 metric cards (dominant class, vegetasi, road, inference time)
- AI insight text
- Jika ada ground truth label: Pixel Accuracy, mIoU, FW-IoU

### Halaman 2: Dataset Overview
- Statistik jumlah gambar per split
- Legenda kelas dengan warna dan kode RGB
- Chart EDA yang sudah digenerate (distribusi kelas, variabilitas, co-occurrence)
- Sample segmentation overlays

### Halaman 3: Evaluation Metrics
- Metric cards: Pixel Accuracy, mIoU, FW-IoU per split
- Tabel per-class IoU (train vs val)
- Bar chart perbandingan IoU
- Chart metrics_comparison.png

### Halaman 4: Batch Insights
- Tabel semua gambar yang sudah dianalisis
- Filter per split
- Distribusi kelas dominan

### Halaman 5: RAG Q&A
- Input API key Gemini (aman, type=password)
- 6 suggested queries dengan tombol langsung
- Text area untuk pertanyaan bebas
- Slider untuk jumlah dokumen yang di-retrieve
- Tampilan sumber dokumen yang ditemukan (dengan similarity score)
- Jawaban Gemini dalam dark card yang elegan

---

## 5. Struktur File Proyek

```
uavid_rag_project/
├── src/
│   ├── config.py           # Path, class map, konstanta global
│   ├── eda.py              # Full EDA: distribusi, heatmap, co-occurrence
│   ├── segformer_model.py  # SegFormer-B0: build, inference, metrics
│   ├── segmentation.py     # Batch pipeline: inference + GT comparison
│   └── rag_pipeline.py     # ChromaDB build + RAG query
├── app.py                  # Streamlit UI (5 halaman)
├── pipeline.py             # Runner utama (EDA → Seg → RAG)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 6. Cara Menjalankan

### Setup Awal
```bash
cd uavid_rag_project
pip install -r requirements.txt
cp .env.example .env
# Edit .env: GEMINI_API_KEY=AIza...
```

### Letakkan Dataset
```
uavid_rag_project/   ← folder proyek
modified_uavid_dataset/  ← sejajar dengan folder proyek
    train_data/
    val_data/
    test_data/
```

### Jalankan Pipeline Lengkap
```bash
python pipeline.py
```
Ini akan menjalankan:
1. EDA → output charts ke `outputs/`
2. SegFormer-B0 inference pada semua train/val/test images
3. Hitung Pixel Acc, mIoU, FW-IoU per split
4. Build ChromaDB vector store dari insights

### Skip Step yang Sudah Selesai
```bash
python pipeline.py --skip-eda        # skip EDA
python pipeline.py --skip-seg        # skip segmentasi
python pipeline.py --skip-rag-build  # skip build vector store
```

### Jalankan Streamlit
```bash
streamlit run app.py
```

---

## 7. Referensi

1. Lyu, Y., et al. (2020). **UAVid: A Semantic Segmentation Dataset for UAV Imagery**. ISPRS Journal of Photogrammetry and Remote Sensing.
2. Xie, E., et al. (2021). **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**. NeurIPS 2021.
3. Lewis, P., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. NeurIPS 2020.
4. Wang, L., et al. (2020). **all-MiniLM-L6-v2: Sentence Embeddings**. HuggingFace.
