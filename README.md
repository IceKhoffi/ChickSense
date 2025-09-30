# ChickSense: Sistem Pemantauan Kesehatan & Kesejahteraan Ayam Secara Real-Time

**Proof of Concept:** [chicken-health-behavior-multimodal](https://github.com/IceKhoffi/chicken-health-behavior-multimodal)

**Prototype:** [prototype-on-huggingface](https://huggingface.co/spaces/IceKhoffi/ChickSense)

**ChickSense** Proyek ini dikembangkan dalam rangka **Final Datathon 2025**, dimana berfokus pada mengubah ide dan model awal dari babak penyisihan menjadi sebuah produk fungsional yang siap digunakan dan didemonstrasikan. 

---
## Highlights
- 2nd Place Datathon 2025
![2nd Place Datathon 2025](https://github.com/user-attachments/assets/4d8dfa8a-32f0-4496-8f7a-cd40311bc185)

---

## Deskripsi Proyek
**ChickSense** merupakan sistem yang dapat memantau konfisi ayam secara langsung dengan menganalisis **perilaku, suara, dan pola aktivitas** mereka. Tujuan proyek ini mendukung peternak dalam **mendeteksi dini tanda-tanda penyakit atau stress**, sehingga kesejahteraan ayam dapat lebih terjamin.

---

## Fitur Utama

- **Analisis Video Real-Time**
  Model YOLOv8n digunakan untuk mendeteksi dan melacak ayam melalui kamera IP atau video stream.
- **Pemantauan Perilaku**
  - *Inactivity Detection*: Ayam yang tidak bergerak dalam waktu lama akan dilabel inaktiv (indikasi sakit).
  - *Clustering Detection*: Ayam yang berkumpul pada suatu titik akan dilabel sebagai suatu cluster (indikasi stress atau masalah sosial).
- **Analisis Vokalisasi**
  Model CNN digunakan menganalisis audio dari kandang untuk mendeteksi pola suara yang berhubungan dengan stress atau penyakit.
- **Dashboard Berbasis Web**
  *Interface* untuk menampilkan *real-time*, statistik, serta hasil analisis audio.
- **Notifikasi Telegram**
  Mengirim peringatan otomatis ketika ambang batas tertentu terlampaui
- **Database & Ekspor Data**
  Semua metrik disimpan pada database lokal dan dapat diekspor ke CSV untuk analisis lebih lanjut.
  
---

## Persiapan Awal

Untuk menjalankan ChickSense secara lokal penggunaan **Conda environment** sangat disarankan untuk memastikan setiap function berjalan dengan baik. Berikut langkah-langkah menjalankan ChickSense pada komputer lokal.

### Pastikan Beberapa aplikasi ini terinstall pada perangkat:
- [Git](https://git-scm.com/downloads)
- [Anaconda/Miniconda](https://www.anaconda.com/download)

### Langkah-langkah Instalasi
Jalankan perintah berikut pada **Anaconda Prompt** ataupun pada terminal jika Conda telah terpasang pada PATH:

1. **Clone Repository**
   ```bash
   git clone https://github.com/IceKhoffi/ChickSense
   cd ChickSense
   ```
2. **Buat dan Aktifkan Conda Environment**
   ```bash
   conda create -n chicksense_env python=3.11.13
   conda activate chicksense_env
   ```
3. **Install Dependensi**
   ```bash
   pip install -r requirements.txt
   conda install anaconda::ffmpeg
   ```
4. **Mengunduh Model & Video Demo**
   ```bash
   python setup.py
   cd src
   ```
Setelah mengikuti langkah-langkah diatas. ChickSense siap dikonfigurasi dan dijalankan. file konfigurasi serta main berada pada folder `src`

### Konfigurasi (`config.py`)
File `config.py` berisi konfigurasi utama aplikasi. Dimana ada beberapa parameter penting yang dapat di atur oleh kamu sebagai user untuk menyesuaikan dengan besar ukuran kandang ataupun jumlah ayam.

- Parameter Analisis Audio
  - `AUDIO_ANALYSIS_DURATION_S`: Durasi pengambilan sample audio untuk analisis. Jendela 30 detik cukup panjang untuk menangkap berbagai pola vokal yang relevan dan kompleks tanpa membebani sistem.
  - `AUDIO_ANALYSIS_INTERVAL_S`: Interval antara dimulainya setiap analisis audio. Interval 60 detik memungkinkan sistem untuk mengelola sumber daya, mencegah overload, juga menjaga stabilitas.
    
- Parameter Pemrosesan Visual
  - `YOLO_IMG_SIZE`: Ukuran gambar 512 piksel, secara signifikan mengurangi beban komputasi dan memori tanpa mengorbankan akurasi secara drastis. Dengan ini memungkinkan model untuk mempertahankan performa.
  - `DETECTION_INTERVAL_FRAMES`: Model deteksi objek dijalankan setiap 24 *frames*. Frekuensi ini dilakukan karena ayam merupakan hewan yang gerakannya relatif lambat. Pemrosesan setiap *frame* dapat menjadi pemborosan pada sumber daya komputasi.
  - `FRAME_READER_FPS`: Memilih laju frames 15 ini memastikan agar algoritma pelacakan memiliki data yang cukup untuk mempertahankan ID objek yang konsisten.
    
- Parameter Streaming
  - `WEBSOCKET_TARGET_FPS`: digunakan untuk streaming video ke interface pengguna. Nilai yang lebih rendah diperuntukan mengurangi beban pada jaringan.
  - `WEBSOCKET_JPEG_QUALITY`: melakukan kompres pada frame video sebelum streaming. memungkinkan untuk menghasilkan file yang lebih kecil, membuat stream yang lebih lancar.
    
- Analisis Parameter Deteksi Perilaku
  - `EMA_ALPHA` (*Exponential Moving Average*): mengambil kecepatan normal suatu objek. berfungsi menghasilkan rata-rata kecepatan, agar tidak terlalu sensitif pada gerakan yang mungkin terjadi tiba-tiba pada objek yang dideteksi.
  - `ENTER_THRESH_NORM_SPEED` dan `EXIT_THRESH_NORM_SPEED`: Nilai yang diberikan pada variabel tersebut merupakan bagaimana suatu kecepatan pada object bisa dinyatakan sebagai inactive ini berfungsi untuk mengatur seberapa gerakan sih yang perlu terjadi sebelum label atau indikasi inactive diberikan pada suatu object ataupun diambil dari suatu object.
  - `MIN_DURATION_S`: Konfigurasi ini berfungsi sebagai waktu kapan suatu objek dapat diberikan label inaktif. Pemilihan nilai **7200 detik (2 jam)** berdasarkan hasil penelitian dimana ayam broiler menghabiskan 70% hingga 80% dari waktunya untuk duduk atau beristirahat. Periode ketidakaktifan yang berkelanjutan selama 2 jam adalah sebuah perilaku yang kurang normal bisa jadi karena kelumpuhan ataupun penyakit yang mencegah pergerakan, atau kematian.
    
- Parameter DBSCAN
  - `EPS_PX`: Berfungsi sebagai parameter yang mengatur seberapa jauh atau dekat dua objek agar dianggap sebagai tetangga. Dalam hal ini nilai dipengaruhi oleh tata letak kamera ataupun ketinggiannya.
  - `MIN_NEIGHBORS`: Menetapkan jumlah minimum objek yang diperlukan dalam sebuah lingkungan   `EPS_PX` sehingga membentuk sebuah klaster. pembentukan klaster cenderung mengindikasikan respons terhadap kondisi lingkungan yang tidak ideal, seperti suhu yang terlalu dingin, ataupun gejala penyakit.

- Ambang Batas Peringatan Notifikasi
  - `INACTIVE_PERCENTAGE_THRESHOLD`: Sebelumnya diketahui bahwa ayam broiler dapat tidak aktif hingga 70-80% waktunya, tetapi tidak selalu seperti itu. Pemilihan batas 15% ini menjadi pilihan karena ketika 15% dari total kawanan menujukkan tidak aktif berarti ada masalah atau abnormal pada kandang.
  - `UNHEALTHY_ALERT_THRESHOLD` and `UNHEALTHY_HISTORY_LENGTH`: Ketika sistem audio mendeteksi "unhealthy" atau "tidak sehat" secara terus-menerus ini dapat menandakan bahwa ada masalah pada kandang dan hal ini juga untuk menghindari alaram palsu yang mungkin terjadi dari waktu ke waktu.
  - `DENSITY_COUNT_THRESHOLD`: Untuk memilih jumlah unique kepadatan tentu berhubungan dengan ukuran kandang dimana ketika terjadi banyak klaster atau pengelompokan (huddling), kemungkinan dikarenakan suhu dingin ataupun penyakit dan hal ini membutuhkan pengawasan segera.

---
## Menjalankan Aplikasi
Terdapat dua cara untuk menjalankan aplikasi, tergantung apakah ingin mengaktifkan notifikasi Telegram.

### Mode : Tanpa Notifikasi Telegram
Secara standar aplikasi dapat dijalankan tanpa notifikasi otomatis.
- Jalankan aplikasi, pastikan bahwa berada pada folder `/src/` dan conda environment aktif
  ```bash
  uvicorn main:app
  ```
Dashboard dapat diakses dengan mengunjungi `http://127.0.0.1:8000`

### Mode : Dengan Notifikasi Telegram
Ikuti langkah-langkah dibawah ini untuk mengaktifkan notifikasi otomatis yang akan mengirimkan chat ke Telegram jika suatu threshold terpenuhi.

Jika belum memiliki bot Telegram, berikut merupakan panduannya:
1. Buka aplikasi Telegram dan cari @BotFather
2. Mulai dengan ketik `/newbot` untuk membuat bot baru.
3. Setelah selesai dengan langkah-langkah yang diberikan, BotFather akan memberikan **Token API**. Salin dan simpan token ini.
4. Untuk mendapatkan **ID chat** mulai percakapan dengan bot yang baru dibuat kemudian kunjungi website berikut `https://api.telegram.org/bot<TOKEN>/getUpdates` Ganti `<TOKEN>` dengan token yang sebelumnya telah didapat. **ID chat akan terlihat di dalam respons JSON**

- Selanjutnaya mengatur variabel *environment*
  ```bash
  conda env config vars set ENABLE_TELEGRAM_NOTIFICATIONS="y"
  conda env config vars set TELEGRAM_BOT_TOKEN="TELEGRAM BOT TOKEN"
  conda env config vars set TELEGRAM_CHAT_ID="TELEGRAM BOT CHAT ID"
  ```
  Untuk melihat apakah variabel env sudah terset dapat di check dengan `echo $<Nama Variable>`
  
  Aplikasi dapat dijalankan seperti sebelumnya
  ```bash
  uvicorn main:app
  ```

---
## Tampilan UI (User Interface)

![UI 1](https://github.com/user-attachments/assets/64366e3f-581d-4cb5-babb-a0a095d2a26f)

---
## Tampilan Notifikasi Telegram
![tampilan_header](https://github.com/user-attachments/assets/6918dfe9-148a-44e0-8cff-e90610408f19)

![tampilan_isi](https://github.com/user-attachments/assets/368e638d-d6ca-4726-9c50-2fdb1984f0ab)



   
