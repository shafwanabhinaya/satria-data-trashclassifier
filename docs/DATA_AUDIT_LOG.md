# Log Audit Data Train — Klasifikasi Sampah

Catatan proses & alasan di balik setiap keputusan, bukan cuma daftar
perintah. Ditulis supaya bisa dipelajari ulang, bukan cuma dijalankan ulang.

## Latar belakang masalah

Data train (26.527 gambar, 3 kelas: `0_Recyclable` 9.999 / `1_Electronic`
3.961 / `2_Organic` 12.567) diduga punya tiga jenis masalah yang **butuh
strategi berbeda-beda**, bukan satu proses "cleaning" yang sama:

1. **Salah label** (mis. mobil dilabel organic) — ini genuinely error, layak dibuang/diperbaiki.
2. **Ambigu** (mis. kemasan plastik bergambar sayur/buah) — ini BUKAN error,
   cuma sulit dibedakan. Kalau test set juga ambigu, membuang sampel ini dari
   train malah bikin model tidak pernah belajar kasusnya dan F1 turun.
3. **Kualitas gambar jelek** (blur, gelap, oversaturated) — hanya layak
   dibuang kalau benar-benar rusak/corrupt, bukan sekadar "kelihatan jelek",
   karena kalau test set juga punya kondisi serupa, membersihkannya dari
   train menciptakan train/test mismatch.

Prinsip yang dipegang selama kerja: **setiap kandidat cleaning diperlakukan
sebagai klaim yang perlu dibuktikan** (visual atau lewat metrik), bukan
diasumsikan benar. "Kelihatan aneh" != "salah".

## Tools yang dipilih & alasannya

- **cleanlab** — untuk deteksi salah-label. Butuh out-of-fold (OOF)
  predicted probabilities, yang mana pipeline ini SUDAH menghasilkannya
  lewat k-fold CV (`SingleModelTrainer.run()`) — jadi tinggal disambung,
  tidak perlu infrastruktur baru.
- **cleanvision** (satu ekosistem dengan cleanlab) — untuk kualitas gambar.
  Deteksi blur/gelap/terlalu-terang/duplikat/rendah-informasi tanpa perlu
  model sama sekali, cuma baca file.
- **Alasan tidak pakai skor custom (mis. varians Laplacian manual):**
  cleanvision sudah menyediakan ini dengan lebih lengkap (termasuk deteksi
  duplikat via hash), jadi tidak perlu reinvent.

## Script & modul yang dibuat

Logic asli (audit label, audit kualitas, dedupe) sekarang tinggal di
`src/trash_classifier/preprocessing/` (lihat entri 2026-07-12 di bawah untuk
alasan reorganisasinya); `scripts/*.py` cuma CLI tipis yang memanggilnya.

| File | Fungsi | Status |
|---|---|---|
| `src/trash_classifier/preprocessing/label_issues.py` | `audit_label_issues()` — cleanlab, butuh training 1 backbone untuk OOF probs. | Audit label **belum** dijalankan penuh (baru smoke test kecil, 24 gambar, buat verifikasi kode saja — lihat "Belum dikerjakan"). |
| `src/trash_classifier/preprocessing/quality.py` | `audit_image_quality()` — cleanvision, scan kualitas gambar. | **Sudah dijalankan penuh** di seluruh dataset. |
| `src/trash_classifier/preprocessing/duplicates.py` | `find_exact_duplicate_groups()` / `build_duplicate_report()` / `apply_quarantine()` — deteksi duplikat piksel-identik, grup label-bentrok tidak pernah di-auto-handle. | **Sudah dijalankan dry-run penuh.** `--apply` belum dijalankan — sengaja diserahkan ke user. |
| `scripts/audit_data.py` | CLI untuk `label_issues.py` + `quality.py`. `--skip-labels`/`--skip-quality` untuk jalankan sebagian. | - |
| `scripts/dedupe_exact_duplicates.py` | CLI untuk `duplicates.py`. Dry run default, `--apply` untuk eksekusi. | - |

Semuanya tidak mengubah `config.py`/`trainer.py`/`pipeline.py` inti sama
sekali — cuma memanggil kode yang sudah ada untuk dapat OOF probs / scan
gambar, supaya alur training dan F1 asli tidak berubah.

## Temuan konkret

### 1. Kualitas gambar (`outputs/audit/image_quality.csv`, 26.527 gambar)

| # issue | jumlah |
|---|---|
| 0 | 21.113 (79,6%) |
| 1 | 4.534 (17,1%) |
| 2 | 841 (3,2%) |
| 3 | 39 (0,15%) |

Per jenis: `odd_size` 4.577, `near_duplicates` 570, `low_information` 430,
`odd_aspect_ratio` 273, `dark` 257, `exact_duplicates` 124, `blurry` 61,
`light` 39, `grayscale` 2.

**Temuan paling penting — pelajaran soal jangan buru-buru filter:**
`odd_size` kena **3.948 dari 3.961 gambar `1_Electronic` (99,7%)**. Setelah
ditelusuri, ini BUKAN sinyal kualitas jelek — cleanvision menandai "odd size"
relatif terhadap ukuran median seluruh dataset, dan kelas Electronic
sepertinya bersumber dari tempat berbeda (foto produk/marketplace dengan
dimensi konsisten yang beda dari 2 kelas lain). Kalau kolom ini dipakai untuk
filter, hampir seluruh kelas minoritas (yang sudah paling sedikit datanya)
akan terhapus — bukan membersihkan data, tapi menghancurkan F1.
**Keputusan: `odd_size` diabaikan total untuk filtering.**

Kolom lain (`near_duplicates`, `low_information`, `dark`, `blurry`, `light`)
jumlahnya cukup kecil (~1.300 dari 26.527, ~5%) untuk ditinjau manual satu
per satu — **belum ditinjau**, lihat "Belum dikerjakan".

### 2. Duplikat exact (`outputs/audit/exact_duplicates_review.csv`, dry run)

62 grup, 124 gambar. Diverifikasi ulang **independen dari cleanvision**
(hash MD5 piksel manual, dua cara: mode asli & dipaksa RGB) — 0 dari 62 grup
salah deteksi, semua benar-benar identik byte-per-byte.

- **61 grup "aman"** (duplikat dalam label yang sama) → 61 gambar kandidat
  dibuang, tidak ada informasi hilang karena kembarannya tetap ada.
- **1 grup bentrok label**: `train\0_Recyclable\R_7520.jpg` vs
  `train\2_Organic\O_1466.jpg` — gambar identik (tas jute "SAY NO TO
  PLASTIC"), tapi satu dilabel Recyclable, satu Organic. Dicek visual: ini
  jelas barang recyclable (tas belanja reusable), jadi **`O_1466.jpg` di
  Organic itu salah label**. Karena kembarannya (`R_7520.jpg`) sudah ada di
  label yang benar, solusinya bukan "relabel", cukup buang `O_1466.jpg` —
  tidak ada informasi yang hilang.
- **Pelajaran:** grup bentrok label sengaja TIDAK di-auto-handle walau
  `--apply` dipakai, karena "duplikat dengan label beda" itu sinyal mislabel,
  bukan sinyal redundansi — dua masalah beda, jangan dicampur jadi satu
  keputusan otomatis.

## 2026-07-12 — Insiden: nama file di `data/train/` "berubah semua" di PC remote

Setelah `python scripts/dedupe_exact_duplicates.py --base-dir data --apply`
dijalankan di PC remote, user melaporkan nama file gambar di SEMUA folder
`data/train/<label>/` berubah -- bukan cuma 61 kandidat `action=drop` yang
seharusnya dipindah ke `data/_duplicates_removed/`.

**Ini tidak cocok dengan logic `apply_quarantine`**: fungsi itu cuma iterasi
baris `action=="drop" & ~cross_label_conflict` (harusnya 61 baris, sesuai
hasil lokal: 62 grup/61 aman/1 konflik) dan `shutil.move` ke folder terpisah
(`data/_duplicates_removed/`), bukan rename di tempat. Dugaan (belum
terverifikasi): kemungkinan katalog cache (`outputs/dataset_catalog.duckdb`)
yang ikut ter-copy dari mesin lain berisi path bergaya Windows (backslash),
yang tidak valid di Linux/remote, sehingga lookup file force-rebuild atau
gagal secara tidak terduga -- tapi ini cuma hipotesis, **root cause belum
terkonfirmasi** karena user memilih download ulang dataset dari awal di
remote daripada mengumpulkan bukti (log lengkap, contoh nama file sebelum
vs sesudah) untuk didiagnosis.

**Status:** dianggap selesai lewat mitigasi (redownload), bukan lewat
perbaikan akar masalah. **Risiko:** kalau insiden ini terjadi lagi, langkah
pertama adalah kumpulkan bukti dulu (log run lengkap + nama file konkret
sebelum/sesudah) sebelum re-run apapun, supaya root cause bisa ditemukan.

**Perbaikan preventif yang ditambahkan** (biar insiden serupa langsung
ketahuan dari log, bukan baru ketahuan lewat File Explorer): `apply_quarantine`
sekarang diikuti `log_apply_verification()` di `preprocessing/duplicates.py`
-- scan ulang `data/train/` dari disk (bukan cuma andalkan hitungan `moved`
di memori) dan bandingkan jumlah file per kelas sebelum vs sesudah, lalu
cocokkan dengan jumlah yang SEHARUSNYA berkurang (dihitung dari CSV). Kalau
selisihnya tidak sesuai ekspektasi, langsung muncul `WARNING` tegas di log,
bukan diam-diam. Diverifikasi dua kasus: 0 duplikat (7 -> 7, tidak crash,
tidak ada bug lama yang kambuh) dan ada duplikat (7 -> 6 tiap kelas, cocok
persis, tidak ada flag mismatch).

## 2026-07-12 — Hasil audit label penuh (cleanlab, vit_b, 3 epoch x 3 fold)

Dijalankan di PC remote (`python scripts/audit_data.py --base-dir data --model vit_b --epochs 3 --folds 3 --skip-quality`),
hasil `outputs/audit/label_issues.csv` ditarik balik ke lokal.

**Angka:** 150/26.527 (0,57%) diflag `is_flagged_label_issue`. Per `given_label`:
`2_Organic` 92, `0_Recyclable` 47, `1_Electronic` 11. Cross-tab prediksi:
mayoritas (87) adalah `given=2_Organic -> predicted=0_Recyclable`.

**Verifikasi visual (7 sampel dicek manual sebelum percaya angka ini)** --
ternyata isinya CAMPURAN 3 masalah berbeda, bukan satu:

1. **Mislabel asli** -- `O_1773.jpg` (given Organic) ternyata foto keran
   wastafel, jelas Recyclable.
2. **TEMUAN BARU -- kontaminasi non-trash:** `O_506.jpg` dan `O_5023.jpg`
   (given Organic) ternyata GRAFIK/CHART dari paper ilmiah (soal "hydrogen
   yield" / "ethanol concentration" dari fermentasi potato/papaya/cucumber).
   Ini BUKAN kasus salah label -- ini gambar yang tidak seharusnya ada di
   dataset sama sekali, kemungkinan ke-scrape karena paper-nya membahas bahan
   organik yang sama. Kategori masalah baru, belum ada di daftar sebelumnya
   (blur/duplikat/mislabel/ambigu) -- **belum ditelusuri berapa banyak yang
   lain seperti ini** di antara 150 yang diflag.
3. **Model salah, bukan datanya** -- `O_10093.jpg` (makanan berjamur, jelas
   Organic), `R_7146.jpg` (kotak plastik), `O_476.jpg` (tablet Tums),
   `O_1870.jpg` (sabun batangan) -- given label sudah benar, model yang salah
   tebak (dugaan: bias dari karakteristik sumber foto kelas Electronic yang
   beda, senada dengan temuan `odd_size` sebelumnya).

**Kesimpulan:** `label_quality_score` rendah TIDAK otomatis berarti mislabel.
150 gambar ini **wajib ditinjau manual satu-satu** -- tidak bisa di-auto-apply
seperti dedupe exact-duplicate. **Belum ditinjau semuanya**, baru 7 sampel.

## 2026-07-12 — Strategi tinjau manual + alat bantu multi-reviewer

Karena hasil audit ini akan dilaporkan ke panitia, disepakati metodologi
formal dulu sebelum eksekusi (bukan cuma "lihat & putuskan" ad-hoc):

- **Rubrik 4 kategori tetap** (disimpan sebagai `RUBRIC` di
  `preprocessing/review.py`, satu sumber kebenaran dipakai UI juga):
  **A** salah label (relabel/buang), **B** kontaminasi non-trash (buang),
  **C** ambigu tapi sah (biarkan -- ini variasi asli, bukan error), **D**
  model error (biarkan, given label sudah benar).
- **Tinjau seluruh 150, bukan sampel** -- jumlahnya cukup kecil untuk 100%
  coverage, klaim laporan jadi lebih kuat.
- **Hindari anchoring bias:** reviewer harus memutuskan dari gambar +
  `given_label` DULU, `predicted_label` model baru ditampilkan SETELAH
  submit (sebagai cross-check, bukan petunjuk).
- Kontaminasi non-trash (chart/paper ilmiah) kemungkinan tidak semua
  ke-flag cleanlab -- rencana ke depan: sampling acak terstratifikasi di
  luar yang di-flag, laporkan sebagai estimasi tingkat kontaminasi (bukan
  angka pasti), bukan tinjau manual 26.527 gambar (tidak realistis).

**Alat bantu dibangun:** `scripts/review_labels.py` -- web app lokal
(Flask) multi-reviewer, bisa dipakai rame-rame lewat ngrok tunnel.

- Logic antrian/klaim/simpan (`ReviewQueue` di `preprocessing/review.py`)
  dipisah dari Flask -- reusable & testable tanpa jalankan web server.
- **Thread-safe:** satu `threading.Lock` melindungi klaim gambar + tulis
  CSV, supaya dua reviewer tidak pernah kebagian gambar yang sama.
  Diverifikasi: 8 thread rebutan 50 kandidat sekaligus -> 0 tabrakan, semua
  tertinjau tepat sekali.
- **Passcode gate** (Flask session) -- WAJIB diisi manual (`--passcode`)
  kalau mau tunnel ke ngrok, karena URL ngrok bisa diakses siapa saja yang
  tahu link-nya. Ngrok sendiri TIDAK dijalankan otomatis oleh AI -- itu
  tindakan yang membuka akses ke internet, sengaja diserahkan ke user.
  Cara pakai: `ngrok http 5050` di terminal terpisah setelah app jalan.
- **Path traversal dicegah** di endpoint `/image` -- cuma boleh serve file
  di dalam `train_dir`, diverifikasi lewat request eksplisit ke path di
  luar itu (hasil: `403`).
- **Verifikasi end-to-end** (login -> klaim -> lihat gambar -> submit)
  lewat Python `urllib`+`http.cookiejar` (curl gagal karena masalah
  escaping backslash Windows di shell test, bukan bug aplikasi). Satu baris
  data uji sempat masuk `label_review.csv` (`TesterC`), sudah dibersihkan
  balik ke kosong sebelum dipakai sungguhan.

## 2026-07-12 — Tim selesai tinjau 150 kandidat, ketemu bug duplikat submission

Setelah tim menjalankan review lewat ngrok, `label_review.csv` berisi
**162 baris untuk 150 kandidat** -- 12 baris duplikat (filepath sama muncul
2-3x, salah satunya identik persis sampai ke detik yang sama).

**Root cause:** tombol A/B/C/D di frontend baru mengunci diri (`awaitingNext
= true`) SETELAH request selesai (di akhir `submitDecision`), bukan sebelum
dikirim. Double-klik atau keyboard-repeat cepat membuat 2 request lolos
sebelum yang pertama selesai, dan backend (`ReviewQueue.submit_decision`)
tidak menolak submission kedua untuk filepath yang sama -- selalu menulis
baris baru. Ini murni bug penulisan tool, BUKAN indikasi reviewer beneran
menilai ulang dua kali dengan hasil beda.

**Verifikasi sebelum dianggap aman:** dicek seluruh grup duplikat -- **0
grup yang decision-nya bentrok** (semua duplikat sepakat pada keputusan yang
sama), dan **150/150 kandidat unik tertinjau, tidak ada yang kosong**. Jadi
substansi hasil review-nya valid, cuma representasinya redundan.

**Perbaikan:**
- Backend: `submit_decision` sekarang idempoten -- kalau filepath sudah ada
  di `_reviewed`, langsung kembalikan record yang sudah ada, tidak menulis
  baris baru. Diverifikasi: submit 5x berturut-turut untuk gambar yang sama
  -> tetap cuma 1 baris di CSV.
- Frontend: `awaitingNext = true` dipindah ke AWAL `submitDecision` (sebelum
  `fetch`), bukan sesudah -- klik/keypress kedua langsung diblokir di baris
  pertama, tidak sempat mengirim request kedua sama sekali.
- `label_review.csv` yang lama di-dedupe (keep baris pertama per filepath,
  aman karena 0 konflik) -- **162 -> 150 baris**, 150/150 unique.

**Hasil bersih setelah dedupe:** A (salah label) 64, D (model error) 39,
B (kontaminasi) 32, C (ambigu) 15. Reviewer: sebagian besar (`anonim`)
masih tidak mengisi nama -- keterbatasan ini tetap berlaku untuk laporan
(lihat entri sebelumnya).

## 2026-07-12 — Eksekusi kategori B (kontaminasi) + temuan: batch data baru di 1_Electronic belum teraudit

Menjalankan `python scripts/remove_contamination.py --base-dir data --apply`
(32 gambar kategori B dari `label_review.csv`) sempat memicu kecurigaan:
log verifikasi menunjukkan `1_Electronic` sebelum-apply = **7.922**, bukan
~3.961 seperti yang diharapkan (dua kali lipat).

**Investigasi:** BUKAN korupsi/bug. `1_Electronic` sekarang berisi 3.961
file berpola nama lama (`E_N.jpg` -- ini yang sudah diaudit/direview) DITAMBAH
~3.956 file berpola nama benar-benar berbeda (`Washing_Machine_156.jpg`,
`Keyboard_106.jpg`, `broken-tv-5.jpg`, `WhatsApp Image....jpeg`, dll) yang
sepertinya sengaja ditambahkan buat mengatasi ketimpangan kelas (Electronic
paling sedikit: 3.961 vs 9.999 vs 12.567 -- nambah ~3.956 bikin hampir
seimbang). `2_Organic` juga naik sedikit (12.567 -> 12.569, +2) dengan
alasan serupa kemungkinan.

**Dampak ke aksi yang baru dijalankan:** AMAN. Ke-32 file kategori B semuanya
berasal dari batch lama (`E_N.jpg`), 0 "file tidak ditemukan" saat dipindah,
verifikasi before/after cocok persis dengan `label_review.csv`. Batch baru
tidak tersentuh sama sekali.

**BLIND SPOT PENTING:** ~3.956 gambar Electronic baru ini **belum pernah
lewat audit apapun** -- belum dicek cleanlab (label issues), belum dicek
cleanvision (kualitas gambar), belum dicek dedupe (duplikat piksel). Kalau
tujuan akhirnya training di atas data yang sudah bersih, batch ini perlu
lewat pipeline audit yang sama sebelum training final.

**Klarifikasi dari user:** bukan penambahan sengaja -- user menimpa
`data/train/` di laptopnya (yang dicurigai berbeda/rusak) dengan data train
**fresh dari Google Drive resmi panitia lomba**. Artinya kebalikan dari
dugaan awal: **3.961 file `E_N.jpg` yang selama ini jadi dasar SELURUH audit
kita (cleanlab 150 kandidat, cleanvision image_quality.csv, dedupe 62 grup)
adalah copy LAMA/TIDAK LENGKAP**, bukan data resmi lomba. Data resmi (yang
sekarang ada) ternyata punya ~7.922 gambar Electronic sebelum kontaminasi
dibuang -- ~3.956 di antaranya belum pernah dilihat pipeline audit sama
sekali.

**Dampak ke hasil audit sebelumnya:**
- `outputs/audit/label_issues.csv` (150 kandidat cleanlab) -- dihitung dari
  model yang dilatih HANYA di atas 3.961 Electronic (bukan ~7.917 yang
  sekarang jadi data resmi). Class imbalance yang lebih parah dari
  seharusnya ini kemungkinan berkontribusi ke pola "model ngotot nebak
  Electronic" yang kita temukan sebelumnya (lihat entri "Hasil audit label
  penuh"). **Perlu di-rerun di atas data lengkap.**
- `outputs/audit/exact_duplicates_review.csv` (62 grup dedupe) -- sama,
  cuma nyakup 3.961 dari ~7.917 Electronic. **Perlu di-rerun.**
- `outputs/audit/image_quality.csv` (cleanvision) -- sama. **Perlu di-rerun.**
- Keputusan kategori B (32 gambar) yang sudah dieksekusi **tetap valid**,
  karena semuanya berasal dari 3.961 file lama yang memang bagian dari data
  resmi juga (nama filenya masih ada di data resmi, cuma bukan satu-satunya).

**Rencana:** rebuild katalog dari nol, lalu ulang seluruh audit (cleanvision
quality + cleanlab label + dedupe) di atas `data/train/` yang sekarang
(resmi, lengkap), sebelum lanjut ke kategori A atau training final.

## 2026-07-12 — Kategori B dieksekusi ulang di PC remote (bukan laptop)

User sadar eksekusi kategori B tadi sempat dijalankan di laptop, padahal
seharusnya di PC remote (mesin training yang sesungguhnya). Diulang di
remote: `python scripts/remove_contamination.py --base-dir data --apply`.

**Hasil:** sukses, semua angka cocok tanpa `WARNING` mismatch --
`0_Recyclable` 9.990->9.979 (-11), `1_Electronic` 3.931->3.926 (-5),
`2_Organic` 12.544->12.528 (-16), total -32 sesuai `label_review.csv`.

**Temuan penting:** angka SEBELUM apply di remote (9.990 / 3.931 / 12.544)
itu **cocok dengan baseline lama** (26.527 gambar dikurangi 61 dedupe exact-
duplicate yang sudah dijalankan sebelumnya) -- BUKAN angka "data fresh dari
gdrive" yang bikin `1_Electronic` di laptop naik jadi ~7.917. Artinya:
**remote TIDAK punya masalah yang sama dengan laptop** -- data di remote
konsisten dengan seluruh audit trail kita sejauh ini (150 kandidat cleanlab,
62 grup dedupe, image_quality.csv semuanya dihitung dari dataset yang
persis sama dengan yang ada di remote sekarang).

**RESOLVED (dikonfirmasi user):** data **remote (3.961 Electronic) yang
benar/official**. Data fresh-gdrive di laptop (~7.917 Electronic) itu yang
tercemar/salah unduh -- bukan data resmi lomba. Implikasi:
- Seluruh audit trail kita (150 kandidat cleanlab, 62 grup dedupe,
  `image_quality.csv`) **tetap valid**, TIDAK perlu diulang.
- Laptop TIDAK dipakai sebagai sumber kebenaran lagi -- kalau perlu data
  lokal di laptop, sebaiknya sync dari remote, bukan unduh ulang dari gdrive
  (sumbernya yang meragukan, entah link salah atau memang beda dataset).
- Aman lanjut ke kategori A tanpa perlu redo audit apapun.

## 2026-07-12 — Alat bantu putaran kedua: tentukan label tujuan kategori A

Untuk kategori A (64 gambar salah label), tool `review_labels.py` cuma
mencatat "salah" -- perlu putaran kedua buat isi kelas tujuan yang benar.

**Refactor dulu supaya tidak duplikat kode:** Flask app (login, `/image`,
`/api/next`, `/api/skip`, `/api/decision`, HTML/JS) diekstrak dari
`scripts/review_labels.py` ke factory generik
`preprocessing/review_app.py::build_review_app()`, diparameterisasi lewat
`decision_options` (kode -> deskripsi -- sebelumnya hardcode rubrik A/B/C/D).
`ReviewQueue` juga diubah terima `decision_options` custom (default tetap
`RUBRIC`). `scripts/review_labels.py` sekarang cuma argparse + panggil
factory, sama kayak `scripts/relabel_confirmed.py` (skrip baru) yang pakai
factory sama dengan rubrik beda (3 nama kelas, bukan A/B/C/D).

**Fitur tambahan di factory (generik, berlaku ke kedua putaran):**
- Keyboard shortcut sekarang berbasis URUTAN tombol (tekan `1`/`2`/`3`...),
  bukan huruf tetap -- supaya cocok untuk rubrik apapun.
- Tombol yang kodenya sama persis dengan `given_label` otomatis dinonaktifkan
  (mencegah reviewer "pilih benar balik ke label yang sedang ditinjau/sudah
  dikonfirmasi salah").
- `predicted_label` tetap disembunyikan sampai setelah submit (prinsip
  anti-anchoring-bias yang sama dipertahankan di putaran kedua).

**Verifikasi:** `scripts/relabel_confirmed.py` dites end-to-end (login ->
klaim -> render HTML 3 tombol kelas -> submit pilihan BEDA dari
`given_label`) lewat kandidat asli (64 gambar kategori A dari
`label_review.csv` lokal) tapi output diarahkan ke folder scratch terpisah
supaya tidak mencemari `outputs/audit/` yang asli. Hasil: kolom `decision`
di `relabel_review.csv` berisi nama kelas tujuan (mis. `0_Recyclable`),
bukan huruf -- sesuai desain. `outputs/audit/label_review.csv` asli
dipastikan tidak berubah (masih 150 baris) setelah tes.

## 2026-07-12 — Tim selesai putaran kedua (64/64), ketemu 2 kasus "salah kategorisasi ulang"

`relabel_review.csv` terisi penuh (64/64, reviewer: shafwan). 2 baris punya
`note` yang menandakan kesalahan lebih dalam:

- **`O_359.jpg`** (note asli: "hapus") -- dicek visual: ini **diagram/skema
  teknis**, bukan foto sampah sama sekali. Sama persis pola kontaminasi yang
  ditemukan sebelumnya (`O_506.jpg`, `O_5023.jpg`). Putaran pertama salah
  menandainya A (salah label) padahal seharusnya B (kontaminasi).
- **`O_6109.jpg`** (note asli: "1", ambigu) -- dicek visual: foto adegan
  pemulung di tumpukan sampah campuran (TPA), bukan objek tunggal yang bisa
  dikategorikan jelas ke satu dari 3 kelas.

**Keputusan user:** kedua-duanya diperlakukan sebagai kontaminasi (buang),
bukan direlabel. Dieksekusi di PC remote (bukan laptop -- user eksplisit
minta ini, dan sistem sempat menahan percobaan saya jalankan `--apply` di
laptop untuk "verifikasi" -- itu keliru, dry-run saja sudah cukup untuk
verifikasi, tidak perlu apply beneran di mesin yang salah).

**Gap tool yang ketahuan:** `relabel_confirmed.py` cuma punya 3 tombol
kelas, tidak ada opsi "ternyata kontaminasi" -- reviewer terpaksa akal-akalan
pilih kelas asal + tulis catatan manual buat sinyal ini.

**Perbaikan:**
1. `relabel_confirmed.py`: tambah opsi ke-4 `CONTAMINATION` di
   `CLASS_OPTIONS` -- putaran berikutnya tidak perlu akal-akalan note lagi.
2. `preprocessing/contamination.py`: `apply_contamination_removal` sekarang
   return subset yang BENAR-BENAR dipindah (bukan cuma count) -- supaya
   verifikasi before/after akurat walau dijalankan ulang sebagian (file yang
   sudah pernah dipindah dilewati tanpa dihitung sebagai "seharusnya
   berkurang", mencegah false-alarm mismatch).
3. `scripts/remove_contamination.py`: tambah `--extra-review-csv` +
   `--extra-decision-value` -- bisa gabung kontaminasi dari `label_review.csv`
   (B, putaran 1) dan `relabel_review.csv` (CONTAMINATION, putaran 2)
   sekaligus dalam satu eksekusi.
4. `outputs/audit/relabel_review.csv` LOKAL dikoreksi manual: `O_359.jpg` dan
   `O_6109.jpg` decision diubah dari kelas asal ke `CONTAMINATION`, note
   diperjelas. **Koreksi yang sama perlu direplikasi di file remote** sebelum
   eksekusi (lihat instruksi di respons chat).

**Verifikasi:** dry-run `remove_contamination.py --extra-review-csv
relabel_review.csv` di lokal -- 32 (B) + 2 (CONTAMINATION) = 34 total,
breakdown per kelas cocok (`2_Organic` naik dari 16 ke 18, sesuai 2 gambar
baru yang keduanya given_label Organic).

**Dieksekusi di remote (`--apply`), sukses sempurna:** 32 gambar lama
otomatis dilewati ("sudah tidak ada") tanpa dihitung dobel -- persis sesuai
desain idempoten. 2 gambar baru (`O_359.jpg`, `O_6109.jpg`) berhasil
dipindah ke `data\_contamination_removed`. Verifikasi before/after cocok
persis: `2_Organic` 12.528->12.526 (-2), `0_Recyclable` & `1_Electronic`
tidak berubah -- sesuai karena keduanya `given_label=2_Organic`. Katalog
cache (`outputs/dataset_catalog.duckdb`) di remote ternyata memang sudah
tidak ada (belum di-rebuild sejak dihapus sebelumnya) -- aman, tidak ada
yang perlu dibersihkan.

## 2026-07-12 — Skrip eksekusi relabel (62 gambar valid)

`preprocessing/relabel.py` + `scripts/execute_relabel.py` -- baca
`relabel_review.csv` (`decision != CONTAMINATION`), pindah file BENAR-BENAR
ke folder kelas yang benar (bukan karantina seperti dedupe/kontaminasi --
ini beneran keluar dari folder lama, masuk ke folder baru). Nama file tidak
diubah; `relabel_review.csv` sendiri (given_label=lokasi lama,
decision=lokasi baru) jadi jejak audit, tidak perlu mekanisme undo terpisah.

**Beda dari dedupe/kontaminasi:** perubahan dua arah (kelas asal berkurang,
kelas tujuan bertambah), jadi verifikasi bandingkan **net** (masuk dikurangi
keluar) per kelas, bukan cuma pengurangan.

**Penanganan tabrakan nama file:** kalau folder tujuan sudah punya file
dengan nama sama persis, dilewati dengan warning (tidak menimpa) -- perlu
ditangani manual kalau terjadi.

**Verifikasi:**
- Dry-run di `relabel_review.csv` asli (read-only, aman): 62 gambar,
  breakdown `0_Recyclable->2_Organic` 9, `2_Organic->0_Recyclable` 51,
  `2_Organic->1_Electronic` 2 (total 62, cocok).
- `--apply` dites di dataset sintetis terisolasi (scratchpad, BUKAN data
  asli) dengan 4 kasus: pindah normal (2x, sukses), file sudah tidak ada
  (dilewati benar), **tabrakan nama di folder tujuan** (dilewati benar,
  tidak menimpa). Matematika verifikasi net cocok persis di semua kelas
  termasuk kasus tabrakan.

**Belum dijalankan di remote** -- itu langkah eksekusi terakhir yang tersisa
sebelum training final di atas data yang sudah bersih.

## 2026-07-12 — Dijalankan di remote: SIKLUS AUDIT & PEMBERSIHAN DATA SELESAI

`python scripts/execute_relabel.py --base-dir data --apply` sukses penuh:
62/62 gambar dipindah (0 dilewati). Verifikasi net cocok persis di semua
kelas: `0_Recyclable` +42 (masuk 51 - keluar 9), `1_Electronic` +2 (masuk 2
- keluar 0), `2_Organic` -44 (masuk 9 - keluar 53). Total dataset tidak
berubah (murni redistribusi antar kelas), tidak ada `WARNING` mismatch.

**Ringkasan seluruh siklus (dari awal sampai sini):**
1. Dedupe exact-duplicate: 61 gambar duplikat dibuang.
2. Kontaminasi non-trash (2 putaran, cleanlab + review manual): 34 gambar
   dibuang (32 dari putaran 1, 2 ketahuan belakangan di putaran 2).
3. Relabel: 62 gambar dipindah ke kelas yang benar.
4. Grup bentrok label (`R_7520.jpg`/`O_1466.jpg`) -- **belum ditangani**,
   masih tersisa di "Belum dikerjakan" (lihat poin 2 di bawah).

Total ~157 gambar train tersentuh (dibuang/dipindah) dari audit sistematis
(bukan tebak-tebakan), semuanya terverifikasi before/after di tiap langkah,
dan setiap keputusan tercatat alasannya (rubrik A/B/C/D + note reviewer) --
ini bahan yang siap dipakai buat laporan ke panitia.

**Yang masih tersisa sebelum training final:** grup bentrok label (item 2 di
bawah), dan review manual kategori kualitas kecil (item 3, opsional/nice-to-
have). Keduanya TIDAK memblokir training -- bisa training dulu sambil
dikerjakan paralel kalau waktu terbatas.

## 2026-07-12 — Verifikasi transfer "data clean" remote -> laptop, ketemu (dan koreksi) analisis salah

User transfer `data/train/` hasil bersih dari remote ke laptop (folder
`D:\SATRIA DATA\data\clean\train\`). Dicek satu-satu:

- `0_Recyclable` (10.021) & `2_Organic` (12.482): **benar persis**, cocok
  breakdown prefix nama file (`R_*`/`O_*` asli + hasil relabel) dengan
  matematika penuh siklus pembersihan.
- `1_Electronic`: laptop 3.926, remote 3.928 -- selisih 2.

**Analisis awal SAYA SALAH:** sempat menyimpulkan `1_Electronic` "salah
total" karena tidak ada file berpola `E_*` (yang saya kira "batch asli
benar" dari investigasi insiden gdrive jauh sebelumnya). Setelah user cek
langsung di remote: remote JUGA 0 file `E_*`, isinya sama-sama nama
"asing" (stock-photo-style) -- dan totalnya (3.928) cocok PERSIS dengan
angka final yang diharapkan. Kesimpulannya: pola nama itu memang
karakteristik asli kelas Electronic, BUKAN kontaminasi/salah unduh seperti
yang saya klaim sebelumnya. Analisis "E_* = batch asli" dari insiden
gdrive dulu ternyata keliru (kemungkinan salah baca snapshot di titik
waktu berbeda) -- **dicatat sebagai pelajaran: jangan simpulkan pola
nama file itu 'benar/salah' tanpa verifikasi langsung ke sumber yang
sedang dipakai (remote), jangan asumsi dari investigasi lama.**

**2 file yang selisih:** nama file-nya terlalu panjang untuk diupload ke
Google Drive (limitasi gdrive, bukan bug transfer) -- user sudah
menyisihkannya terpisah. Total train akhir yang dipakai: 26.429 (laptop,
tanpa 2 file itu) vs 26.431 (remote, lengkap) -- selisih 0,008%, tidak
signifikan.

**Status: data train "clean" di laptop (`D:\SATRIA DATA\data\clean\train\`)
sudah diverifikasi benar dan siap dipakai**, dengan catatan 2 file hilang
di atas.

## 2026-07-12 — Verifikasi ulang total: dedupe + kontaminasi + relabel, semua bersih

Data "clean" ternyata sudah dipindah/konsolidasi user ke
`trash_classifier/data/clean/train/` (bukan lagi di `D:\SATRIA DATA\data\clean\`)
dengan 2 file yang sempat hilang tadi sudah ditambahkan lagi (total kini
26.431, sama seperti remote). User minta cek ulang apakah duplikat & salah
label sudah benar-benar bersih -- dijalankan verifikasi sistematis, bukan
cuma percaya:

1. **Dedupe exact-duplicate, scan ulang PENUH dari nol** (bukan incremental)
   di seluruh 26.431 gambar via `scripts/dedupe_exact_duplicates.py
   --base-dir data/clean` -- hasil: **0 grup duplikat**. Bersih total.
2. **34 file kontaminasi** (32 dari `label_review.csv` decision=B + 2 dari
   `relabel_review.csv` decision=CONTAMINATION) dicek satu-satu terhadap
   isi folder saat ini -- **0 yang masih ada**.
3. **62 file relabel** dicek: filename harus ada persis di folder yang
   sama dengan kolom `decision` (bukan `given_label` lama) -- **0 hilang,
   0 di folder salah**. Semua tepat.
4. **Grup bentrok label `R_7520.jpg`/`O_1466.jpg`** -- dicek statusnya,
   **masih belum ditangani** (keduanya masih di lokasi asal masing-masing),
   konsisten dengan catatan sebelumnya, bukan regresi baru.

**Temuan baru saat verifikasi (bukan yang dicari, ketemu tidak sengaja):**
scan dedupe pertama CRASH dengan `FileNotFoundError` pada file
`1_Electronic/1.tumpukan-e-limbah-...jpg` -- ternyata **2 file di
`1_Electronic` punya path > 260 karakter** (batas MAX_PATH Windows), nama
file bahasa Indonesia yang sangat panjang (deskriptif, gaya stock-photo).
Ini kemungkinan besar 2 file yang sama dengan "2 file kepanjangan buat
gdrive" dari entri sebelumnya, yang sekarang sudah ditambahkan balik via
jalur lain tapi masih kena limit path Windows di lokasi barunya.

**Dampak kalau tidak diperbaiki:** `TrainValDataset._safe_open()` di
`datasets.py` menangkap exception apapun saat `Image.open()` gagal dan
diam-diam mengganti dengan gambar hitam kosong (`Image.new`) -- persis
pola bug "2 gambar train hilang diam-diam gara-gara MAX_PATH" yang
didokumentasikan di draft README project ini jauh sebelum project OOP ini
dibangun. Kalau dibiarkan, 2 gambar Electronic ini akan jadi sampel
"racun" (gambar hitam berlabel Electronic) tanpa ada yang sadar.

**Perbaikan:** rename kedua file ke nama pendek (`longname_fix_1.jpg`,
`longname_fix_2.jpg`) pakai prefix extended-length `\\?\` supaya bisa
diakses meski lebih dari MAX_PATH -- isi file TIDAK diubah, cuma nama.
Diverifikasi: `PIL.Image.open()` berhasil baca keduanya normal setelah
rename, dan scan ulang path panjang -> 0 tersisa. Scan dedupe diulang
penuh setelah fix -- hasilnya konsisten 0 duplikat (lihat poin 1 di atas).

**Catatan buat next steps:** kalau ke depan ada gambar baru ditambahkan
lagi ke dataset (dari sumber manapun), cek path length dulu sebelum
diproses -- ini bisa jadi masalah berulang kalau sumber datanya sering
pakai nama file deskriptif panjang.

## 2026-07-12 — Grup bentrok label ternyata sudah tidak relevan (HAMPIR salah eksekusi)

User minta tangani grup bentrok label (`R_7520.jpg`/`O_1466.jpg`, ditemukan
jauh sebelumnya sebagai duplikat piksel-identik tas jute "SAY NO TO
PLASTIC" berlabel beda). Sebelum eksekusi (rencana: buang `O_1466.jpg`),
dicek visual ulang dulu -- **untung dicek dulu**:

- `R_7520.jpg` sekarang: sampul majalah Men's Fashion
- `O_1466.jpg` sekarang: ilustrasi bawang

**BUKAN gambar yang sama lagi**, sama sekali tidak mirip dengan temuan
tas jute yang dicatat sebelumnya. Kemungkinan besar nama file itu sekarang
menunjuk ke konten berbeda dari titik waktu ditemukannya (data sudah
banyak berubah -- dedupe, kontaminasi, relabel, transfer ulang dari
remote, konsolidasi folder). **HAMPIR menghapus `O_1466.jpg` (ilustrasi
bawang, sah sebagai Organic) berdasarkan catatan lama yang sudah usang** --
dibatalkan setelah verifikasi visual.

**Resolusi:** scan dedupe PENUH yang baru saja dijalankan (entri
sebelumnya, 0 grup duplikat di 26.431 gambar) sudah otomatis membuktikan
pasangan duplikat asli itu tidak ada lagi di data saat ini. Item ini
**selesai dengan sendirinya** -- bukan lewat perbaikan, tapi karena
masalah aslinya sudah tidak ada di data yang sekarang dipakai.

**Pelajaran:** jangan pernah eksekusi tindakan berdasarkan temuan lama
tanpa verifikasi ulang terhadap data SAAT INI, terutama setelah banyak
operasi pindah/transfer/konsolidasi terjadi di antaranya. Nama file bukan
identitas konten yang stabil kalau ada kemungkinan re-assignment.

## 2026-07-13 — Skrip verifikasi menyeluruh (`verify_cleaning.py`)

User khawatir hasil pembersihan yang sudah dikerjakan "sia-sia" (hilang/
tertimpa) -- diminta skrip verifikasi yang bisa dijalankan ulang kapan saja
di remote (sumber kebenaran) untuk mengecek semuanya sekaligus, bukan
manual satu-satu seperti yang saya lakukan sejauh ini.

**Dibangun:** `preprocessing/verify.py` + `scripts/verify_cleaning.py`,
read-only (tidak mengubah apapun), 5 pengecekan:
1. **Kelengkapan katalog** -- bandingkan jumlah file mentah (`os.walk`) vs
   yang terdeteksi `ImageCatalog.from_folders` (dipakai training
   sungguhan). Ini pengecekan BARU yang belum pernah ada sebelumnya --
   nangkep kasus di mana pathlib glob diam-diam melewatkan file (kelas
   masalah yang sama dengan insiden MAX_PATH kemarin, tapi general: cek
   selisih jumlah, bukan cuma panjang path).
2. File dengan path > 259 karakter (reuse pelajaran insiden kemarin).
3. File kontaminasi yang seharusnya sudah dibuang tapi masih ada.
4. File relabel yang tidak ada di folder tujuan yang benar.
5. Scan ulang duplikat exact penuh (bisa dilewati via `--skip-dedupe-scan`
   buat cek cepat).

Exit code 0/1 -- bisa dipakai sebagai gerbang otomatis sebelum training.

**Verifikasi ketat (bukan cuma jalan di data bersih):** dites di data
sintetis TERPISAH yang sengaja disuntik 4 jenis masalah sekaligus (duplikat,
kontaminasi tersisa, file di folder salah, path kepanjangan) -- keempatnya
terdeteksi benar. Ketemu bug di tengah jalan: cek #5 (dedupe scan) **crash
total** (traceback mentah, bukan laporan graceful) kalau ada file path
kepanjangan yang belum diperbaiki -- diperbaiki jadi menangkap exception
dan melewati cek itu dengan pesan jelas, bukan mati total. Dites ulang
setelah fix: kelima cek lolos benar (4 error terdeteksi + 1 dilewati
graceful saat path masih panjang; 5/5 benar setelah path diperbaiki).

**Cara pakai di remote:**
```powershell
python scripts/verify_cleaning.py --base-dir data
```

## Belum dikerjakan (next steps)

Siklus utama (dedupe -> kontaminasi -> relabel) **sudah selesai** di remote,
lihat entri 2026-07-12 "SIKLUS AUDIT & PEMBERSIHAN DATA SELESAI" untuk
rekap lengkap. Sisanya, tidak ada yang memblokir training:

1. ~~Grup bentrok label `R_7520.jpg`/`O_1466.jpg`~~ — **selesai dengan
   sendirinya**, lihat entri 2026-07-12 "Grup bentrok label ternyata sudah
   tidak relevan". Kedua file sekarang berisi konten berbeda (bukan
   duplikat lagi), dan scan dedupe penuh terbaru (0 duplikat di 26.431
   gambar) membuktikan tidak ada pasangan bentrok label tersisa di data
   saat ini.
2. **Sampling acak buat estimasi kontaminasi di luar yang ke-flag cleanlab**
   -- baru ketemu 2 kasus chart/diagram ilmiah (`O_506.jpg`, `O_359.jpg`,
   dst), belum tahu seberapa luas di seluruh dataset. Opsional/nice-to-have.
3. **Review manual `near_duplicates`/`low_information`/`dark`/`blurry`/`light`**
   (~1.300 gambar, ~5% dataset) dari `image_quality.csv` -- belum ditinjau.
   Opsional/nice-to-have.
4. **Rapikan `scripts/dedupe_exact_duplicates.py`** supaya setipis `main.py`
   (orkestrasinya masih di `main()`, belum diekstrak ke `preprocessing/`
   seperti `audit_data.py`/`review_labels.py` sudah dirapikan). Kosmetik,
   belum diminta.

## 2026-07-12 — Rapikan struktur kode audit jadi `preprocessing`

User menyadari bagian audit ini konsepnya adalah *preprocessing*, bukan
sekadar "script lepas" — jadi logic-nya dipindah supaya sejalan dengan pola
yang sudah dipakai project ini (`catalog.py`/`trainer.py`/`pipeline.py` isi
logic, `main.py` cuma CLI tipis).

**Sebelum:** `scripts/audit_data.py` dan `scripts/dedupe_exact_duplicates.py`
berisi argparse SEKALIGUS logic (fungsi `audit_label_issues`,
`audit_image_quality`, `find_exact_duplicate_groups`, dst langsung di file
script).

**Sesudah:**
```
src/trash_classifier/preprocessing/
    __init__.py         re-export semua fungsi publik
    quality.py           audit_image_quality()      (cleanvision)
    label_issues.py      audit_label_issues()        (cleanlab + OOF probs)
    duplicates.py         find_exact_duplicate_groups(), build_duplicate_report(), apply_quarantine()
scripts/audit_data.py                 -- CLI tipis, import dari preprocessing
scripts/dedupe_exact_duplicates.py    -- CLI tipis, import dari preprocessing
```

**Bug yang ketemu & diperbaiki saat verifikasi refactor:** `build_duplicate_report`
crash (`KeyError: cross_label_conflict`) kalau tidak ada duplikat sama sekali
(`groups` kosong) -- `pd.DataFrame(rows)` dari list kosong tidak punya kolom
sama sekali. Ini bug LAMA (ada sejak versi awal script), cuma belum pernah
kepicu karena dataset asli selalu punya duplikat di setiap tes sebelumnya.
Baru ketahuan saat coba di dataset mini yang kebetulan 0 duplikat. Fix:
`pd.DataFrame(rows, columns=[...])` eksplisit.

**Verifikasi tidak ada regresi:** dijalankan ulang penuh di 26.527 gambar
train, hasilnya di-diff baris-per-baris dengan output sebelum refactor —
**identik 100%** (62 grup, 61 aman, 1 bentrok label, urutan sama persis).

### Kenapa `scripts/*.py` tidak ikut dipindah ke `src/.../preprocessing/`

User bertanya kenapa `audit_data.py` (dan `dedupe_exact_duplicates.py`) tidak
sekalian dimasukkan ke `preprocessing/` kalau memang itu tempat logic-nya.

**Jawaban:** yang dipindah adalah *logic*-nya (fungsi audit/deteksi/karantina),
bukan *CLI*-nya. Ini pola yang sama dengan `main.py`: `main.py` ada di root
project, BUKAN di dalam `src/trash_classifier/`, walau isinya "tentang
training" — karena orkestrasi training sesungguhnya ada di `pipeline.py`
(`TrainingPipeline`), dan `main.py` cuma argparse + wiring config + panggil
`.run()`. Alasan kode yang mengandung `argparse`/`if __name__ == "__main__"`
sengaja tidak masuk package yang bisa di-import: supaya package tetap bisa
dipakai bersih dari mana saja (notebook, test, script lain) tanpa ikut
menyeret parsing argumen command-line. Jadi `scripts/` vs `src/.../preprocessing/`
itu pembagian CLI vs library, sama seperti `main.py` vs `pipeline.py`.

**Inkonsistensi yang ketahuan saat menjawab ini:** `main.py` sungguh tipis
(bikin config, panggil satu method `.run()`). `scripts/audit_data.py` **belum**
setipis itu — `main()`-nya masih memegang logic asli: bikin `audit_dir`,
percabangan `--skip-labels`/`--skip-quality`, tulis CSV, hitung ringkasan log.
Idealnya itu jadi satu fungsi orkestrator di `preprocessing/` juga (mirip
`TrainingPipeline`), supaya `scripts/audit_data.py` beneran sama tipisnya
dengan `main.py`. **Belum dieksekusi** — ditawarkan ke user, menunggu keputusan
apakah mau dirapikan lagi atau dibiarkan (lihat "Belum dikerjakan").

## 2026-07-12 — Ekstrak `AuditPipeline` (samain `audit_data.py` dengan `main.py`)

Tindak lanjut dari poin inkonsistensi di atas. `main.py` cuma bikin config
lalu panggil `TrainingPipeline(config).run()` -- satu baris orkestrasi.
`scripts/audit_data.py` sebelumnya belum sesederhana itu: `main()`-nya masih
memegang logic (bikin `audit_dir`, percabangan `--skip-labels`/`--skip-quality`,
tulis CSV, hitung ringkasan log).

**Perbaikan:** tambah `src/trash_classifier/preprocessing/pipeline.py` berisi
`AuditPipeline` (persis peran `TrainingPipeline`, tapi untuk audit) yang
membungkus seluruh orkestrasi itu: build katalog, bikin `audit_dir`, jalankan
`audit_label_issues`/`audit_image_quality` sesuai flag, tulis CSV, log
ringkasan. `scripts/audit_data.py` sekarang cuma: argparse -> bikin config ->
`AuditPipeline(config).run(model_name=..., skip_labels=..., skip_quality=...)`
-- sama tipisnya dengan `main.py`.

**Verifikasi tidak ada regresi:** smoke test end-to-end di dataset mini yang
sama (24 gambar, 1 epoch, 2 fold) menghasilkan angka **identik** dengan smoke
test pertama sebelum refactor ini (10/24 label issues, 9/24 quality issues).

`scripts/dedupe_exact_duplicates.py` punya pola inkonsistensi yang sama
(orkestrasi masih di `main()`) tapi belum diminta dirapikan -- lihat poin 5
di "Belum dikerjakan".

## 2026-07-13 — Memulai Siklus Pembersihan Data Baru (Fresh Dataset)

Mengulang seluruh siklus pembersihan karena dataset di-reset menggunakan data fresh/mentah (26.527 gambar: `0_Recyclable` 9.999 / `1_Electronic` 3.961 / `2_Organic` 12.567).

### Tindakan yang dilakukan:
1. **Fix Isu MAX_PATH Windows:**
   Dua berkas di folder `data/work/train/1_Electronic` memiliki nama file sangat panjang sehingga gagal dibaca oleh pustaka python standar tanpa prefix extended path (`\\?\`). Berkas tersebut diganti namanya menjadi nama pendek yang aman:
   - `1.tumpukan-e-limbah-yang-kacau-dari-laptop...jpg` -> `longname_fix_1.jpg`
   - `64.dampak-lingkungan-dari-ewaste-tangkap...jpg` -> `longname_fix_2.jpg`
   Isi berkas tidak diubah sama sekali.
2. **Deduplikasi Exact (Byte-Identical):**
   Melakukan pemindaian duplikat penuh pada `data/work`. Ditemukan **62 grup duplikat** (124 gambar).
   - **61 grup aman (label sama):** Sebanyak 61 berkas dipindahkan (karantina) ke `data/work/_duplicates_removed/` menggunakan perintah:
     `python scripts/dedupe_exact_duplicates.py --base-dir data/work --apply`
     Statistik berkurang per kelas: `0_Recyclable` -9, `1_Electronic` -30, `2_Organic` -22.
   - **1 grup bentrok label:** Ditemukan duplikat exact dengan label berbeda:
     - `data\work\train\0_Recyclable\R_799.jpg` vs `data\work\train\2_Organic\O_8873.jpg` (Tas kain jute bertuliskan "SAY NO TO PLASTIC").
     - **Resolusi:** Secara manual menghapus berkas `data\work\train\2_Organic\O_8873.jpg` karena kelas sebenarnya adalah `0_Recyclable` (non-organik, recyclable) dan sudah ada berkas kembarannya `R_799.jpg` dengan label benar.
3. **Pembersihan Cache Katalog:**
   Katalog database lama (`outputs/dataset_catalog.duckdb`) dihapus setelah relokasi berkas agar tidak mengacu ke file yang sudah dipindah.
4. **Verifikasi Awal:**
   Menjalankan skrip verifikasi `python scripts/verify_cleaning.py --base-dir data/work --skip-dedupe-scan` dan semua cek kelengkapan katalog serta panjang path lolos dengan sukses.

## Setup teknis yang perlu diingat

- `data/` dan `outputs/` di-gitignore (dataset besar + hasil generate) — kalau
  audit dijalankan di PC lain, dataset perlu ditransfer manual dulu.

## 2026-07-14 — Penyusunan & Pembersihan Akhir dataset contructed (Training Sukses)

Melakukan audit akhir dan pembersihan otomatis pada dataset `contructed` yang dirakit oleh user. Dataset ini kemudian disalin ke `data/clean_dataset` untuk training final.

### Tindakan yang dilakukan:
1. **Rename File MAX_PATH Windows:**
   Ditemukan 3 file dengan path > 259 karakter di folder `1_Electronic`. Nama file di-rename menjadi nama pendek:
   - `1.tumpukan-e-limbah-yang-kacau...jpg` -> `e_waste_long_1.jpg`
   - `12.pile-discarded-motherboards...jpg` -> `e_waste_long_2.jpg`
   - `64.dampak-lingkungan-dari-ewaste...jpg` -> `e_waste_long_3.jpg`
2. **Resolusi Konflik Lintas Kelas:**
   Ditemukan 1 duplikat exact bertentangan label: `0_Recyclable/O_1485.jpg` vs `2_Organic/O_315.jpg`. Hasil peninjauan visual membuktikan gambar tersebut berisi produk botol plastik *Fruit & Vegetable Wash*. Label yang benar adalah `0_Recyclable`.
   - **Resolusi:** Menghapus `O_315.jpg` di `2_Organic` dan mempertahankan `O_1485.jpg` di `0_Recyclable`.
3. **Deduplikasi Same-Class:**
   Mengidentifikasi 55 gambar duplikat piksel-identik sekelas. Seluruh file duplikat ("drop") dipindahkan (karantina) ke `data/Preprocessing/quarantined_duplicates`.
4. **Sinkronisasi File Kontaminasi & Relabel (Sesuai Log Audit):**
   Memperbaiki ketidaksesuaian dataset hasil rakitan tim dengan log keputusan audit putaran 1:
   - Menghapus **24 file kontaminasi** non-sampah.
   - Memindahkan **24 file salah folder** ke folder kelas yang benar (contoh: `O_5263.jpg` ke `0_Recyclable`).
   - Memulihkan **5 file relabel** yang sempat hilang dari folder constructed (mengambil dari folder raw/work asli).
5. **Verifikasi Akhir & Penghapusan Cache Katalog:**
   Menghapus `outputs/dataset_catalog.duckdb` lama. Hasil verifikasi akhir menggunakan `verify_cleaning.py --base-dir data/clean_dataset` menghasilkan status **`SEMUA CEK LOLOS`** (100% bersih, lengkap, dan aman).
6. **Training Model Ensemble:**
   Training diluncurkan menggunakan `python main.py --base-dir data/clean_dataset --epochs 10 --folds 3` di GPU RTX 3060.
   - **Hasil Training**:
     - EfficientNetV2-L OOF macro-F1: `0.9736`
     - Bobot ensemble terbaik: `vit_b` 0.45, `swin_b` 0.15, `efficientnet_v2_l` 0.40
     - **Ensemble OOF macro-F1**: **0.9853** (meningkat dari baseline lama `0.9849` sebelum pembersihan data ekstra ini dilakukan).
     - Prediksi tersimpan di `outputs/submission.csv`.

