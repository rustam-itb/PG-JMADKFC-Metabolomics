# Langkah unggah ke GitHub

1. Buka GitHub, login, pilih **New repository**, beri nama misalnya `PG-JMADKFC-Metabolomics`. Pilih **Public** setelah semua hak akses dan lisensi disetujui. Jangan aktifkan README tambahan karena paket sudah memilikinya.
2. Ekstrak `PG_JMADKFC_GitHub_Repository_Ready.zip` ke satu folder. Buka halaman repo baru lalu **Add file → Upload files**, unggah **isi folder**, bukan ZIP itu sendiri. Jika ribuan file melebihi batas unggah browser, gunakan Git di terminal (langkah 3).
3. Cara Git yang lebih andal dari terminal di dalam folder tersebut:

```bash
git init
git add .
git commit -m "Initial PG-JMADKFC reproducibility release"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

4. Periksa bahwa `data/*.npz`, data sumber, dan direktori `results/reproduced` tidak ikut terunggah secara tidak sengaja.
5. Pada GitHub, buka **Releases → Draft a new release**, buat tag `v1.0.0`, ringkas cakupan benchmark, dan terbitkan.
6. Salin URL yang nyata, lalu ganti placeholder pada `docs/CODE_AVAILABILITY.tex` dan pernyataan Code Availability manuskrip. Jika menggunakan Zenodo, tautkan GitHub setelah versi dan berkas publik final.

Sebelum membuat repo publik, pelajari `docs/LICENSE_DECISION.md` dan `docs/RELEASE_CHECKLIST.md`.
