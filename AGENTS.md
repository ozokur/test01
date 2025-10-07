# AGENTS.md — Project Guidance for Codex / Cursor
**Project Name:** YOLO Eğitim Yöneticisi  
**Owner:** Ozan Okur (Electromag Makine A.Ş.)  
**Version:** 1.4.2  
**Updated:** 2025-10-07  
**Language:** Türkçe 🇹🇷 / English (dual)

---

## 🧠 Genel Proje Tanımı
Bu proje, **Windows ortamında çalışan**, **PySide6 tabanlı GUI’ye sahip**, **CUDA hızlandırmalı YOLO eğitim yöneticisi**dir.  
Amaç: Kullanıcıya dataset yolunu seçtirip, veriyi analiz ederek, augmentations uygulayarak ve eğitim sürecini kaydedip devam ettirebilen bir arayüz sağlamak.

---

## 🤖 [agent:core-dev]
**Rol:** Ana yazılım geliştiricisi (Python + ML mühendisi)  
**Amaç:** CUDA tespiti, dataset analizi, eğitim döngüsü, loglama, TensorBoard entegrasyonu.  
**Kurallar:**
- Kodlama dili: Python 3.11+
- GUI: PySide6
- Eğitim: Ultralytics YOLO (v5, v8, v10, v11, v12 adapter destekli)
- GPU tespiti: `torch.cuda.is_available()`
- Dosya yapısı:  
  ```
  app/
   ├─ ui/
   ├─ core/
   ├─ configs/
   ├─ assets/
   ├─ outputs/
   └─ logs/
  ```
- Log formatı: JSON satırları (`structured logging`)
- Tüm modüller `try/except` ve `logger.exception` içermeli
- `requirements.txt` dışına ek paket yüklenmemeli
- Kod bloklarında açıklayıcı yorumlar (`# 🧠 Logic:`) bulunmalı

---

## 🪟 [agent:ui-designer]
**Rol:** Arayüz tasarımcısı (PySide6 / Qt Designer uyumlu)  
**Amaç:** Modern, minimal, test edilebilir arayüz üretmek.  
**Kurallar:**
- Ana pencere: `MainWindow`
- Bileşenler: `widgets/` altına modüler dosyalar
- Tema: Koyu arka plan, yeşil vurgu rengi (#00C853)
- Kodlama stili:  
  - `QHBoxLayout`, `QVBoxLayout` kullan
  - Dinamik genişlik (sabit piksel kullanma)
- Ek dosya: `style.qss`  
  - Scrollbar’lar, progress bar ve buton stilleri burada tutulur.

---

## ⚙️ [agent:trainer]
**Rol:** Eğitim yöneticisi  
**Amaç:** Model eğitimi, augmentations ve resume özelliğini kontrol etmek.  
**Kurallar:**
- Eğitim verisi: `ImageFolder` yapısında (`train/`, `val/`)
- Eğer ayrım yoksa, veri %80/%20 otomatik split edilmelidir
- `augment.py` yalnızca eğitim verisine uygulanır
- Eğitim metrikleri TensorBoard’da izlenebilir olmalı
- Son eğitim `outputs/runs/last_run.yaml` altında kaydedilir
- Eğitime devam (`resume=True`) parametresi desteklenmeli

---

## 🧩 [agent:version-control]
**Rol:** Versiyon yöneticisi  
**Amaç:** Kod tabanının ve konfigürasyonların sürüm takibini sağlamak.  
**Kurallar:**
- Her değişiklik `app/configs/version.json` dosyasına kaydedilir:
  ```json
  {
    "version": "1.4.2",
    "last_update": "2025-10-07",
    "changelog": [
      "Trainer resume özelliği eklendi",
      "GUI log panelinde hata izleme geliştirildi",
      "CUDA fallback bug fix"
    ]
  }
  ```
- `AGENTS.md` sürüm numarası bu dosya ile eşleşmelidir.
- Git commit mesajı formatı:
  ```
  [vX.Y.Z] <kısa açıklama> — <agent adı>
  ```
  Örnek: `[v1.4.2] dataset analyzer optimize edildi — core-dev`

---

## 🧰 [agent:utils]
**Rol:** Yardımcı modül yazarı  
**Amaç:** Projede tekrar kullanılabilir işlevler üretmek.  
**Kurallar:**
- Modül adı: `app/core/utils.py`
- Fonksiyonlar loglama, hata yakalama, GPU/CPU tespiti ve dosya işlemleriyle ilgili olmalı
- Kodlarda yorum zorunlu:
  ```python
  def check_cuda():
      """CUDA'nın aktif olup olmadığını döndürür."""
      import torch
      return torch.cuda.is_available()
  ```

---

## 🧾 [agent:docs]
**Rol:** Belgelendirme yazarı  
**Amaç:** Modüller için açıklama, kullanım ve bakım notları oluşturmak.  
**Kurallar:**
- Tüm modüller `.md` belgeleriyle belgelenmeli (`docs/` klasörü)
- Değişiklikler `CHANGELOG.md` dosyasına eklenmeli
- Sürüm geçişlerinde `docs/version_notes.md` güncellenmeli

---

## 🚫 [Genel Yasaklar]
- API anahtarlarını (`.env`) doğrudan kod içine yazma.
- Kod içinde Türkçe/İngilizce karışık değişken adı kullanma.
- Model dosyalarını repoya dahil etme (`.pt`, `.onnx` hariç).
- Hardcoded dizin (örn. `C:\Users\...`) kullanma.

---

## 🧱 [agent:build]
**Rol:** Derleme ve dağıtım yöneticisi  
**Amaç:** Projeyi tek `.exe` veya `.app` haline getirmek.  
**Kurallar:**
- Derleme aracı: `PyInstaller`
- Komut:  
  ```bash
  pyinstaller --noconfirm --onefile --name "YOLO_Trainer" app/ui/main_window.py
  ```
- Çıktı: `dist/YOLO_Trainer.exe`
- `build/` klasörü `.gitignore` içinde olmalı.

---

## 🧩 Ek Notlar
- Tüm agent’lar `core-dev` ile koordineli çalışır.
- Cursor / Codex promptları Türkçe veya İngilizce olabilir.
- `AGENTS.md` dosyası bu yapıyı referans alır ve her değişiklik commit öncesi doğrulanır.

---

**End of AGENTS.md**
