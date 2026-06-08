import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import chromadb
from dotenv import load_dotenv
from fpdf import FPDF, XPos, YPos
from groq import Groq
from sqlalchemy import text
from sqlalchemy.orm import Session

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Inisialisasi Basis Data Vektor (ChromaDB) untuk penyimpanan dan pencarian semantik
chroma_client = chromadb.PersistentClient(path="./vektor_db_aset")
collection = chroma_client.get_or_create_collection(name="laporan_aset")

def _has_value(value) -> bool:
    """Cek nilai database yang benar-benar terisi."""
    return value is not None and str(value).strip() not in ("", "-", "None", "null")

def _safe_float(value) -> float:
    """Konversi angka database yang mungkin null/string menjadi float."""
    try:
        return float(value) if _has_value(value) else 0.0
    except (TypeError, ValueError):
        return 0.0

def _top_counter(counter: Counter, limit: int = 8) -> dict:
    """Ambil distribusi teratas agar konteks ke LLM tetap ringkas."""
    return {str(key): value for key, value in counter.most_common(limit) if _has_value(key)}

def _unique_field_values(rows: list[dict], field_name: str) -> list[str]:
    """Ambil nilai unik field dari hasil query untuk deteksi filter eksplisit."""
    values = {str(row.get(field_name)).strip() for row in rows if _has_value(row.get(field_name))}
    return sorted(values, key=len, reverse=True)

def _filter_rows_by_field_value(rows: list[dict], field_name: str, expected_value: str) -> list[dict]:
    expected = expected_value.strip().lower()
    return [
        row for row in rows
        if str(row.get(field_name, "")).strip().lower() == expected
    ]

def apply_explicit_question_filters(pertanyaan: str, rows: list[dict]) -> list[dict]:
    """
    Memaksa filter yang disebut eksplisit oleh user.

    Semantic search dan LLM bagus untuk pencarian makna, tetapi filter seperti
    "merek Import" harus tetap dikunci secara deterministik agar merek lain
    tidak ikut masuk ke tabel/PDF.
    """
    question = pertanyaan.lower()
    filtered_rows = rows

    filter_rules = [
        (("merk", "merek", "brand"), "merek"),
        (("kategori",), "kategori"),
        (("sub kategori", "sub-kategori", "subkategori"), "sub_kategori"),
        (("klasifikasi", "jadwal"), "klasifikasi"),
        (("gedung", "lokasi"), "lokasi_gedung"),
        (("lantai",), "lokasi_lantai"),
        (("zona",), "lokasi_zona"),
    ]

    for cues, field_name in filter_rules:
        if not any(cue in question for cue in cues):
            continue

        for value in _unique_field_values(filtered_rows, field_name):
            if value.lower() in question:
                filtered_rows = _filter_rows_by_field_value(filtered_rows, field_name, value)
                print(f"[INFO] Filter eksplisit diterapkan: {field_name} = {value}")
                break

    return filtered_rows

def build_analysis_context(rows: list[dict]) -> dict:
    """
    Membuat ringkasan statistik deterministik dari data kandidat.

    Ringkasan ini dikirim ke LLM agar kesimpulan tidak hanya menghitung jumlah,
    tetapi bisa membaca pola kategori, lokasi, klasifikasi, komplain, severity,
    dan biaya berdasarkan data aktual.
    """
    assets = {}
    complaint_count_by_asset = defaultdict(int)
    repair_cost_by_asset = defaultdict(float)
    severity_values = []

    for row in rows:
        asset_id = str(row.get("id_aset"))
        if asset_id not in assets:
            assets[asset_id] = row

        if _has_value(row.get("id_komplain")) or _has_value(row.get("nama_komplain")) or _has_value(row.get("jenis_kerusakan")):
            complaint_count_by_asset[asset_id] += 1

        repair_cost_by_asset[asset_id] += _safe_float(row.get("biaya_perbaikan"))

        if _has_value(row.get("severity")):
            severity_values.append(_safe_float(row.get("severity")))

    unique_rows = list(assets.values())
    kategori_counter = Counter(row.get("kategori") for row in unique_rows if _has_value(row.get("kategori")))
    sub_kategori_counter = Counter(row.get("sub_kategori") for row in unique_rows if _has_value(row.get("sub_kategori")))
    merek_counter = Counter(row.get("merek") for row in unique_rows if _has_value(row.get("merek")))
    klasifikasi_counter = Counter(row.get("klasifikasi") for row in unique_rows if _has_value(row.get("klasifikasi")))
    gedung_counter = Counter(row.get("lokasi_gedung") for row in unique_rows if _has_value(row.get("lokasi_gedung")))
    kerusakan_counter = Counter(row.get("jenis_kerusakan") for row in rows if _has_value(row.get("jenis_kerusakan")))
    teknisi_counter = Counter(row.get("teknisi_pelaksana") for row in rows if _has_value(row.get("teknisi_pelaksana")))

    high_severity_count = sum(1 for value in severity_values if value >= 4)
    total_repair_cost = sum(repair_cost_by_asset.values())
    top_problem_assets = sorted(
        [
            {
                "id_aset": asset_id,
                "jumlah_komplain": complaint_count_by_asset[asset_id],
                "total_biaya_perbaikan": round(repair_cost_by_asset[asset_id], 2),
                "kategori": assets[asset_id].get("kategori"),
                "merek": assets[asset_id].get("merek"),
                "lokasi_gedung": assets[asset_id].get("lokasi_gedung"),
                "klasifikasi": assets[asset_id].get("klasifikasi"),
            }
            for asset_id in assets
            if complaint_count_by_asset[asset_id] > 0
        ],
        key=lambda item: (item["jumlah_komplain"], item["total_biaya_perbaikan"]),
        reverse=True,
    )[:5]

    return {
        "total_baris_data": len(rows),
        "total_aset_unik": len(assets),
        "total_komplain": sum(complaint_count_by_asset.values()),
        "total_biaya_perbaikan": round(total_repair_cost, 2),
        "rata_rata_severity": round(sum(severity_values) / len(severity_values), 2) if severity_values else None,
        "jumlah_severity_tinggi_4_ke_atas": high_severity_count,
        "distribusi_kategori": _top_counter(kategori_counter),
        "distribusi_sub_kategori": _top_counter(sub_kategori_counter),
        "distribusi_merek": _top_counter(merek_counter),
        "distribusi_klasifikasi": _top_counter(klasifikasi_counter),
        "distribusi_lokasi_gedung": _top_counter(gedung_counter),
        "jenis_kerusakan_teratas": _top_counter(kerusakan_counter),
        "teknisi_teratas": _top_counter(teknisi_counter),
        "aset_dengan_komplain_terbanyak": top_problem_assets,
    }

class PDF(FPDF):
    """
    Kelas turunan dari FPDF untuk melakukan kustomisasi format laporan PDF.
    
    Menyediakan implementasi khusus untuk bagian header dan footer 
    yang akan diterapkan secara otomatis pada setiap halaman dokumen.
    """
    def header(self):
        """Mendefinisikan tampilan kop (header) pada setiap halaman PDF."""
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Laporan Data Aset PT Nusa Tekno Global', border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

    def footer(self):
        """Mendefinisikan tampilan penutup (footer) pada setiap halaman PDF, termasuk nomor halaman."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', border=0, align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)

def process_nlp_report(pertanyaan: str, db: Session, baseurl: str) -> dict:
    """
    Memproses permintaan pengguna dalam bahasa alami dan menghasilkan laporan analitik.
    
    Alur kerja fungsi:
    1. Klasifikasi Intent: Menggunakan LLM untuk menentukan kategori pertanyaan.
    2. Ekstraksi Data & Vektorisasi: Mensinkronkan data termutakhir dari basis data relasional
       ke dalam basis data vektor (ChromaDB).
    3. Pencarian Semantik: Melakukan kueri pada basis data vektor untuk mencari aset yang relevan.
    4. Analisis Lanjut & Filtrasi: Menggunakan LLM untuk menyaring hasil pencarian secara ketat
       dan menyusun narasi kesimpulan (insight).
    5. Pembuatan Laporan: Menghasilkan dokumen PDF berisi tabel data terekstraksi dan kesimpulan AI.
    
    Parameter:
        pertanyaan (str): Kueri atau pertanyaan yang diajukan oleh pengguna.
        db (Session): Sesi koneksi basis data relasional (SQLAlchemy).
        baseurl (str): URL dasar aplikasi untuk penyusunan tautan unduhan PDF.
        
    Mengembalikan:
        dict: Struktur kamus data yang memuat status operasi, pesan respon, narasi AI, 
              jumlah data yang relevan, tautan laporan PDF, dan daftar identitas aset terkait.
    """
    print(f"\n[INFO] Memulai proses analitik untuk kueri: '{pertanyaan}'")
    
    # 1. KLASIFIKASI INTENT (INTENT CLASSIFICATION)
    # Prompt untuk menginstruksikan LLM bertindak sebagai pengklasifikasi
    validasi_prompt = (
        "Anda adalah asisten klasifikasi intent. Evaluasi pertanyaan user terkait manajemen aset. "
        "Pilih salah satu dari 3 kategori berikut yang paling tepat:\n"
        "1. 'ASET_MASTER' : Jika user menanyakan data fisik, spesifikasi, daftar aset, lokasi, atau merek.\n"
        "2. 'KOMPLAIN_HISTORI' : Jika user menanyakan riwayat kerusakan, komplain, perbaikan, biaya, atau teknisi.\n"
        "3. 'INVALID' : Jika pertanyaan di luar konteks manajemen aset.\n"
        "Jawab HANYA dengan nama kategori (ASET_MASTER / KOMPLAIN_HISTORI / INVALID)."
    )
    
    try:
        print("[INFO] Melakukan klasifikasi intent menggunakan LLM...")
        # Pemanggilan API LLM untuk mendapatkan klasifikasi kueri
        val_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": validasi_prompt},
                {"role": "user", "content": pertanyaan}
            ],
            temperature=0.0,
            max_completion_tokens=15,
        )
        val_result = val_completion.choices[0].message.content.strip().upper()
        print(f"[INFO] Hasil klasifikasi intent: {val_result}")
        
        # Validasi apabila kueri di luar cakupan aplikasi
        if "INVALID" in val_result:
            return {
                "status": "rejected",
                "message": "Maaf, pertanyaan Anda kurang jelas atau di luar konteks.",
                "ai_response": "Saya hanya dapat membantu seputar data aset dan komplain. Silakan perjelas permintaan Anda.",
                "matched_data_count": 0,
                "pdf_url": None,
                "used_ids": []
            }
            
        # Penentuan intent utama 
        intent = "KOMPLAIN_HISTORI" if "KOMPLAIN" in val_result or "HISTORI" in val_result else "ASET_MASTER"
        
    except Exception:
        # Penanganan galat standar apabila servis LLM mengalami gangguan
        intent = "ASET_MASTER"
        pass
    
    # 2. EKSTRAKSI DATA & SINKRONISASI BASIS DATA VEKTOR
    # Penyusunan skrip SQL guna mengagregasi data aset beserta riwayat perbaikan dan penggantian
    sql_script_sync = """
        SELECT 
            ma.id_aset, ak.nama as nama_komplain, ak.tanggal_perencanaan, ak.tanggal_pengerjaan, 
            ak.tanggal_selesai, ak.jenis_kerusakan, ak.severity, ak.penyebab, 
            ak.biaya_perbaikan, ak.spare_part_digunakan, ak.teknisi_pelaksana,
            ma.kategori, ma.sub_kategori, ma.tipe, ma.merek, ma.lokasi_gedung, ma.lokasi_lantai, ma.lokasi_zona, ma.tgl_instalasi, ma.status_jadwal as klasifikasi,
            rp.tanggal_penggantian, rp.alasan_penggantian
        FROM master_aset ma
        LEFT JOIN aset_komplain ak ON ma.id_aset = ak.id_aset
        LEFT JOIN riwayat_penggantian_aset rp ON ma.id_aset = rp.id_aset_lama
        WHERE ma.status = 'Aktif'
        ORDER BY ma.id_aset ASC
    """
    
    # Pembatasan kuantitas data jika berada pada lingkungan pengembangan (Development)
    if ENVIRONMENT == "development":
        sql_script_sync += " LIMIT 200"
        
    # Eksekusi kueri agregasi
    query_sync = text(sql_script_sync)
    result_sync = db.execute(query_sync)
    data_sync_raw = [dict(row._mapping) for row in result_sync.fetchall()]
    
    # Restrukturisasi data mentah hasil agregasi SQL menjadi struktur JSON yang terkelompok per aset
    grouped_assets = {}
    for row in data_sync_raw:
        asset_id = str(row['id_aset'])
        if asset_id not in grouped_assets:
            grouped_assets[asset_id] = {
                'id_aset': asset_id,
                'kategori': row.get('kategori', '-'),
                'sub_kategori': row.get('sub_kategori', '-'),
                'tipe': row.get('tipe', '-'),
                'merek': row.get('merek', '-'),
                'lokasi_gedung': row.get('lokasi_gedung', '-'),
                'lokasi_lantai': row.get('lokasi_lantai', '-'),
                'lokasi_zona': row.get('lokasi_zona', '-'),
                'tgl_instalasi': row.get('tgl_instalasi', '-'),
                'klasifikasi': row.get('klasifikasi', '-'),
                'komplain': set(),
                'penggantian': set()
            }
        
        # Penggabungan rekaman histori komplain menjadi satu struktur teks kolektif
        if row.get('nama_komplain'):
            grouped_assets[asset_id]['komplain'].add(f"{row.get('nama_komplain')} (Rusak: {row.get('jenis_kerusakan')}, Sev: {row.get('severity')})")
            
        # Penggabungan rekaman histori penggantian menjadi satu struktur teks kolektif
        if row.get('tanggal_penggantian'):
            grouped_assets[asset_id]['penggantian'].add(f"{row.get('tanggal_penggantian')} (Alasan: {row.get('alasan_penggantian')})")

    data_sync = list(grouped_assets.values())
    total_data = len(data_sync)
    
    # Evaluasi pencegahan proses lebih lanjut jika basis data kosong
    if total_data == 0:
        return {
            "status": "success",
            "message": "Tidak ada data di database.",
            "ai_response": "Tidak ada data untuk dianalisis.",
            "matched_data_count": 0,
            "pdf_url": None,
            "used_ids": []
        }

    # Kalkulasi delta jumlah dokumen pada Vector DB untuk inisiasi sinkronisasi mandiri
    jumlah_vektor_sekarang = collection.count()

    if jumlah_vektor_sekarang < total_data:
        print(f"[INFO] Terdeteksi perubahan data. Memproses {total_data - jumlah_vektor_sekarang} dokumen unik untuk Vector DB...")
        batch_size = 250
        documents = []
        metadatas = []
        ids = []
        
        # Ekstraksi rekaman data baru yang belum terindeks
        data_baru = data_sync[jumlah_vektor_sekarang:]
        total_baru = len(data_baru)
        
        # Proses iteratif penyusunan teks representasional (vektor) dari setiap entitas aset
        for index, aset in enumerate(data_baru):
            index_asli = jumlah_vektor_sekarang + index
            
            komplain_str = "; ".join(aset['komplain']) if aset['komplain'] else "-"
            penggantian_str = "; ".join(aset['penggantian']) if aset['penggantian'] else "-"
            
            teks_dokumen = (
                f"ID Aset: {aset.get('id_aset', '-')}. "
                f"Klasifikasi: {aset.get('klasifikasi', '-')}. "
                f"Kategori: {aset.get('kategori', '-')}, Sub Kategori: {aset.get('sub_kategori', '-')}, Tipe: {aset.get('tipe', '-')}, Merek: {aset.get('merek', '-')}. "
                f"Lokasi: Gedung {aset.get('lokasi_gedung', '-')}, Lantai {aset.get('lokasi_lantai', '-')}, Zona {aset.get('lokasi_zona', '-')}. "
                f"Tgl Instalasi: {aset.get('tgl_instalasi', '-')}. "
                f"Riwayat Komplain: {komplain_str}. "
                f"Riwayat Penggantian: {penggantian_str}."
            )
            documents.append(teks_dokumen)
            metadatas.append({"id_aset": aset.get('id_aset', str(index_asli))})
            ids.append(f"id_{index_asli}")
            
            # Injeksi batch ke dalam Vector DB guna menyeimbangkan utilisasi memori (Bulk Insert)
            if (index + 1) % batch_size == 0 or (index + 1) == total_baru:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                documents.clear()
                metadatas.clear()
                ids.clear()
        print("[INFO] Proses sinkronisasi Vector DB selesai.")

    # 3. PENCARIAN SEMANTIK (SEMANTIC SEARCH)
    print("[INFO] Menjalankan pencarian semantik (Top 500)...")
    # Melakukan pemanggilan kueri vektor untuk mendapatkan 500 dokumen teratas (Nearest Neighbors)
    hasil_search = collection.query(
        query_texts=[pertanyaan],
        n_results=500
    )
    
    data_terbatas = []
    used_ids = []
    
    # Ekstraksi ID entitas dari hasil pencarian
    if hasil_search and 'metadatas' in hasil_search and hasil_search['metadatas']:
        matched_ids = [meta['id_aset'] for meta in hasil_search['metadatas'][0]]
        if matched_ids:
            # Penghapusan duplikasi ID
            matched_ids = list(dict.fromkeys(matched_ids))
            id_list_str = ", ".join([f"'{id}'" for id in matched_ids])
            
            print("[INFO] Mengekstraksi data aset dan riwayat komplain secara lengkap.")
            # Pemanggilan ulang ke basis data relasional guna mengakuisisi rincian informasi entitas secara utuh
            query_matched = text(f"""
                SELECT 
                    ma.id_aset, ak.id as id_komplain, ak.nama as nama_komplain, ak.tanggal_perencanaan, ak.tanggal_pengerjaan, 
                    ak.tanggal_selesai, ak.jenis_kerusakan, ak.severity, ak.penyebab, 
                    ak.biaya_perbaikan, ak.spare_part_digunakan, ak.teknisi_pelaksana,
                    ma.kategori, ma.sub_kategori, ma.tipe, ma.merek, ma.lokasi_gedung, ma.lokasi_lantai, ma.lokasi_zona, ma.tgl_instalasi, ma.status_jadwal as klasifikasi,
                    rp.tanggal_penggantian, rp.alasan_penggantian
                FROM master_aset ma
                LEFT JOIN aset_komplain ak ON ma.id_aset = ak.id_aset
                LEFT JOIN riwayat_penggantian_aset rp ON ma.id_aset = rp.id_aset_lama
                WHERE ma.id_aset IN ({id_list_str})
                ORDER BY ma.id_aset ASC, ak.id ASC
            """)
                
            result_matched = db.execute(query_matched)
            data_terbatas = [dict(row._mapping) for row in result_matched.fetchall()]
            used_ids = list(dict.fromkeys([str(d['id_aset']) for d in data_terbatas]))

    # 4. ANALISIS LANJUT & FILTRASI MENGGUNAKAN LLM
    print("[INFO] Menghasilkan ringkasan dan rekomendasi menggunakan LLM...")
    data_terbatas = apply_explicit_question_filters(pertanyaan, data_terbatas)
    used_ids = list(dict.fromkeys([str(d['id_aset']) for d in data_terbatas]))
    analysis_context = build_analysis_context(data_terbatas)

    # Penyusunan prompt sistem instruksional dengan tata cara yang presisi (Output JSON format)
    prompt_sistem = (
        "Anda adalah AI Advisor di PT Nusa Tekno Global. Tugas Anda mengevaluasi data dan memfilter mana yang paling relevan.\n"
        "ATURAN FILTER SANGAT KETAT:\n"
        "1. Jika user mencari klasifikasi tertentu (misal: 'Mingguan'), maka HANYA ID dengan klasifikasi 'Mingguan' yang boleh masuk relevant_ids.\n"
        "2. JANGAN PERNAH memasukkan klasifikasi lain (seperti 'Reaktif') jika user secara spesifik meminta 'Mingguan' atau jadwal rutin lainnya.\n"
        "3. Output WAJIB dalam format JSON yang berisi 2 key:\n"
        "{\n"
        "  \"relevant_ids\": [daftar ID aset (integer) yang BENAR-BENAR SESUAI, MAKSIMAL 15 teratas],\n"
        "  \"kesimpulan\": \"Analisis berbasis data dalam 3-5 paragraf singkat.\"\n"
        "}\n"
        "ATURAN KESIMPULAN:\n"
        "- Jangan hanya mengulang jumlah data. Jelaskan pola yang terlihat dari ringkasan statistik dan data aset.\n"
        "- Wajib sebutkan angka penting yang tersedia, misalnya total aset, kategori/merek/lokasi/klasifikasi dominan, jumlah komplain, severity, biaya, atau aset prioritas.\n"
        "- Berikan interpretasi operasional: apa arti pola tersebut untuk maintenance, risiko, stok spare part, vendor, atau prioritas inspeksi.\n"
        "- Berikan rekomendasi tindakan yang konkret dan realistis.\n"
        "- Jika data komplain/severity/biaya tidak tersedia, katakan bahwa analisis risiko teknis terbatas oleh data tersebut, lalu fokus pada distribusi aset yang tersedia.\n"
        "- Jangan membuat angka, lokasi, kategori, biaya, atau ID yang tidak ada pada data.\n"
        "PENTING: Kejujuran data adalah prioritas utama. Lebih baik memberikan sedikit ID yang benar daripada mencampurnya dengan data yang tidak sesuai (halusinasi)."
    )
    
    # Pruning data untuk LLM agar tidak melebihi limit token
    data_untuk_llm = []
    for d in data_terbatas[:150]:
        data_untuk_llm.append({
            "id_aset": d.get("id_aset"),
            "nama": d.get("nama_komplain"),
            "kategori": d.get("kategori"),
            "sub_kategori": d.get("sub_kategori"),
            "tipe": d.get("tipe"),
            "merek": d.get("merek"),
            "klasifikasi": d.get("klasifikasi"),
            "lokasi_gedung": d.get("lokasi_gedung"),
            "lokasi_lantai": d.get("lokasi_lantai"),
            "lokasi_zona": d.get("lokasi_zona"),
            "jenis_kerusakan": d.get("jenis_kerusakan"),
            "severity": d.get("severity"),
            "biaya_perbaikan": d.get("biaya_perbaikan"),
            "teknisi_pelaksana": d.get("teknisi_pelaksana"),
            "tanggal_pengerjaan": d.get("tanggal_pengerjaan"),
            "tanggal_penggantian": d.get("tanggal_penggantian"),
            "alasan_penggantian": d.get("alasan_penggantian")
        })
    
    prompt_user = (
        f"Permintaan user: {pertanyaan}\n"
        f"Ringkasan Statistik Kandidat: {json.dumps(analysis_context, default=str)}\n"
        f"Sample Data Aset Kandidat Maksimal 150 Baris: {json.dumps(data_untuk_llm, default=str)}"
    )

    # Pelaksanaan proses penalaran (Reasoning) oleh model AI untuk menghasilkan simpulan dan penyaringan sekunder
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": prompt_sistem},
            {"role": "user", "content": prompt_user}
        ],
        temperature=0.0, 
        max_completion_tokens=1300,
        response_format={"type": "json_object"}, 
        top_p=1,
    )

    full_response_raw = completion.choices[0].message.content or "{}"
    
    ai_kesimpulan = "Tidak ada kesimpulan yang dapat ditarik."
    try:
        parsed_response = json.loads(full_response_raw)
        relevant_ids = parsed_response.get("relevant_ids", [])
        ai_kesimpulan = parsed_response.get("kesimpulan", full_response_raw)
        
        if relevant_ids:
            str_relevant_ids = [str(x) for x in relevant_ids]
            data_terbatas = [d for d in data_terbatas if str(d['id_aset']) in str_relevant_ids]
            data_terbatas = apply_explicit_question_filters(pertanyaan, data_terbatas)

            # Normalisasi duplikasi entri jika intent klasifikasinya adalah ASET_MASTER
            if intent == "ASET_MASTER":
                unique_data = []
                seen_ids = set()
                for d in data_terbatas:
                    if str(d['id_aset']) not in seen_ids:
                        unique_data.append(d)
                        seen_ids.add(str(d['id_aset']))
                data_terbatas = unique_data
                
            used_ids = list(dict.fromkeys([str(d['id_aset']) for d in data_terbatas]))
        else:
            # Penyesuaian ke set kosong jika tidak ditemukan korelasi data menurut penilaian AI
            data_terbatas = []
            used_ids = []
    except Exception as e:
        print(f"[ERROR] Gagal parsing JSON LLM: {e}")
        ai_kesimpulan = full_response_raw

    # 5. PEMBUATAN LAPORAN PDF (REPORT GENERATION)
    # Inisialisasi dokumen PDF dalam orientasi lanskap (Landscape)
    pdf = PDF(orientation="L")
    pdf.add_page()
    
    filter_text = f"Hasil Pencarian RAG Semantik untuk: {pertanyaan}"

    # Pencetakan sub-judul
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 8, filter_text, border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    
    # Pembuatan tabel matriks data apabila terdapat entitas yang memadai
    if data_terbatas:
        pdf.set_font("Helvetica", '', 7)
        
        # Penyesuaian komposisi kolom mengikuti klasifikasi intent pada tahapan 1
        if intent == "ASET_MASTER":
            headers = ["ID", "Klasifikasi", "Kategori", "Sub-Kategori", "Tipe", "Merek", "Gedung", "Lantai", "Zona", "Instalasi"]
            col_widths = (12, 22, 35, 35, 35, 30, 35, 15, 25, 30)
        else:
            headers = ["ID", "Klas", "Kategori", "Merek", "Tgl Kerja", "Kerusakan", "Sev", "Biaya", "Teknisi", "Gedung", "Lantai", "Tgl Ganti", "Alasan Ganti"]
            col_widths = (10, 15, 18, 20, 18, 24, 10, 16, 20, 16, 12, 16, 23)
            
        from fpdf.fonts import FontFace
        headings_style = FontFace(emphasis="BOLD", color=0, fill_color=(200, 220, 255))
        
        # Render komponen struktural tabel
        with pdf.table(
            borders_layout="ALL",
            align="CENTER",
            col_widths=col_widths,
            text_align="CENTER",
            headings_style=headings_style,
            line_height=6
        ) as table:
            row = table.row()
            for header_name in headers:
                row.cell(header_name)
                
            last_id = None
            # Iterasi pencetakan sel data untuk setiap baris rekam jejak
            for aset in data_terbatas:
                row = table.row()
                current_id = str(aset.get('id_aset', '-'))
                
                # Render baris tabel khusus klasifikasi ASET_MASTER
                if intent == "ASET_MASTER":
                    row.cell(current_id)
                    row.cell(str(aset.get('klasifikasi', '-')))
                    row.cell(str(aset.get('kategori', '-')))
                    row.cell(str(aset.get('sub_kategori', '-')))
                    row.cell(str(aset.get('tipe', '-')))
                    row.cell(str(aset.get('merek', '-')))
                    gedung = str(aset.get('lokasi_gedung', '-'))
                    row.cell(gedung if gedung != 'None' else '-')
                    lantai = str(aset.get('lokasi_lantai', '-'))
                    row.cell(lantai if lantai != 'None' else '-')
                    zona = str(aset.get('lokasi_zona', '-'))
                    row.cell(zona if zona != 'None' else '-')
                    tgl_inst = str(aset.get('tgl_instalasi', '-'))
                    row.cell(tgl_inst if tgl_inst != 'None' else '-')
                
                # Render baris tabel khusus klasifikasi KOMPLAIN_HISTORI
                else:
                    # Logika penggabungan sel (merging effect) untuk data duplikat pada id_aset yang sama
                    if current_id == last_id:
                        row.cell("")
                        row.cell("")
                        row.cell("")
                        row.cell("")
                    else:
                        row.cell(current_id)
                        row.cell(str(aset.get('klasifikasi', '-')))
                        row.cell(str(aset.get('kategori', '-')))
                        row.cell(str(aset.get('merek', '-')))
                    
                    tgl_kerja = str(aset.get('tanggal_pengerjaan', '-'))
                    if tgl_kerja != '-' and tgl_kerja != 'None' and ' ' in tgl_kerja:
                        tgl_kerja = tgl_kerja.split(' ')[0]
                    row.cell(tgl_kerja if tgl_kerja != 'None' else '-')
                    
                    row.cell(str(aset.get('jenis_kerusakan', '-')))
                    row.cell(str(aset.get('severity', '-')))
                    
                    biaya = str(aset.get('biaya_perbaikan', '-'))
                    row.cell(biaya if biaya != 'None' else '-')
                    
                    teknisi = str(aset.get('teknisi_pelaksana', '-'))
                    row.cell(teknisi if teknisi != 'None' else '-')
                    
                    if current_id == last_id:
                        row.cell("")
                        row.cell("")
                        row.cell("")
                        row.cell("")
                    else:
                        gedung = str(aset.get('lokasi_gedung', '-'))
                        row.cell(gedung if gedung != 'None' else '-')
                        
                        lantai = str(aset.get('lokasi_lantai', '-'))
                        row.cell(lantai if lantai != 'None' else '-')
                        
                        tgl_ganti = str(aset.get('tanggal_penggantian', '-'))
                        if tgl_ganti != '-' and tgl_ganti != 'None' and ' ' in tgl_ganti:
                            tgl_ganti = tgl_ganti.split(' ')[0]
                        row.cell(tgl_ganti if tgl_ganti != 'None' else '-')
                        
                        alasan = str(aset.get('alasan_penggantian', '-'))
                        row.cell(alasan if alasan != 'None' else '-')
                    
                last_id = current_id
    else:
        # Penanganan kasus tanpa hasil pencarian (Null Output)
        pdf.set_font("Helvetica", 'I', 11)
        pdf.cell(0, 10, 'Tidak ada data aset yang sesuai dengan pencarian tersebut.', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(10)
    
    # Pencetakan narasi analitik (Insight Generation) pada segmen akhir PDF
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, 'Saran & Kesimpulan:', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", '', 11)
    cleaned_text = ai_kesimpulan.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, cleaned_text)
    
    # Manajemen direktori dan penyimpanan berkas dokumen lokal
    os.makedirs("reports", exist_ok=True)
    report_filename = f"Laporan_Aset_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    report_path = os.path.join("reports", report_filename)
    pdf.output(report_path)
    
    print(f"[INFO] Daftar ID Aset yang digunakan: {used_ids}")
    print("[INFO] Pembuatan laporan telah selesai.")
    
    # Mengembalikan payload data balasan (Response Payload) untuk rutinitas antarmuka pengguna
    return {
        "status": "success",
        "message": "Laporan berhasil di-generate.",
        "ai_response": ai_kesimpulan,
        "matched_data_count": len(data_terbatas),
        "pdf_url": baseurl + f"/report/{report_filename}",
        "asset_id": used_ids
    }
