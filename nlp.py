import json
import os
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

chroma_client = chromadb.PersistentClient(path="./vektor_db_aset")
collection = chroma_client.get_or_create_collection(name="laporan_aset")

bulan_map = {
    'januari': '01', 'februari': '02', 'maret': '03', 'april': '04', 
    'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08', 
    'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
}

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Laporan Data Aset PT Nusa Tekno Global', border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', border=0, align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)

def process_nlp_report(pertanyaan: str, db: Session, baseurl: str):
    print(f"\n[INFO] Memulai proses analitik untuk kueri: '{pertanyaan}'")
    
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
        val_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            # model=llama-3.3-70b-versatile,
            messages=[
                {"role": "system", "content": validasi_prompt},
                {"role": "user", "content": pertanyaan}
            ],
            temperature=0.0,
            max_completion_tokens=15,
        )
        val_result = val_completion.choices[0].message.content.strip().upper()
        print(f"[INFO] Hasil klasifikasi intent: {val_result}")
        
        if "INVALID" in val_result:
            return {
                "status": "rejected",
                "message": "Maaf, pertanyaan Anda kurang jelas atau di luar konteks.",
                "ai_response": "Saya hanya dapat membantu seputar data aset dan komplain. Silakan perjelas permintaan Anda.",
                "matched_data_count": 0,
                "pdf_url": None,
                "used_ids": []
            }
            
        intent = "KOMPLAIN_HISTORI" if "KOMPLAIN" in val_result or "HISTORI" in val_result else "ASET_MASTER"
        
    except Exception:
        intent = "ASET_MASTER"
        pass
    
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
    
    if ENVIRONMENT == "development":
        sql_script_sync += " LIMIT 200"
        
    query_sync = text(sql_script_sync)
    result_sync = db.execute(query_sync)
    data_sync_raw = [dict(row._mapping) for row in result_sync.fetchall()]
    
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
        
        if row.get('nama_komplain'):
            grouped_assets[asset_id]['komplain'].add(f"{row.get('nama_komplain')} (Rusak: {row.get('jenis_kerusakan')}, Sev: {row.get('severity')})")
            
        if row.get('tanggal_penggantian'):
            grouped_assets[asset_id]['penggantian'].add(f"{row.get('tanggal_penggantian')} (Alasan: {row.get('alasan_penggantian')})")

    data_sync = list(grouped_assets.values())
    total_data = len(data_sync)
    
    if total_data == 0:
        return {
            "status": "success",
            "message": "Tidak ada data di database.",
            "ai_response": "Tidak ada data untuk dianalisis.",
            "matched_data_count": 0,
            "pdf_url": None,
            "used_ids": []
        }

    jumlah_vektor_sekarang = collection.count()

    if jumlah_vektor_sekarang < total_data:
        print(f"[INFO] Terdeteksi perubahan data. Memproses {total_data - jumlah_vektor_sekarang} dokumen unik untuk Vector DB...")
        batch_size = 250
        documents = []
        metadatas = []
        ids = []
        
        data_baru = data_sync[jumlah_vektor_sekarang:]
        total_baru = len(data_baru)
        
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

    print("[INFO] Menjalankan pencarian semantik (Top 15)...")
    hasil_search = collection.query(
        query_texts=[pertanyaan],
        n_results=15
    )
    
    data_terbatas = []
    used_ids = []
    if hasil_search and 'metadatas' in hasil_search and hasil_search['metadatas']:
        matched_ids = [meta['id_aset'] for meta in hasil_search['metadatas'][0]]
        if matched_ids:
            matched_ids = list(dict.fromkeys(matched_ids))
            id_list_str = ", ".join([f"'{id}'" for id in matched_ids])
            
            print("[INFO] Mengekstraksi data aset dan riwayat komplain secara lengkap.")
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

    print("[INFO] Menghasilkan ringkasan dan rekomendasi menggunakan LLM...")
    prompt_sistem = (
        "Anda adalah AI Advisor di PT Nusa Tekno Global. Tugas Anda mengevaluasi data dan memfilter mana yang paling relevan.\n"
        "Keluarkan output WAJIB dalam format JSON yang berisi 2 key:\n"
        "{\n"
        "  \"relevant_ids\": [daftar ID aset (integer) yang BENAR-BENAR SESUAI dengan permintaan user],\n"
        "  \"kesimpulan\": \"Saran singkat 1-2 paragraf berdasarkan temuan.\"\n"
        "}\n"
        "PENTING: Jika user meminta merek, kategori, atau klasifikasi tertentu, filter secara ketat pada key 'relevant_ids'! Jangan masukkan ID yang tidak sesuai."
    )
    prompt_user = f"Permintaan user: {pertanyaan}\nData Aset: {json.dumps(data_terbatas, default=str)}"

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        # model=llama-3.3-70b-versatile,
        messages=[
            {"role": "system", "content": prompt_sistem},
            {"role": "user", "content": prompt_user}
        ],
        temperature=0.0,
        max_completion_tokens=700,
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
            
            # Remove duplicates for ASET_MASTER intent based on id_aset
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
            # Jika AI merasa tidak ada yang relevan sama sekali
            data_terbatas = []
            used_ids = []
    except Exception as e:
        print(f"[ERROR] Gagal parsing JSON LLM: {e}")
        ai_kesimpulan = full_response_raw

    pdf = PDF(orientation="L")
    pdf.add_page()
    
    filter_text = f"Hasil Pencarian RAG Semantik untuk: {pertanyaan}"

    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 8, filter_text, border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    
    if data_terbatas:
        pdf.set_font("Helvetica", '', 7)
        
        if intent == "ASET_MASTER":
            headers = ["ID", "Klasifikasi", "Kategori", "Sub-Kategori", "Tipe", "Merek", "Gedung", "Lantai", "Zona", "Instalasi"]
            col_widths = (12, 22, 35, 35, 35, 30, 35, 15, 25, 30)
        else:
            headers = ["ID", "Klas", "Kategori", "Merek", "Tgl Kerja", "Kerusakan", "Sev", "Biaya", "Teknisi", "Gedung", "Lantai", "Tgl Ganti", "Alasan Ganti"]
            col_widths = (10, 15, 18, 20, 18, 24, 10, 16, 20, 16, 12, 16, 23)
            
        from fpdf.fonts import FontFace
        headings_style = FontFace(emphasis="BOLD", color=0, fill_color=(200, 220, 255))
        
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
            for aset in data_terbatas:
                row = table.row()
                current_id = str(aset.get('id_aset', '-'))
                
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
                else:
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
        pdf.set_font("Helvetica", 'I', 11)
        pdf.cell(0, 10, 'Tidak ada data aset yang sesuai dengan pencarian tersebut.', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(10)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, 'Saran & Kesimpulan:', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", '', 11)
    cleaned_text = ai_kesimpulan.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, cleaned_text)
    
    os.makedirs("reports", exist_ok=True)
    report_filename = f"Laporan_Aset_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    report_path = os.path.join("reports", report_filename)
    pdf.output(report_path)
    
    print(f"[INFO] Daftar ID Aset yang digunakan: {used_ids}")
    print("[INFO] Pembuatan laporan telah selesai.")
    
    return {
        "status": "success",
        "message": "Laporan berhasil di-generate.",
        "ai_response": ai_kesimpulan,
        "matched_data_count": len(data_terbatas),
        "pdf_url": baseurl + f"/report/{report_filename}",
        "asset_id": used_ids
    }