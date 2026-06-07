import json
import re
import os
import nltk
from nltk.corpus import stopwords
from groq import Groq
from fpdf import FPDF, XPos, YPos
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

nltk.download('stopwords', quiet=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

try:
    daftar_stopword = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    daftar_stopword = set(stopwords.words('indonesian'))

bulan_map = {
    'januari': '01', 'februari': '02', 'maret': '03', 'april': '04', 
    'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08', 
    'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
}

def ekstrak_tanggal(teks: str):
    teks = teks.lower()
    
    pola_iso = r'\b(20\d{2})-(\d{2})-(\d{2})\b'
    match = re.search(pola_iso, teks)
    if match:
        return {"tipe": "tanggal", "tahun": match.group(1), "bulan": match.group(2), "hari": int(match.group(3))}

    pola_range = r'\b(\d{1,2})\b\s*(?:sampai|hingga|-|s/d)\s*\b(\d{1,2})\b(?:\s+([a-z]+))?(?:\s+(20\d{2}))?'
    match = re.search(pola_range, teks)
    if match:
        start_day, end_day = int(match.group(1)), int(match.group(2))
        bulan_num = bulan_map.get(match.group(3)) if match.group(3) else None
        tahun_str = match.group(4)
        return {"tipe": "range", "start_day": start_day, "end_day": end_day, "bulan": bulan_num, "tahun": tahun_str}

    pola_tanggal = r'(?:tanggal\s+)?\b(\d{1,2})\b\s+([a-z]+)(?:\s+(20\d{2}))?'
    match = re.search(pola_tanggal, teks)
    if match:
        hari = int(match.group(1))
        bulan_num = bulan_map.get(match.group(2))
        if bulan_num:
            return {"tipe": "tanggal", "hari": hari, "bulan": bulan_num, "tahun": match.group(3)}

    pola_bulan = r'(?:di\s+|pada\s+|bulan\s+)?([a-z]+)(?:\s+(20\d{2}))?'
    match = re.search(pola_bulan, teks)
    if match:
        bulan_num = bulan_map.get(match.group(1))
        if bulan_num:
            return {"tipe": "bulan", "bulan": bulan_num, "tahun": match.group(2)}

    pola_tahun = r'(?:tahun\s+)?(20\d{2})'
    match = re.search(pola_tahun, teks)
    if match:
        return {"tipe": "tahun", "tahun": match.group(1)}

    return None

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
    print(f"\n[DEBUG 1] Menerima input pertanyaan: '{pertanyaan}'")
    
    # Validasi Intent Pertanyaan
    print("[DEBUG] Memvalidasi intent pertanyaan...")
    validasi_prompt = (
        "Anda adalah asisten validasi. Evaluasi pertanyaan user berikut. "
        "Apakah pertanyaan ini jelas dan bermaksud untuk mencari, meminta laporan, "
        "atau bertanya tentang data aset, komplain, perbaikan, kerusakan, teknisi, atau pemeliharaan? "
        "Jika iya, jawab HANYA dengan kata 'VALID'. "
        "Jika pertanyaannya tidak jelas, di luar konteks, tidak relevan, atau hanya sekadar sapaan, jawab HANYA dengan kata 'INVALID'."
    )
    
    try:
        val_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": validasi_prompt},
                {"role": "user", "content": pertanyaan}
            ],
            temperature=0.0,
            max_completion_tokens=10,
        )
        val_result = val_completion.choices[0].message.content.strip().upper()
        if "INVALID" in val_result:
            print(f"[DEBUG] Pertanyaan ditolak. Hasil validasi: {val_result}")
            return {
                "status": "rejected",
                "message": "Maaf, pertanyaan Anda kurang jelas atau di luar konteks. Silakan ajukan pertanyaan yang spesifik terkait laporan aset, komplain, perbaikan, atau pemeliharaan.",
                "ai_response": "Saya hanya dapat membantu membuat laporan dan menjawab pertanyaan seputar data aset, komplain, dan pemeliharaan. Silakan perjelas permintaan Anda.",
                "matched_data_count": 0,
                "pdf_url": None
            }
    except Exception as e:
        print(f"[DEBUG] Error saat validasi intent: {e}")
        pass # Lanjut jika terjadi error API
    
    query = text("""
        SELECT 
            ak.id_aset, ak.nama as nama_komplain, ak.tanggal_perencanaan, ak.tanggal_pengerjaan, 
            ak.tanggal_selesai, ak.jenis_kerusakan, ak.severity, ak.penyebab, 
            ak.biaya_perbaikan, ak.spare_part_digunakan, ak.teknisi_pelaksana,
            ma.kategori, ma.sub_kategori, ma.tipe, ma.merek, ma.lokasi_gedung, ma.lokasi_lantai, ma.lokasi_zona, ma.tgl_instalasi,
            rp.tanggal_penggantian, rp.alasan_penggantian
        FROM aset_komplain ak
        JOIN master_aset ma ON ak.id_aset = ma.id_aset
        LEFT JOIN riwayat_penggantian_aset rp ON ak.id_aset = rp.id_aset_lama
    """)
    result = db.execute(query)
    data = [dict(row._mapping) for row in result.fetchall()]
    
    print(f"[DEBUG 2] Total row ditarik dari database: {len(data)}")
    if len(data) > 0:
        print(f"[DEBUG 2] Cek sample kunci dari row pertama: {list(data[0].keys())}")

    date_filter = ekstrak_tanggal(pertanyaan)
    print(f"[DEBUG 3] Hasil filter NLP Waktu: {date_filter}")
    
    teks_bersih = pertanyaan.lower()
    teks_bersih = re.sub(r'\b(20\d{2})-(\d{2})-(\d{2})\b', '', teks_bersih) 
    for bulan in bulan_map.keys():
        teks_bersih = teks_bersih.replace(bulan, '')
    teks_bersih = re.sub(r'\b(?:laporan|data|aset|komplain|tampilkan|berikan|buatkan|cari|carikan|tanggal|bulan|tahun|sampai|hingga|s/d|-|di|pada)\b', '', teks_bersih)
    teks_bersih = re.sub(r'\b\d{1,2}\b', '', teks_bersih) 
    teks_bersih = re.sub(r'\b20\d{2}\b', '', teks_bersih) 
    
    tokens = re.findall(r'\b\w+\b', teks_bersih)
    filtered_tokens = [kata for kata in tokens if kata not in daftar_stopword]
    
    print(f"[DEBUG 4] Sisa kata kunci pencarian (Tokens): {filtered_tokens}")
    
    target_date_field = "tanggal_perencanaan"
    if date_filter:
        print("[DEBUG] Menentukan target kolom tanggal berdasarkan intent...")
        kolom_prompt = (
            "Anda adalah asisten database. Berdasarkan pertanyaan user, tentukan kolom tanggal mana yang paling tepat untuk difilter.\n"
            "Pilihan kolom:\n"
            "- 'tanggal_perencanaan' (gunakan ini jika membahas komplain, masalah, perbaikan, teknisi, pemeliharaan, atau jika tidak spesifik)\n"
            "- 'tgl_instalasi' (gunakan ini jika membahas instalasi aset, master aset)\n"
            "- 'tanggal_penggantian' (gunakan ini jika secara spesifik membahas riwayat penggantian aset)\n"
            "Jawab HANYA dengan nama kolom, tanpa tanda kutip atau penjelasan tambahan."
        )
        try:
            kolom_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": kolom_prompt},
                    {"role": "user", "content": pertanyaan}
                ],
                temperature=0.0,
                max_completion_tokens=10,
            )
            ai_kolom = kolom_completion.choices[0].message.content.strip().lower()
            if ai_kolom in ["tanggal_perencanaan", "tgl_instalasi", "tanggal_penggantian"]:
                target_date_field = ai_kolom
            print(f"[DEBUG] Target kolom tanggal yang dipilih AI: {target_date_field}")
        except Exception as e:
            print(f"[DEBUG] Error saat menentukan kolom tanggal AI, menggunakan default: {e}")

    data_cocok = []
    
    if date_filter:
        print(f"[DEBUG 5] Masuk ke proses filter berdasarkan WAKTU (menggunakan kolom {target_date_field})...")
        for aset in data:
            if target_date_field in aset and aset[target_date_field]:
                try:
                    tgl_str = str(aset[target_date_field]).split(' ')[0]
                    
                    if '-' in tgl_str and len(tgl_str.split('-')[0]) == 4:
                        date_obj = datetime.strptime(tgl_str, '%Y-%m-%d')
                    elif '-' in tgl_str:
                        date_obj = datetime.strptime(tgl_str, '%d-%m-%Y')
                    else: 
                        print(f"[DEBUG 5] Format tanggal aneh dilewati: {tgl_str}")
                        continue
                    
                    cocok_waktu = True
                    
                    if date_filter["tipe"] == "range":
                        if not (date_filter["start_day"] <= date_obj.day <= date_filter["end_day"]): cocok_waktu = False
                        if date_filter["bulan"] and date_obj.strftime('%m') != date_filter["bulan"]: cocok_waktu = False
                        if date_filter["tahun"] and str(date_obj.year) != date_filter["tahun"]: cocok_waktu = False
                    elif date_filter["tipe"] == "tanggal":
                        if date_obj.day != date_filter["hari"]: cocok_waktu = False
                        if date_obj.strftime('%m') != date_filter["bulan"]: cocok_waktu = False
                        if date_filter["tahun"] and str(date_obj.year) != date_filter["tahun"]: cocok_waktu = False
                    elif date_filter["tipe"] == "bulan":
                        if date_obj.strftime('%m') != date_filter["bulan"]: cocok_waktu = False
                        if date_filter["tahun"] and str(date_obj.year) != date_filter["tahun"]: cocok_waktu = False
                    elif date_filter["tipe"] == "tahun":
                        if str(date_obj.year) != date_filter["tahun"]: cocok_waktu = False
                        
                    if cocok_waktu:
                        if not filtered_tokens:
                            data_cocok.append(aset)
                        else:
                            cocok_keyword = False
                            for keyword in filtered_tokens:
                                for val in aset.values():
                                    if keyword in str(val).lower():
                                        cocok_keyword = True
                                        break
                                if cocok_keyword:
                                    break
                            if cocok_keyword:
                                data_cocok.append(aset)
                except Exception as e:
                    print(f"[DEBUG 5] Error parsing tanggal {aset.get(target_date_field)}: {str(e)}")
                    pass
    
    elif filtered_tokens:
        print("[DEBUG 5] Masuk ke proses filter berdasarkan KATA KUNCI (Tanpa Waktu)...")
        for aset in data:
            for keyword in filtered_tokens:
                cocok = False
                for val in aset.values():
                    if keyword in str(val).lower():
                        cocok = True
                        break
                if cocok and aset not in data_cocok:
                    data_cocok.append(aset)

    print(f"[DEBUG 6] Total data aset yang COCOK: {len(data_cocok)}")

    data_terbatas = data_cocok[:15]

    prompt_sistem = (
        "Anda adalah AI Advisor di PT Nusa Tekno Global. Berikan saran dan kesimpulan singkat "
        "(maksimal 2 paragraf) berdasarkan temuan data aset yang diberikan. "
        "Fokus pada insight, tindakan pemeliharaan, atau rekomendasi perbaikan. "
        "JANGAN buat tabel, poin-poin yang terlalu panjang, atau markdown, karena output ini "
        "hanya akan dicetak sebagai teks paragraf di akhir laporan PDF."
    )
    prompt_user = f"Permintaan user: {pertanyaan}\nData Aset: {json.dumps(data_terbatas, default=str)}"

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt_sistem},
            {"role": "user", "content": prompt_user}
        ],
        temperature=0.5,
        max_completion_tokens=512,
        top_p=1,
    )

    full_response = completion.choices[0].message.content or ""

    pdf = PDF(orientation="L")
    pdf.add_page()
    
    inverse_bulan = {v: k.title() for k, v in bulan_map.items()}
    filter_text = "Menampilkan Seluruh Data"

    if date_filter:
        bulan_str = inverse_bulan.get(date_filter.get('bulan'), '') if date_filter.get('bulan') else ''
        tahun_str = date_filter.get('tahun', '') or ''
        
        if date_filter["tipe"] == "range":
            filter_text = f"Filter Waktu: {date_filter['start_day']} - {date_filter['end_day']} {bulan_str} {tahun_str}".strip()
        elif date_filter["tipe"] == "tanggal":
            hari_str = date_filter.get('hari', '')
            filter_text = f"Filter Waktu: {hari_str} {bulan_str} {tahun_str}".strip()
        elif date_filter["tipe"] == "bulan":
            filter_text = f"Filter Waktu: Bulan {bulan_str} {tahun_str}".strip()
        elif date_filter["tipe"] == "tahun":
            filter_text = f"Filter Waktu: Tahun {tahun_str}".strip()
        else:
            filter_text = "Filter Waktu: Berdasarkan Tanggal"
            
        filter_text = " ".join(filter_text.split())
            
        if filtered_tokens:
            filter_text += f" | Keyword: {', '.join(filtered_tokens)}"
    elif filtered_tokens:
        filter_text = f"Filter Keyword: {', '.join(filtered_tokens)}"

    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 8, filter_text, border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    
    if data_terbatas:
        pdf.set_font("Helvetica", '', 7)
        headers = ["ID", "Kategori", "Merek", "Tgl Kerja", "Kerusakan", "Sev", "Biaya", "Teknisi", "Gedung", "Lantai", "Tgl Ganti", "Alasan Ganti"]
        
        from fpdf.fonts import FontFace
        headings_style = FontFace(emphasis="BOLD", color=0, fill_color=(200, 220, 255))
        
        with pdf.table(
            borders_layout="ALL",
            align="CENTER",
            col_widths=(10, 18, 20, 18, 26, 12, 18, 22, 16, 12, 18, 28),
            text_align="CENTER",
            headings_style=headings_style,
            line_height=6
        ) as table:
            row = table.row()
            for header_name in headers:
                row.cell(header_name)
                
            for aset in data_terbatas:
                row = table.row()
                row.cell(str(aset.get('id_aset', '-')))
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
    else:
        pdf.set_font("Helvetica", 'I', 11)
        pdf.cell(0, 10, 'Tidak ada data aset yang sesuai dengan pencarian tersebut.', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(10)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, 'Saran & Kesimpulan:', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", '', 11)
    cleaned_text = full_response.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, cleaned_text)
    
    os.makedirs("reports", exist_ok=True)
    report_filename = f"Laporan_Aset_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    report_path = os.path.join("reports", report_filename)
    pdf.output(report_path)
    
    return {
        "status": "success",
        "message": "Laporan berhasil di-generate.",
        "ai_response": full_response,
        "matched_data_count": len(data_cocok),
        "pdf_url": baseurl + f"/report/{report_filename}"
    }