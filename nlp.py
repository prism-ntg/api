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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "dummy_key_for_testing_123")
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
    
    sql_prompt = f"""Anda adalah ahli SQL. Berikan HANYA klausa WHERE yang valid untuk MySQL berdasarkan pertanyaan user.
Jika tidak ada filter yang dibutuhkan, kembalikan teks kosong.
Gunakan format 'YYYY-MM-DD' untuk tanggal.
Jangan gunakan markdown backticks, kembalikan murni teks klausa WHERE.

Tabel dan alias yang tersedia:
- ak (aset_komplain): id_aset, nama (komplain), tanggal_perencanaan, tanggal_pengerjaan, tanggal_selesai, jenis_kerusakan, severity, penyebab, biaya_perbaikan, spare_part_digunakan, teknisi_pelaksana
- ma (master_aset): id_aset, kategori, sub_kategori, tipe, merek, lokasi_gedung, lokasi_lantai, lokasi_zona, tgl_instalasi
- rp (riwayat_penggantian_aset): id_aset_lama, tanggal_penggantian, alasan_penggantian

Nilai yang sering muncul (gunakan LIKE dengan % agar fleksibel, misal '%Mechanical%'):
- kategori: 'Mechanical', 'Security Sistem', 'Electrical', 'Arsitektur', 'Civil', 'Sistem Pemadam Kebakaran', 'Distribusi Air'
- severity: 'Ringan', 'Sedang', 'Berat', 'Fatal'

Pertanyaan user: "{pertanyaan}"
Kembalikan HANYA klausa WHERE (contoh: WHERE ma.kategori LIKE '%Mechanical%' AND ak.severity = 'Fatal').
"""

    try:
        sql_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Anda adalah asisten SQL. Jawab HANYA dengan klausa WHERE atau string kosong jika tidak perlu."},
                {"role": "user", "content": sql_prompt}
            ],
            temperature=0.0,
            max_completion_tokens=150,
        )
        where_clause = sql_completion.choices[0].message.content.strip()
        where_clause = re.sub(r'^```sql|```$', '', where_clause, flags=re.IGNORECASE).strip()
        where_clause = re.sub(r'^```|```$', '', where_clause).strip()
        if where_clause and not where_clause.upper().startswith("WHERE"):
            where_clause = f"WHERE {where_clause}"
    except Exception as e:
        print(f"[DEBUG] Error Text-to-SQL: {e}")
        where_clause = ""

    print(f"[DEBUG] LLM Generated WHERE clause: {where_clause}")

    base_query = """
        SELECT 
            ak.id_aset, ak.nama as nama_komplain, ak.tanggal_perencanaan, ak.tanggal_pengerjaan, 
            ak.tanggal_selesai, ak.jenis_kerusakan, ak.severity, ak.penyebab, 
            ak.biaya_perbaikan, ak.spare_part_digunakan, ak.teknisi_pelaksana,
            ma.kategori, ma.sub_kategori, ma.tipe, ma.merek, ma.lokasi_gedung, ma.lokasi_lantai, ma.lokasi_zona, ma.tgl_instalasi,
            rp.tanggal_penggantian, rp.alasan_penggantian
        FROM aset_komplain ak
        JOIN master_aset ma ON ak.id_aset = ma.id_aset
        LEFT JOIN riwayat_penggantian_aset rp ON ak.id_aset = rp.id_aset_lama
    """
    
    final_query_str = f"{base_query} {where_clause}"
    query = text(final_query_str)
    
    try:
        result = db.execute(query)
        data_cocok = [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f"[DEBUG] Error executing query: {e}")
        data_cocok = []

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
    
    filter_text = f"Kriteria Pencarian: {pertanyaan}"

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