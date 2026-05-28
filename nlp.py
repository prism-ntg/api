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
    
    query = text("SELECT m.*, p.tanggal_instalasi, p.target_frekuensi FROM main_data m JOIN predictions p ON m.id_aset = p.id_aset")
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
    teks_bersih = re.sub(r'\b(?:laporan|data|aset|tampilkan|berikan|buatkan|cari|carikan|tanggal|bulan|tahun|sampai|hingga|s/d|-|di|pada)\b', '', teks_bersih)
    teks_bersih = re.sub(r'\b\d{1,2}\b', '', teks_bersih) 
    teks_bersih = re.sub(r'\b20\d{2}\b', '', teks_bersih) 
    
    tokens = re.findall(r'\b\w+\b', teks_bersih)
    filtered_tokens = [kata for kata in tokens if kata not in daftar_stopword]
    
    print(f"[DEBUG 4] Sisa kata kunci pencarian (Tokens): {filtered_tokens}")
    
    data_cocok = []
    
    if date_filter:
        print("[DEBUG 5] Masuk ke proses filter berdasarkan WAKTU...")
        for aset in data:
            if 'tanggal_instalasi' in aset and aset['tanggal_instalasi']:
                try:
                    tgl_str = str(aset['tanggal_instalasi']).split(' ')[0]
                    
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
                    print(f"[DEBUG 5] Error parsing tanggal {aset['tanggal_instalasi']}: {str(e)}")
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
        headers = ["ID", "Kategori", "Sub Kat", "Tipe", "Kerusakan", "Severity", "Penyebab", "Biaya", "Spare Part", "Gedung", "Lantai", "Zona", "Tgl Instalasi", "Target Freq"]
        
        from fpdf.fonts import FontFace
        headings_style = FontFace(emphasis="BOLD", color=0, fill_color=(200, 220, 255))
        
        with pdf.table(
            borders_layout="ALL",
            align="CENTER",
            col_widths=(10, 16, 17, 22, 22, 12, 20, 15, 22, 14, 10, 10, 16, 16),
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
                row.cell(str(aset.get('sub_kategori', '-')))
                row.cell(str(aset.get('tipe', '-')))
                row.cell(str(aset.get('jenis_kerusakan', '-')))
                row.cell(str(aset.get('severity', '-')))
                row.cell(str(aset.get('penyebab', '-')))
                row.cell(str(aset.get('biaya_perbaikan', '-')))
                row.cell(str(aset.get('spare_part_digunakan', '-')))
                row.cell(str(aset.get('lokasi_gedung', '-')))
                row.cell(str(aset.get('lokasi_lantai', '-')))
                row.cell(str(aset.get('lokasi_zona', '-')))
                
                tgl = str(aset.get('tanggal_instalasi', '-'))
                if tgl != '-' and ' ' in tgl:
                    tgl = tgl.split(' ')[0]
                row.cell(tgl)
                
                row.cell(str(aset.get('target_frekuensi', '-')))
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