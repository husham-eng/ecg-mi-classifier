"""
app.py — تطبيق الويب الرئيسي
=================================
واجهة ويب بسيطة (Flask) لتصنيف نوع احتشاء عضلة القلب من:
  - ملف إشارة رقمية (CSV/TXT، عمود واحد من القيم)
  - أو صورة مخطط ECG (JPG/PNG) مصوَّرة أو ممسوحة ضوئياً

يدعم إدخال أكثر من قطب من الأقطاب الأربعة المدعومة (Lead I, aVR, V2, V6)
في نفس الطلب؛ يُدمَج القرار النهائي تلقائياً عبر تصويت مرجّح إن توفر
أكثر من قطب.

⚠️ هذا نموذج أولي بحثي وليس جهازاً طبياً معتمداً. النتائج للمساعدة على
تحديد الأولوية فقط ويجب دائماً تأكيدها طبياً.
"""

import os
import tempfile
import numpy as np
from flask import Flask, request, render_template, jsonify

from ecg_pipeline import classify_patient, classify_from_image, SUPPORTED_LEADS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB حد أقصى لكل طلب

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SIGNAL_EXTENSIONS = {".csv", ".txt"}


def load_signal_file(file_storage) -> np.ndarray:
    """يقرأ ملف إشارة (عمود واحد من الأرقام، مع أو بدون رأس نصي)."""
    content = file_storage.read().decode("utf-8", errors="ignore")
    values = []
    for line in content.splitlines():
        line = line.strip().split(",")[0].split("\t")[0]
        try:
            values.append(float(line))
        except ValueError:
            continue
    return np.array(values)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", leads=SUPPORTED_LEADS)


@app.route("/classify", methods=["POST"])
def classify():
    signals_by_lead = {}
    image_results = {}
    errors = {}

    # قطب Lead II اختياري: لو رُفع، يُستخدم فقط كمرجع محازاة موحّد لمواقع R
    # عبر كل الأقطاب المُدخَلة (لا يُصنَّف بنفسه، النماذج الحالية لا تغطيه).
    reference_lead_signal = None
    ref_file = request.files.get("file_II")
    if ref_file and ref_file.filename != "":
        ref_ext = os.path.splitext(ref_file.filename)[1].lower()
        if ref_ext in SIGNAL_EXTENSIONS:
            ref_fs = float(request.form.get("fs_II", 500))
            ref_raw = load_signal_file(ref_file)
            if len(ref_raw) >= ref_fs:
                reference_lead_signal = (ref_raw, ref_fs)
            else:
                errors["II"] = "إشارة Lead II المرجعية قصيرة جداً — تم تجاهلها."
        else:
            errors["II"] = "Lead II المرجعي مدعوم كملف إشارة رقمية (CSV/TXT) فقط حالياً."

    for lead in SUPPORTED_LEADS:
        file = request.files.get(f"file_{lead}")
        if not file or file.filename == "":
            continue

        ext = os.path.splitext(file.filename)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            try:
                result = classify_from_image(tmp_path, lead)
                if "error" in result:
                    errors[lead] = result["error"]
                else:
                    image_results[lead] = result
            finally:
                os.unlink(tmp_path)

        elif ext in SIGNAL_EXTENSIONS:
            fs = float(request.form.get(f"fs_{lead}", 500))
            raw = load_signal_file(file)
            if len(raw) < fs:  # أقل من ثانية واحدة من البيانات
                errors[lead] = "الإشارة قصيرة جداً (أقل من ثانية واحدة)."
                continue
            signals_by_lead[lead] = (raw, fs)
        else:
            errors[lead] = f"صيغة ملف غير مدعومة: {ext}"

    if not signals_by_lead and not image_results:
        return jsonify({"error": "لم يتم رفع أي قطب صالح.", "details": errors}), 400

    # الحالة الشائعة: قطب واحد فقط كصورة — نرجع نتيجته مباشرة
    if image_results and not signals_by_lead and len(image_results) == 1:
        lead, result = next(iter(image_results.items()))
        result["errors"] = errors
        return jsonify(result)

    # خلاف ذلك: ندمج كل الأقطاب المتوفرة كإشارات (الصور تُحوَّل مسبقاً)
    combined_signals = dict(signals_by_lead)
    result = classify_patient(combined_signals, reference_lead_signal=reference_lead_signal) if combined_signals else {
        "error": "الدمج متعدد الأقطاب متاح حالياً لملفات الإشارة الرقمية فقط في هذا الإصدار."
    }
    if image_results:
        result["image_leads_processed_separately"] = image_results
    result["errors"] = errors
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
