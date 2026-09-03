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
import io
import base64
import tempfile
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify

from ecg_pipeline import (classify_patient, classify_from_image, classify_lead_signal,
                           combine_lead_probabilities, SUPPORTED_LEADS)
from ecg_pipeline.panel_detector import detect_panel_leads
from translations import get_translation, DEFAULT_LANG

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB حد أقصى لكل طلب

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
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
    lang = request.args.get("lang", DEFAULT_LANG)
    t = get_translation(lang)
    return render_template("index.html", leads=SUPPORTED_LEADS, t=t)


@app.route("/detect_panel", methods=["POST"])
def detect_panel():
    """
    يستقبل صورة لوحة ECG كاملة (12 قطباً)، يكتشف ويقصّ الأقطاب الأربعة
    المدعومة تلقائياً (LeadI, aVR, V2, V6)، ويُرجعها كصور مصغّرة (base64)
    مع مستوى ثقة لكل واحدة — لعرضها بخطوة تأكيد بصرية قبل التصنيف
    النهائي (لا يُصنَّف شيء هنا، فقط اكتشاف وتقطيع).
    """
    file = request.files.get("panel_image")
    if not file or file.filename == "":
        return jsonify({"error": "لم يتم رفع أي صورة."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return jsonify({"error": f"صيغة ملف غير مدعومة لصورة اللوحة الكاملة: {ext}"}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        detected = detect_panel_leads(tmp_path)
    except Exception as e:
        return jsonify({"error": f"تعذّر اكتشاف اللوحة — تحقق من أنها صورة تخطيط قياسية واضحة. ({e})"}), 400
    finally:
        os.unlink(tmp_path)

    # نحتاج فقط الأقطاب الأربعة المدعومة فعلياً بالتطبيق
    panel_to_project = {"I": "LeadI", "aVR": "aVR", "V2": "V2", "V6": "V6"}
    leads_out = {}
    for panel_name, project_name in panel_to_project.items():
        info = detected.get(panel_name)
        if info is None:
            continue
        crop = info["crop"]
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        leads_out[project_name] = {
            "image_b64": b64,
            "confidence": info["confidence"],  # confirmed / weak / not_found
        }

    if not leads_out:
        return jsonify({"error": "تعذّر اكتشاف أي قطب مدعوم بهذي الصورة."}), 400

    return jsonify({"leads": leads_out})


@app.route("/classify", methods=["POST"])
def classify():
    per_lead_results = {}
    errors = {}

    # قطب Lead II اختياري: لو رُفع، يُستخدم فقط كمرجع محازاة موحّد لمواقع R
    # عبر الأقطاب المُدخَلة كإشارات رقمية (لا يُصنَّف بنفسه، ولا يُطبَّق
    # حالياً على مسار الصور — كل صورة لها توقيتها الخاصة من تحليلها).
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

    external_r_locs = None
    if reference_lead_signal is not None:
        from ecg_pipeline.preprocessing import denoise, remove_baseline, detect_r_peaks
        ref_raw, ref_fs = reference_lead_signal
        ref_clean = remove_baseline(denoise(ref_raw, ref_fs), degree=6)
        external_r_locs = detect_r_peaks(ref_clean, ref_fs, polarity_robust=True)

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
            finally:
                os.unlink(tmp_path)
            if "error" in result:
                errors[lead] = result["error"]
            else:
                per_lead_results[lead] = result

        elif ext in SIGNAL_EXTENSIONS:
            fs = float(request.form.get(f"fs_{lead}", 500))
            raw = load_signal_file(file)
            if len(raw) < fs:  # أقل من ثانية واحدة من البيانات
                errors[lead] = "الإشارة قصيرة جداً (أقل من ثانية واحدة)."
                continue
            result = classify_lead_signal(raw, fs, lead, external_r_locs=external_r_locs)
            if "error" in result:
                errors[lead] = result["error"]
            else:
                per_lead_results[lead] = result
        else:
            errors[lead] = f"صيغة ملف غير مدعومة: {ext}"

    if not per_lead_results:
        return jsonify({"error": "لم يتم رفع أي قطب صالح.", "details": errors}), 400

    # دمج فعلي لكل الأقطاب المتوفرة (صور و/أو إشارات معاً) بتصويت مرجّح واحد،
    # بغض النظر عن نوع المصدر لكل قطب — هذا يصلح الفجوة التي كانت تمنع
    # دمج الصور المتعددة سابقاً (كانت تُرجع خطأً بدل قرار نهائي).
    result = combine_lead_probabilities(per_lead_results)
    if "error" not in result:
        result["reference_lead_alignment_used"] = reference_lead_signal is not None
    result["errors"] = errors
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
