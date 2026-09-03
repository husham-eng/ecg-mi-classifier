"""
ecg_pipeline.panel_detector
==============================
كشف وتقطيع تلقائي لصورة لوحة ECG كاملة (12 قطباً مرصوصة معاً) إلى صور
منفصلة لكل قطب — يعتمد أساساً على الترتيب التقليدي الثابت عالمياً
(I, II, III, aVR, aVL, aVF من الأعلى للأسفل بالعمود الأيسر، وV1-V6
بنفس الترتيب بالعمود الأيمن) كمصدر أساسي موثوق، ويستخدم OCR فقط كتأكيد
إضافي (وليس كمصدر وحيد) — لأن اختبارات فعلية أظهرت أن OCR وحده غير
موثوق بما يكفي (خصوصاً مع تسميات رقمية رومانية بسيطة كـ I/II/III)، بينما
الترتيب الهيكلي شبه عالمي الثبات بهذا النوع من طباعة تخطيط القلب.

⚠️ يفترض حالياً تخطيط "6 صفوف × عمودين زمنيين متتاليين" (الأكثر شيوعاً
بالاختبار الفعلي). تخطيطات أخرى (3×4 جنباً إلى جنب مثلاً) غير مدعومة بعد.
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

LEFT_COLUMN_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]
RIGHT_COLUMN_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]
ALL_LEADS_ORDER = LEFT_COLUMN_LEADS + RIGHT_COLUMN_LEADS

# تطابق أسماء الأقطاب بمخطط المشروع (LeadI بدل I) لمن يحتاجها لاحقاً
PROJECT_LEAD_NAME = {"I": "LeadI", "aVR": "aVR", "V2": "V2", "V6": "V6"}

_OCR_WHITELIST = "IaVRLF123456"
_OCR_CONFIG = f"--psm 7 -c tessedit_char_whitelist={_OCR_WHITELIST}"


def _ink_mask(bgr_crop: np.ndarray) -> np.ndarray:
    """يعزل بكسلات الحبر الأسود (نص/خط) عن الشبكة الملوّنة (وردية عادة)."""
    brightness = bgr_crop.mean(axis=2)
    saturation = bgr_crop.max(axis=2).astype(int) - bgr_crop.min(axis=2).astype(int)
    return ((brightness < 130) & (saturation < 45)).astype(np.uint8) * 255


def detect_grid_bounds(img: np.ndarray) -> tuple[int, int, int, int]:
    """
    يكتشف حدود منطقة الشبكة الفعلية (أعلى/أسفل/يسار/يمين)، مستبعداً مناطق
    العنوان بالأعلى والعلامة المائية/التسمية التوضيحية بالأسفل، عبر تحليل
    نسبة البكسلات غير-البيضاء لكل صف/عمود (منطقة الشبكة مغطاة بالكامل
    تقريباً بالشبكة الملوّنة، بعكس الخلفية البيضاء بمناطق النص المحيطة).
    """
    h, w = img.shape[:2]
    brightness = img.mean(axis=2)
    non_white = (brightness < 240).astype(np.uint8)

    row_coverage = non_white.mean(axis=1)
    # نبحث عن بداية الشريط السفلي الصلب (علامة مائية/تسمية) بالتحرك من
    # أسفل الصورة للأعلى — هذا يتجنّب الخلط مع نص العنوان بالأعلى الذي
    # قد تكون كثافته أيضاً عالية محلياً.
    y = h - 1
    while y > 0 and row_coverage[y] > 0.6:
        y -= 1
    solid_bar_start = y + 1
    search_rows = [y for y in range(solid_bar_start) if row_coverage[y] > 0.85]
    if not search_rows:
        raise ValueError("تعذّر تحديد حدود الشبكة عمودياً — الصورة قد لا تكون لوحة ECG قياسية.")
    top, bottom = min(search_rows), max(search_rows)

    col_coverage = non_white[top:bottom, :].mean(axis=0)
    search_cols = [x for x in range(w) if col_coverage[x] > 0.5]
    if not search_cols:
        raise ValueError("تعذّر تحديد حدود الشبكة أفقياً.")
    left, right = min(search_cols), max(search_cols)

    return top, bottom, left, right


def detect_divider_x(img: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    """
    يكتشف موقع الخط الفاصل بين عمود الأقطاب الطرفية (I..aVF) وعمود الصدرية
    (V1..V6) — أعلى تغطية عمودية ضمن النطاق الأوسط (بين ثلث وثلثي عرض
    الشبكة)، لأن خط الفاصل عادة أكثف من خطوط الشبكة العادية.
    """
    brightness = img.mean(axis=2)
    non_white = (brightness < 240).astype(np.uint8)
    col_coverage = non_white[top:bottom, :].mean(axis=0)

    span = right - left
    search_range = range(left + span // 3, left + 2 * span // 3)
    return max(search_range, key=lambda x: col_coverage[x])


def _find_label_blob(row_crop_bgr: np.ndarray, x_gap: int = 18) -> tuple[int, int, int, int] | None:
    """
    يعزل كتلة التسمية النصية داخل منطقة هامش ضيقة (يسار كل صف)، عبر تحليل
    المكوّنات المتصلة مع استبعاد الخطوط الرفيعة الممتدة (خط الإشارة نفسه)
    والاحتفاظ فقط بكتل مضغوطة (أحرف)، ثم دمج الكتل المتقاربة أفقياً (أحرف
    نفس الكلمة) في كتلة واحدة شاملة.
    """
    ink = _ink_mask(row_crop_bgr)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    candidates = []
    for lbl in range(1, n_labels):
        x, y, w, h, area = stats[lbl]
        aspect = w / max(h, 1)
        if area > 8 and w < 45 and h < 30 and aspect < 3.5:
            candidates.append((x, y, w, h))

    if not candidates:
        return None

    candidates.sort(key=lambda b: b[0])
    merged = [list(candidates[0])]
    for x, y, w, h in candidates[1:]:
        mx, my, mw, mh = merged[-1]
        if x - (mx + mw) <= x_gap:
            nx1, ny1 = min(mx, x), min(my, y)
            nx2, ny2 = max(mx + mw, x + w), max(my + mh, y + h)
            merged[-1] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
        else:
            merged.append([x, y, w, h])

    best = max(merged, key=lambda b: b[2] * b[3])
    return tuple(best)


def _ocr_confirm(row_crop_bgr: np.ndarray, expected_label: str) -> str:
    """
    يحاول تأكيد التسمية المتوقّعة (حسب الترتيب الهيكلي) عبر OCR على كتلة
    النص المعزولة. يُرجع: 'confirmed' (تطابق تام أو قريب جداً)،
    'weak' (كتلة نصية موجودة لكن القراءة غير مطابقة بوضوح)، أو
    'not_found' (لم تُعزل أي كتلة نصية أصلاً).
    """
    blob = _find_label_blob(row_crop_bgr)
    if blob is None:
        return "not_found"

    x, y, w, h = blob
    pad = 5
    ink = _ink_mask(row_crop_bgr)
    label_crop = ink[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    label_crop = 255 - label_crop
    big = cv2.resize(label_crop, (label_crop.shape[1] * 10, label_crop.shape[0] * 10),
                      interpolation=cv2.INTER_LANCZOS4)
    text = pytesseract.image_to_string(big, config=_OCR_CONFIG).strip()
    text_clean = re.sub(r"[^A-Za-z0-9]", "", text)

    if not text_clean:
        return "weak"

    similarity = SequenceMatcher(None, text_clean.lower(), expected_label.lower()).ratio()
    return "confirmed" if similarity >= 0.6 else "weak"


def detect_panel_leads(image_path: str) -> dict:
    """
    نقطة الدخول الرئيسية: يأخذ مسار صورة لوحة ECG كاملة، ويُرجع قاموساً
    {اسم القطب التقليدي: {"crop": مصفوفة BGR، "confidence": ...}} لكل
    الأقطاب الـ12، مبنياً أساساً على الترتيب الهيكلي التقليدي مع تأكيد
    OCR إضافي حيثما أمكن.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("تعذّرت قراءة الصورة.")

    top, bottom, left, right = detect_grid_bounds(img)
    divider_x = detect_divider_x(img, top, bottom, left, right)

    row_h = (bottom - top) / 6
    results = {}

    for i in range(6):
        y1 = int(top + i * row_h)
        y2 = int(top + (i + 1) * row_h)

        left_name = LEFT_COLUMN_LEADS[i]
        left_crop = img[y1:y2, left:divider_x]
        left_margin = img[y1:y2, left:left + 65]
        results[left_name] = {
            "crop": left_crop,
            "confidence": _ocr_confirm(left_margin, left_name),
        }

        right_name = RIGHT_COLUMN_LEADS[i]
        right_crop = img[y1:y2, divider_x:right]
        right_margin = img[y1:y2, divider_x:divider_x + 65]
        results[right_name] = {
            "crop": right_crop,
            "confidence": _ocr_confirm(right_margin, right_name),
        }

    return results
