"""
ecg_pipeline.reference
========================
بناء المرجع الطبيعي الإحصائي (Normal Reference Envelope) وتطبيق
المقارنة المرجعية (Reference-Subtraction / Stemming).
"""

from __future__ import annotations
import numpy as np


def build_class_envelope(beats: np.ndarray, k: float = 2.5,
                          trim_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    يبني مدى إحصائي (المتوسط ± k×الانحراف المعياري) لأي فئة من النبضات
    (طبيعية أو مرضية على حد سواء)، بعد **استبعاد الشواذ أولاً**.

    لماذا استبعاد الشواذ مهم: أي فئة (حتى المرضية) تحوي عملياً بعض
    النبضات "السيئة" — محاذاة غير دقيقة حول R (أو حول القمة المرجعية
    المستخدَمة)، ضوضاء متبقية، أو نبضات غير نمطية حتى ضمن نفس الفئة.
    ضم هذي النبضات الشاذة مباشرة لحساب المتوسط/الانحراف يُشوّه "المرجع"
    الناتج ويجعله أقل تمثيلاً للنمط الحقيقي للفئة.

    الطريقة: نحسب متوسطاً أولياً تقريبياً (median بدل mean، أكثر مقاومة
    للشواذ)، نقيس بُعد كل نبضة عنه (مجموع مربعات الفروق)، ثم نستبعد نسبة
    trim_fraction من النبضات الأبعد قبل حساب (mean ± k*std) النهائي على
    الباقي فقط.

    Parameters
    ----------
    trim_fraction : نسبة النبضات الأبعد عن النمط النموذجي التي تُستبعد
        قبل حساب الإحصائيات النهائية (افتراضياً 10%). صفر يعني عدم
        استبعاد أي شيء (نفس السلوك القديم لـbuild_normal_envelope).
    """
    beats = np.asarray(beats, dtype=float)
    n = beats.shape[0]

    if trim_fraction > 0 and n >= 10:
        pilot_median = np.median(beats, axis=0)
        distances = np.sum((beats - pilot_median) ** 2, axis=1)
        n_keep = max(int(n * (1 - trim_fraction)), n - 1)  # نبقي على الأقل كل الحالات إلا الأشذ
        keep_idx = np.argsort(distances)[:n_keep]
        beats = beats[keep_idx]

    mu = beats.mean(axis=0)
    sigma = beats.std(axis=0)
    return mu - k * sigma, mu + k * sigma


def build_normal_envelope(normal_beats: np.ndarray, k: float = 2.5,
                           trim_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    يبني المدى الطبيعي — غلاف رقيق فوق build_class_envelope للحفاظ على
    التوافق مع الاستخدام القديم (اسم الدالة كان خاصاً بـ"الطبيعي" فقط،
    والآن نفس المنطق يُطبَّق على أي فئة عبر build_class_envelope).
    """
    return build_class_envelope(normal_beats, k=k, trim_fraction=trim_fraction)


def stem_beat(beat: np.ndarray, ref_min: np.ndarray, ref_max: np.ndarray) -> np.ndarray:
    """يصفّر أي عينة تقع داخل المدى الطبيعي؛ يُبقي الباقي (الانحراف المرضي)."""
    stemmed = beat.copy()
    within_range = (stemmed >= ref_min) & (stemmed <= ref_max)
    stemmed[within_range] = 0
    return stemmed


def stem_beat_with_class_envelopes(beat: np.ndarray, normal_range: tuple[np.ndarray, np.ndarray],
                                    class_ranges: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    نسخة محسَّنة من stem_beat: تُبقي فقط الانحرافات عن الطبيعي التي **تطابق
    فعلياً نمطاً مرضياً معروفاً على الأقل**، بدل إبقاء أي انحراف عن
    الطبيعي بغض النظر عن معناه.

    الفكرة (مقترحة صراحة): بدل بناء مرجع للطبيعي فقط، نبني أيضاً مدى
    إحصائي لكل فئة مرضية (بنفس أسلوب build_class_envelope مع استبعاد
    الشواذ). نقطة بالنبضة تُعتبر "إشارة مرضية حقيقية" (تُبقى غير صفرية)
    فقط إذا كانت:
      (أ) خارج مدى الطبيعي عند هذا الموضع (المعيار القديم لوحده)، و
      (ب) داخل مدى فئة مرضية واحدة على الأقل عند نفس الموضع.

    هذا يستبعد الانحرافات "الغريبة" التي لا تطابق أي نمط مرضي معروف (قد
    تكون ضوضاء متبقية، أو خصوصية فردية للمريض لا علاقة لها بالمرض) —
    فيبقى فقط ما هو فعلاً "نمطي لمرض ما"، وهو أدق فيزيولوجياً من مجرد
    "غير طبيعي" الفضفاضة.
    """
    ref_min, ref_max = normal_range
    stemmed = beat.copy()
    outside_normal = (stemmed < ref_min) | (stemmed > ref_max)

    if not class_ranges:
        stemmed[~outside_normal] = 0
        return stemmed

    matches_some_class = np.zeros_like(outside_normal)
    for cls_min, cls_max in class_ranges.values():
        matches_some_class |= (stemmed >= cls_min) & (stemmed <= cls_max)

    keep_mask = outside_normal & matches_some_class
    stemmed[~keep_mask] = 0
    return stemmed


def is_likely_normal(stemmed_beat: np.ndarray, threshold: int = 50) -> bool:
    """قاعدة تصنيف سريعة: إذا تبقى عدد قليل جداً من العينات بعد Stemming، فالنبضة على الأغلب طبيعية."""
    return int(np.count_nonzero(stemmed_beat)) < threshold
