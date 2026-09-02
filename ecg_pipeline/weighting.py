"""
ecg_pipeline.weighting
=========================
تطبيع وترجيح السمات (TF-IDF + قمع التداخل بين الفئات / Class Specificity)
المُطبَّق على النبضات بعد Stemming.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict


def weight_with_class_specificity(stemmed_beats: np.ndarray, labels: np.ndarray,
                                   decimals: int = 1) -> np.ndarray:
    """
    يجمع بين:
      (أ) IDF التقليدي: وزن أعلى للقيم النادرة عبر كل النبضات.
      (ب) الاختصاص الطبقي: وزن أعلى للقيم "المختصة" بفئة واحدة، ووزن
          أقل للقيم المشتركة بقوة بين عدة فئات مرضية (لأنها لا تميّز).
    """
    n_beats = stemmed_beats.shape[0]
    bins = np.round(stemmed_beats, decimals)
    classes = sorted(set(labels))

    class_beat_counts = {c: (labels == c).sum() for c in classes}
    value_freq_per_class = defaultdict(lambda: defaultdict(int))
    doc_freq_total = defaultdict(int)

    for i, row in enumerate(bins):
        cls = labels[i]
        uniq = set(v for v in row if v != 0)
        for v in uniq:
            value_freq_per_class[v][cls] += 1
            doc_freq_total[v] += 1

    specificity, idf = {}, {}
    for v, per_class in value_freq_per_class.items():
        freqs = np.array([per_class.get(c, 0) / class_beat_counts[c] for c in classes])
        specificity[v] = freqs.max() / (freqs.sum() + 1e-9)
        idf[v] = np.log(n_beats / doc_freq_total[v])

    weighted = np.zeros_like(stemmed_beats)
    for i, row in enumerate(bins):
        for j, v in enumerate(row):
            if v != 0:
                weighted[i, j] = stemmed_beats[i, j] * idf.get(v, 0.0) * specificity.get(v, 0.0)
    return weighted


def build_weight_lookup(table: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    يحوّل جدول ترجيح (idf_table أو specificity_table) من قاموس {قيمة: وزن}
    إلى زوج مصفوفات مرتّبة (مفاتيح، أوزان) جاهزة للاستيفاء الخطي.
    يُستدعى مرة واحدة فقط عند تحميل النموذج (وليس بكل نبضة) لأسباب أداء.
    """
    if not table:
        return np.array([]), np.array([])
    keys = np.array(sorted(table.keys()), dtype=float)
    values = np.array([table[k] for k in keys], dtype=float)
    return keys, values


def lookup_weight(v: float, lookup: tuple[np.ndarray, np.ndarray]) -> float:
    """
    يبحث عن وزن القيمة v ضمن جدول مُجهَّز مسبقاً بـ build_weight_lookup.

    لماذا الاستيفاء لا البحث الحرفي: قيم الانحراف بعد Stemming مستمرة
    الطبيعة (مثلاً 0.35 مم انحراف عن الطبيعي)، وليست فئات منفصلة محدودة.
    مريض جديد سيُنتج غالباً قيماً لم تظهر حرفياً بجدول التدريب (خصوصاً مع
    عيّنة تدريب محدودة الحجم). النسخة القديمة كانت تُرجع صفراً في هذه
    الحالة (dict.get(v, 0.0)) — أي تتعامل مع أي انحراف "غريب" على أنه
    عديم الأهمية الطبية، حتى لو كان أشد خطورة من أي قيمة شوهدت بالتدريب.
    الاستيفاء الخطي بين أقرب قيمتين مُشاهدتين يحافظ على الاتجاه الصحيح
    فيزيولوجياً (انحراف أكبر => وزن أكبر عادة)، ويُثَبِّت القيمة عند طرفي
    المدى المُشاهَد بدل الصفر في الحالات الطرفية جداً.
    """
    keys, values = lookup
    if len(keys) == 0:
        return 0.0
    return float(np.interp(v, keys, values))


def weight_single_beat(stemmed_beat: np.ndarray, idf_lookup, specificity_lookup,
                        decimals: int = 1) -> np.ndarray:
    """
    يطبّق أوزان IDF/الاختصاص المحسوبة مسبقاً وقت التدريب على نبضة واحدة
    جديدة وقت الاستدلال، عبر استيفاء خطي بدل مطابقة حرفية.

    idf_lookup / specificity_lookup: إما قاموس خام {قيمة: وزن} (يُحوَّل
    تلقائياً)، أو زوج مصفوفات جاهز من build_weight_lookup (أسرع، مفضّل
    عند التصنيف المتكرر لعدة نبضات).
    """
    if isinstance(idf_lookup, dict):
        idf_lookup = build_weight_lookup(idf_lookup)
    if isinstance(specificity_lookup, dict):
        specificity_lookup = build_weight_lookup(specificity_lookup)

    bins = np.round(stemmed_beat, decimals)
    weighted = np.zeros_like(stemmed_beat)
    for j, v in enumerate(bins):
        if v != 0:
            weighted[j] = (stemmed_beat[j]
                           * lookup_weight(v, idf_lookup)
                           * lookup_weight(v, specificity_lookup))
    return weighted
