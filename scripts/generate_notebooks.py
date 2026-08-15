"""Genera los 7 notebooks del proyecto."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

COMMON_SETUP = """import sys
from pathlib import Path

_bootstrap = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(_bootstrap / 'src') not in sys.path:
    sys.path.insert(0, str(_bootstrap / 'src'))

from discurso_odio.paths import PROJECT_ROOT

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_all_splits, load_split, get_class_weights
from discurso_odio.paths import REPORTS_FIGURES, REPORTS_METRICS
from discurso_odio.utils.plotting import setup_plot_style, get_label_names, get_class_palette
from discurso_odio.utils.seed import set_seed

settings = load_settings()
set_seed(settings['project']['seed'])
setup_plot_style()
LABELS = get_label_names()
REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
print('Proyecto:', settings['project']['name'])
"""


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "cells": cells,
    }


def md(source):
    lines = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(source):
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [l + "\n" for l in lines],
        "outputs": [],
        "execution_count": None,
    }


NOTEBOOKS = {
    "01_eda.ipynb": [
        md("# 01 — Análisis Exploratorio de Datos (EDA)\n\nReproduce el informe v1.1 sobre el corpus salvadoreño."),
        code(COMMON_SETUP),
        code("""splits = load_all_splits()
corpus = splits['corpus']
train, val, test = splits['train'], splits['val'], splits['test']
print(f'Corpus: {len(corpus)} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}')
corpus.head(3)"""),
        md("## 2. Auditoría de integridad"),
        code("""audit = {
    'filas_totales': len(corpus),
    'ids_duplicados': corpus['ID'].duplicated().sum(),
    'textos_duplicados': corpus['texto_modelo'].duplicated().sum(),
    'usuarios_repetidos': corpus['Usuario_ID_Anonimizado'].duplicated().sum(),
}
pd.Series(audit).to_frame('valor')"""),
        md("## 3. Distribución de clases"),
        code("""class_dist = corpus['clase'].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
class_dist.plot(kind='bar', color=get_class_palette(), ax=ax)
ax.set_title('Distribución de clases canónicas')
ax.set_ylabel('Registros')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(REPORTS_FIGURES / '01_distribucion_clases.png')
plt.show()
class_dist"""),
        md("## 3.1 Pisos de referencia sin modelo"),
        code("""from discurso_odio.evaluation.metrics import trivial_baselines
counts = corpus['label'].value_counts().to_dict()
baselines = trivial_baselines(counts)
pd.Series(baselines, name='F1-macro').to_frame()"""),
        md("## 4. Análisis cross-platform"),
        code("""from scipy.stats import chi2_contingency, mannwhitneyu

ct = pd.crosstab(corpus['Plataforma'], corpus['clase'])
chi2, p, dof, _ = chi2_contingency(ct)
n = ct.sum().sum()
cramers_v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5

x_len = corpus[corpus['Plataforma']=='X']['n_palabras']
fb_len = corpus[corpus['Plataforma']=='Facebook']['n_palabras']
u_stat, u_p = mannwhitneyu(x_len, fb_len, alternative='two-sided')

pd.DataFrame({
    'prueba': ['Chi-cuadrado', 'V de Cramér', 'Mann-Whitney U (longitud)'],
    'estadistico': [f'{chi2:.1f}', f'{cramers_v:.3f}', f'{u_stat:.0f}'],
    'p_valor': [f'{p:.2e}', '-', f'{u_p:.2e}'],
})"""),
        code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pd.crosstab(corpus['Plataforma'], corpus['clase'], normalize='index').plot(
    kind='bar', stacked=True, ax=axes[0], color=get_class_palette())
axes[0].set_title('Composición por plataforma')
corpus.boxplot(column='n_palabras', by='Plataforma', ax=axes[1])
axes[1].set_title('Longitud por plataforma')
plt.suptitle('')
plt.tight_layout()
plt.savefig(REPORTS_FIGURES / '02_plataforma_clases.png')
plt.show()"""),
        md("## 5. Caracterización lingüística"),
        code("""dialect_stats = {
    'textos_con_jerga': f\"{(corpus['n_jerga']>0).mean()*100:.1f}%\",
    'media_jerga': corpus['n_jerga'].mean(),
    'lift_toxicidad_jerga': (
        (corpus[corpus['n_jerga']>0]['label']>0).mean() /
        max((corpus[corpus['n_jerga']==0]['label']>0).mean(), 1e-9)
    ),
}
pd.Series(dialect_stats, name='valor')"""),
        md("## 7. Tokenización mBERT — cobertura max_length"),
        code("""from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(settings['model']['name'])
lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in corpus['texto_modelo'].head(500)]
p99 = np.percentile(lengths, 99)
print(f'Percentil 99 tokens: {p99:.0f} | max_length configurado: {settings[\"model\"][\"max_length\"]}')
plt.hist(lengths, bins=30, color='#3498db', edgecolor='white')
plt.axvline(settings['model']['max_length'], color='red', linestyle='--', label='max_length=128')
plt.title('Distribución de longitud en tokens (muestra 500)')
plt.xlabel('Tokens')
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_FIGURES / '03_token_lengths.png')
plt.show()"""),
    ],
    "02_preprocesamiento_features.ipynb": [
        md("# 02 — Preprocesamiento y Feature Engineering\n\nVerifica el pipeline de normalización y genera features."),
        code(COMMON_SETUP + """
from discurso_odio.features.normalization import (
    normalize_for_model, has_accents_or_enye, has_uppercase,
    has_punctuation, has_mention_literal, has_url_literal, has_emoji_raw,
)
from discurso_odio.features.text_features import build_text_features
"""),
        code("""corpus = load_split('corpus')
train = load_split('train')
print(len(corpus), 'registros')"""),
        md("## Verificación del pipeline de normalización"),
        code("""orig = corpus['texto_original']
modelo = corpus['texto_modelo']
verification = pd.DataFrame({
    'propiedad': ['tildes/ñ', 'mayúsculas', 'puntuación', 'mención literal', 'URL literal', 'emoji crudo'],
    'texto_original': [
        orig.apply(has_accents_or_enye).sum(),
        orig.apply(has_uppercase).sum(),
        orig.apply(has_punctuation).sum(),
        orig.apply(has_mention_literal).sum(),
        orig.apply(has_url_literal).sum(),
        orig.apply(has_emoji_raw).sum(),
    ],
    'texto_modelo': [
        modelo.apply(has_accents_or_enye).sum(),
        modelo.apply(has_uppercase).sum(),
        modelo.apply(has_punctuation).sum(),
        modelo.apply(lambda t: '@usuario' in str(t)).sum(),
        modelo.apply(lambda t: 'http://url' in str(t)).sum(),
        modelo.apply(has_emoji_raw).sum(),
    ],
})
verification"""),
        md("## Feature engineering"),
        code("""features_df = build_text_features(corpus)
feature_cols = [c for c in features_df.columns if c.startswith(('n_', 'upper_', 'exclamation', 'question', 'has_', 'es_', 'sarcasmo', 'ironia', 'contexto', 'dialecto'))]
features_df[feature_cols].describe().T"""),
        md("## Pesos de clase (solo train)"),
        code("""weights = get_class_weights(train)
expected = {3: 1.3779, 2: 1.2099, 0: 0.9404, 1: 0.7224}
comparison = pd.DataFrame({
    'clase': [LABELS[i] for i in sorted(weights)],
    'peso_calculado': [round(weights[i], 4) for i in sorted(weights)],
    'peso_informe': [expected[i] for i in sorted(weights)],
})
comparison"""),
        code("""comparison.to_csv(REPORTS_METRICS / 'class_weights.csv', index=False)
print('Pesos exportados.')"""),
    ],
    "03_baselines_tfidf.ipynb": [
        md("# 03 — Modelos Baseline TF-IDF\n\nRegresión Logística, LinearSVC y MultinomialNB como referencia."),
        code(COMMON_SETUP + """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import joblib
from discurso_odio.paths import MODELS_DIR
"""),
        code("""train = load_split('train')
test = load_split('test')
X_train, y_train = train['texto_modelo'], train['label']
X_test, y_test = test['texto_modelo'], test['label']
print(f'Train: {len(X_train)} | Test: {len(X_test)}')"""),
        code("""vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)
models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'LinearSVC': LinearSVC(class_weight='balanced', random_state=42),
    'MultinomialNB': MultinomialNB(),
}
results = {}
for name, clf in models.items():
    pipe = Pipeline([('tfidf', vectorizer), ('clf', clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    results[name] = {'f1_macro': f1, 'predictions': preds}
    print(f'{name}: F1-macro = {f1:.4f}')
    print(classification_report(y_test, preds, target_names=LABELS))
    joblib.dump(pipe, MODELS_DIR / f'baseline_{name.lower()}.joblib')"""),
        code("""results_df = pd.DataFrame({k: {'f1_macro': v['f1_macro']} for k, v in results.items()}).T
results_df.to_csv(REPORTS_METRICS / 'baselines_tfidf.csv')
results_df"""),
    ],
    "04_finetuning_mbert.ipynb": [
        md("# 04 — Fine-tuning mBERT\n\n`bert-base-multilingual-cased` con pérdida ponderada y early stopping."),
        code(COMMON_SETUP + """
from discurso_odio.models.mbert import train_mbert, get_device
from discurso_odio.paths import MODELS_DIR
"""),
        code("""print('Dispositivo:', get_device())
train = load_split('train')
val = load_split('val')
print(f'Train: {len(train)} | Val: {len(val)}')"""),
        md("## Entrenamiento principal (texto_modelo)"),
        code("""model_path = MODELS_DIR / 'mbert-sv'
if model_path.exists() and (model_path / 'config.json').exists():
    print('Modelo ya entrenado en', model_path, '- omitiendo re-entrenamiento.')
    print('Para re-entrenar, elimine el directorio models/mbert-sv/')
else:
    trainer = train_mbert(train, val, output_dir=model_path, text_col='texto_modelo')
    print('Métricas validación:', trainer.evaluate())"""),
        md("## Ablación: texto_agresivo (grupo de control)"),
        code("""# Descomentar para ejecutar ablación completa (~30-60 min adicionales)
# trainer_ag = train_mbert(train, val, output_dir=MODELS_DIR / 'mbert-sv-agresivo', text_col='texto_agresivo')
# print('Ablación agresivo:', trainer_ag.evaluate())
print('Ablación documentada. Ejecutar celdas comentadas para comparar variantes.')"""),
    ],
    "05_metricas_y_comparacion.ipynb": [
        md("# 05 — Métricas y Comparación\n\nCuadro completo de métricas sobre el conjunto de test."),
        code(COMMON_SETUP + """
from discurso_odio.evaluation.metrics import compute_all_metrics, bootstrap_ci, mcnemar_test, f1_score
from discurso_odio.evaluation.slices import slice_metrics, dialect_bias_probe
from discurso_odio.models.mbert import predict
from discurso_odio.paths import MODELS_DIR
import json
import joblib
from sklearn.metrics import f1_score as sk_f1
"""),
        code("""test = load_split('test')
model_dir = MODELS_DIR / 'mbert-sv'
y_pred, y_prob = predict(test['texto_modelo'].tolist(), model_dir)
y_true = test['label'].values
metrics = compute_all_metrics(y_true, y_pred, y_prob)
print('F1-macro:', round(metrics['f1_macro'], 4))
print('Accuracy:', round(metrics['accuracy'], 4))
print('MAE ordinal:', round(metrics['ordinal_mae'], 4))
print(metrics['classification_report'])"""),
        code("""# Intervalos de confianza bootstrap
f1_mean, f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, lambda yt, yp: sk_f1(yt, yp, average='macro'))
acc_mean, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, lambda yt, yp: (yt==yp).mean())
print(f'F1-macro bootstrap: {f1_mean:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]')
print(f'Accuracy bootstrap: {acc_mean:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]')
print(f'Objetivo F1>0.80: {\"CUMPLE\" if f1_mean >= 0.80 else \"NO CUMPLE\"}')"""),
        code("""# McNemar vs baselines
mcnemar_results = []
for baseline_name in ['logisticregression', 'linearsvc', 'multinomialnb']:
    path = MODELS_DIR / f'baseline_{baseline_name}.joblib'
    if path.exists():
        pipe = joblib.load(path)
        bl_pred = pipe.predict(test['texto_modelo'])
        mcn = mcnemar_test(y_true, y_pred, bl_pred)
        mcnemar_results.append({'baseline': baseline_name, **mcn})
pd.DataFrame(mcnemar_results)"""),
        code("""# Métricas por cortes (mBERT + XLM-R si hay predicciones auditadas)
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
from compute_slice_metrics import compute_all
from discurso_odio.evaluation.slices import dialect_bias_probe, export_dialect_bias
from discurso_odio.paths import TEST_CSV

slice_summary = compute_all(model_dir=model_dir, include_xlmr=True)
xlmr_status = slice_summary.attrs.get('xlmr_status', 'unknown')
from discurso_odio.evaluation.slices import export_slice_metrics
export_slice_metrics(
    slice_summary,
    REPORTS_METRICS,
    extra_metadata={'xlmr_status': xlmr_status},
    merge=True,
)
bias = dialect_bias_probe(
    test, y_true, y_pred,
    model='mbert',
    checkpoint=str(model_dir),
    split=TEST_CSV.name,
)
export_dialect_bias(bias, REPORTS_METRICS, extra_metadata={'xlmr_status': xlmr_status})
print('Sonda sesgo dialectal:')
print(bias[['group', 'n', 'fp', 'fpr', 'fpr_ci_lo', 'fpr_ci_hi']].to_string(index=False))
slice_summary"""),
        code("""# Matriz de confusión
from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABELS, ax=ax, cmap='Blues')
plt.title('Matriz de confusión — mBERT test')
plt.tight_layout()
plt.savefig(REPORTS_FIGURES / '04_confusion_matrix.png')
plt.show()"""),
        code("""# Exportar métricas
export = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
export['f1_bootstrap_lo'] = f1_lo
export['f1_bootstrap_hi'] = f1_hi
with open(MODELS_DIR / 'mbert-sv' / 'test_metrics.json', 'w') as f:
    json.dump(export, f, indent=2)
pd.DataFrame([export]).to_csv(REPORTS_METRICS / 'mbert_test_metrics.csv', index=False)
print('Métricas exportadas.')"""),
    ],
    "06_benchmark_externo.ipynb": [
        md("# 06 — Benchmark Externo Zero-Shot\n\nEvaluación fuera de dominio contra 6 datasets públicos."),
        code(COMMON_SETUP + """
from discurso_odio.evaluation.benchmark import run_benchmark
from discurso_odio.paths import MODELS_DIR
"""),
        code("""model_dir = MODELS_DIR / 'mbert-sv'
assert model_dir.exists(), 'Entrene mBERT primero (notebook 04)'
results = run_benchmark(str(model_dir))
results"""),
        code("""results.to_csv(REPORTS_METRICS / 'benchmark_externo.csv', index=False)
fig, ax = plt.subplots(figsize=(12, 5))
plot_data = results[results['status']=='ok']
if len(plot_data):
    plot_data.plot(x='dataset', y=['f1_macro', 'f1_binary'], kind='bar', ax=ax)
    ax.set_title('Benchmark zero-shot — F1 por dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES / '05_benchmark_externo.png')
    plt.show()
print('Resultados exportados.')"""),
    ],
    "07_interpretabilidad.ipynb": [
        md("# 07 — Interpretabilidad (LIME + SHAP)\n\nExplicaciones locales y sonda de sesgo dialectal."),
        code(COMMON_SETUP + """
from functools import partial
from discurso_odio.interpretability.lime_explainer import (
    build_lime_explainer, make_predict_proba_fn, explain_instance,
    aggregate_lime_weights, jaccard_top_k,
)
from discurso_odio.interpretability.shap_explainer import (
    make_shap_predict_fn, explain_texts_shap, extract_top_tokens,
)
from discurso_odio.models.mbert import predict, get_device, build_model, build_tokenizer
from discurso_odio.paths import MODELS_DIR
from transformers import AutoModelForSequenceClassification
import torch

# Predictor en memoria (mismo patrón que scripts/run_interpretability.py)
class CachedPredictor:
    def __init__(self, model_dir):
        settings = load_settings()
        self.device = get_device()
        self.tokenizer = build_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = settings['model']['max_length']

    @torch.no_grad()
    def predict_proba(self, texts):
        import torch
        texts = [str(t) if t else ' ' for t in texts]
        enc = self.tokenizer(texts, truncation=True, padding=True,
                             max_length=self.max_length, return_tensors='pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
"""),
        code("""test = load_split('test')
model_dir = MODELS_DIR / 'mbert-sv'
predictor = CachedPredictor(model_dir)
y_pred = predictor.predict_proba(test['texto_modelo'].tolist()).argmax(axis=1)
y_true = test['label'].values
samples = []
for cls in range(4):
    idx_correct = np.where((y_true==cls) & (y_pred==cls))[0][:2]
    idx_error = np.where((y_true==cls) & (y_pred!=cls))[0][:1]
    samples.extend(idx_correct.tolist() + idx_error.tolist())
texts_sample = test.iloc[samples[:8]]['texto_modelo'].tolist()
print(f'Instancias seleccionadas: {len(texts_sample)}')"""),
        md("## LIME"),
        code("""explainer = build_lime_explainer()
predict_proba_fn = make_predict_proba_fn(predictor.predict_proba)
lime_explanations = []
for text in texts_sample[:5]:
    exp = explain_instance(text, predict_proba_fn, explainer, num_samples=150)
    lime_explanations.append(exp)
    print(f'Clase predicha: {LABELS[exp["predicted_class"]]}')
    print('Top tokens:', exp['top_tokens'][:5])
    print('---')
global_lime = aggregate_lime_weights(lime_explanations)
global_lime"""),
        md("## SHAP"),
        code("""shap_predict_fn = make_shap_predict_fn(predictor.model, predictor.tokenizer, predictor.device)
shap_texts = texts_sample[:3]
try:
    shap_values = explain_texts_shap(shap_texts, shap_predict_fn, tokenizer=predictor.tokenizer, max_evals=25)
    for i, text in enumerate(shap_texts):
        top = extract_top_tokens(shap_values, i, top_k=5)
        print(f'Texto {i}: {text[:60]}...')
        print('SHAP top:', top)
    shap_ok = True
except Exception as e:
    print(f'SHAP falló ({e}). LIME cubre el requisito.')
    shap_ok = False"""),
        md("## Sonda de sesgo dialectal"),
        code("""# FP en No Tóxico con jerga
fp_mask = (y_true == 0) & (y_pred >= 1)
fp_with_jerga = test.iloc[np.where(fp_mask & (test['n_jerga']>0).values)[0]]
fp_without_jerga = test.iloc[np.where(fp_mask & (test['n_jerga']==0).values)[0]]
print(f'FP No Tóxico con jerga: {len(fp_with_jerga)}')
print(f'FP No Tóxico sin jerga: {len(fp_without_jerga)}')
if len(fp_with_jerga):
    print('Ejemplo FP con jerga:', fp_with_jerga['texto_modelo'].iloc[0][:100])"""),
        code("""# Contraste LIME vs SHAP (Jaccard top-k)
if shap_ok and lime_explanations:
    for i in range(min(3, len(lime_explanations))):
        lime_tokens = lime_explanations[i]['top_tokens']
        shap_top = extract_top_tokens(shap_values, i, top_k=10)
        shap_tokens = [t for t, _ in shap_top]
        j = jaccard_top_k(lime_tokens, shap_tokens)
        print(f'Instancia {i}: Jaccard LIME-SHAP = {j:.3f}')
print('Análisis de interpretabilidad completado.')"""),
    ],
}

for name, cells in NOTEBOOKS.items():
    path = NOTEBOOKS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, ensure_ascii=False, indent=1)
    print(f"Creado: {path}")
