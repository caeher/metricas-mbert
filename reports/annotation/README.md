# Revisión y acuerdo de anotación — ISSUE-05

Métricas generadas desde las exportaciones humanas indicadas en agreement.json.

`agreement.json` informa acuerdo porcentual (escala 0–100), kappa de Cohen para dos anotadores o alfa nominal de Krippendorff cuando hay más de dos, con n y denominadores explícitos. El objetivo operativo inicial es kappa/alfa ≥ 0.75; no garantiza F1. `agreement_by_slice.csv` contiene cortes por fuente, plataforma, clase provisional y subclase. `agreement_matrix.csv` y `agreement_by_class.csv` muestran desacuerdos y resultados por clase.

`adjudication_ledger.csv` conserva las respuestas humanas originales y separa decisión final, motivo, adjudicador, versión de guía y revisión de procedencia. Completa manualmente los campos de adjudicación en los casos pendientes; al volver a ejecutar el script, las decisiones se conservan mientras no cambien las respuestas originales.

`qualitative_review.csv` reúne razones y desacuerdos para revisión; completa en el ledger los campos `qualitative_theme` y `qualitative_note`. `qualitative_themes.csv` resume esa codificación humana. `adjudicated_records.csv` contiene solo decisiones con clase final 0–3 y controles de calidad completos. Los casos con contexto insuficiente no se exportan como clase 0. Los registros nuevos requieren además `provenance_review_status=verified`.

`label_changes_by_source.csv` y `label_changes_by_subclass.csv` cuantifican las clases provisionales que cambian frente a decisiones finales. Los registros compartidos por varias fuentes cuentan en cada fuente a la que pertenecen.
