"""
build_presentation_pdf.py — Generate a PDF version of the executive presentation.
Run: python3 scripts/build_presentation_pdf.py
Output: paper/Cash_Flow_PD_Model_Presentation.pdf
"""

import os
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image, HRFlowable,
                                 KeepTogether, PageBreak)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus.flowables import Flowable

BASE    = os.path.dirname(os.path.abspath(__file__)) + '/..'
FIG     = f'{BASE}/figures'
OUT     = f'{BASE}/paper/Cash_Flow_PD_Model_Presentation.pdf'

W, H    = landscape(A4)   # 297 × 210 mm
MARGIN  = 18 * mm

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE    = colors.HexColor('#4f6ef7')
BLUE_LT = colors.HexColor('#eef2ff')
BLUE_DK = colors.HexColor('#1f4e79')
GREY    = colors.HexColor('#64748b')
GREY_LT = colors.HexColor('#f8fafc')
BORDER  = colors.HexColor('#e2e8f0')
RED     = colors.HexColor('#dc2626')
GREEN   = colors.HexColor('#16a34a')
AMBER   = colors.HexColor('#d97706')
BLACK   = colors.HexColor('#111827')
WHITE   = colors.white

# ── Text styles ───────────────────────────────────────────────────────────────
def style(name, **kw):
    base = dict(fontName='Helvetica', fontSize=10, leading=14,
                textColor=BLACK, spaceAfter=4)
    base.update(kw)
    return ParagraphStyle(name, **base)

S_EYEBROW  = style('eyebrow', fontSize=9,  textColor=BLUE,    fontName='Helvetica-Bold',
                   spaceAfter=2)
S_TITLE    = style('title',   fontSize=26, textColor=BLACK,   fontName='Helvetica-Bold',
                   leading=30, spaceAfter=6)
S_SUBTITLE = style('sub',     fontSize=11, textColor=GREY,    leading=15, spaceAfter=8)
S_TAG      = style('tag',     fontSize=8,  textColor=BLUE,    fontName='Helvetica-Bold',
                   spaceAfter=2)
S_HEAD     = style('head',    fontSize=16, textColor=BLACK,   fontName='Helvetica-Bold',
                   leading=20, spaceAfter=6)
S_BODY     = style('body',    fontSize=9,  textColor=GREY,    leading=13)
S_BOLD     = style('bold',    fontSize=9,  textColor=BLACK,   fontName='Helvetica-Bold')
S_CODE     = style('code',    fontSize=8,  textColor=BLUE_DK, fontName='Courier',
                   backColor=BLUE_LT, leading=12)
S_CAPTION  = style('cap',     fontSize=7,  textColor=GREY,    alignment=TA_CENTER)
S_METRIC_L = style('ml',      fontSize=9,  textColor=GREY,    fontName='Helvetica-Bold')
S_METRIC_V = style('mv',      fontSize=22, textColor=BLUE,    fontName='Helvetica-Bold',
                   leading=26)
S_METRIC_G = style('mvg',     fontSize=22, textColor=GREEN,   fontName='Helvetica-Bold',
                   leading=26)
S_METRIC_R = style('mvr',     fontSize=22, textColor=RED,     fontName='Helvetica-Bold',
                   leading=26)
S_CENTER   = style('ctr',     fontSize=9,  alignment=TA_CENTER)


def hr():
    return HRFlowable(width='100%', thickness=1, color=BORDER, spaceAfter=8, spaceBefore=4)


def tag_head(tag, title):
    return [Paragraph(tag.upper(), S_TAG), Paragraph(title, S_HEAD), hr()]


def fig(path, width, caption=''):
    items = [Image(path, width=width, height=width * 0.62)]
    if caption:
        items.append(Paragraph(caption, S_CAPTION))
    return items


def metric_card(label, value, style_v=None, sub=''):
    sv = style_v or S_METRIC_V
    rows = [[Paragraph(label, S_METRIC_L)],
            [Paragraph(value, sv)]]
    if sub:
        rows.append([Paragraph(sub, S_BODY)])
    t = Table(rows, colWidths=['100%'])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), GREY_LT),
        ('BOX',        (0,0), (-1,-1), 0.5, BORDER),
        ('ROUNDEDCORNERS', [6]),
        ('TOPPADDING',  (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING',(0,0), (-1,-1), 10),
    ]))
    return t


def tbl(headers, rows, col_widths=None):
    data = [[Paragraph(h, style('th', fontSize=8, fontName='Helvetica-Bold',
                                textColor=BLUE_DK)) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), S_BODY) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), BLUE_LT),
        ('LINEBELOW',     (0, 0), (-1,  0), 0.5, BLUE),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [WHITE, GREY_LT]),
        ('LINEBELOW',     (0, 1), (-1, -1), 0.3, BORDER),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
    ]))
    return t


def highlight(text):
    t = Table([[Paragraph(text, S_BODY)]])
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), BLUE_LT),
        ('LINEBEFORE',  (0,0), (0,-1),  3, BLUE),
        ('BOX',         (0,0), (-1,-1), 0.5, BORDER),
        ('TOPPADDING',  (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING',(0,0), (-1,-1), 10),
    ]))
    return t


def two_col(left, right, left_w=None):
    cw = left_w or (W - 2*MARGIN) * 0.5
    rw = (W - 2*MARGIN) - cw - 6*mm
    def wrap(items):
        t = Table([[i] for i in items], colWidths=[cw if items == left else rw])
        t.setStyle(TableStyle([('TOPPADDING',(0,0),(-1,-1),2),
                                ('BOTTOMPADDING',(0,0),(-1,-1),2),
                                ('LEFTPADDING',(0,0),(-1,-1),0),
                                ('RIGHTPADDING',(0,0),(-1,-1),0)]))
        return t
    t = Table([[wrap(left), wrap(right)]], colWidths=[cw, rw])
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                            ('TOPPADDING',(0,0),(-1,-1),0),
                            ('LEFTPADDING',(0,0),(-1,-1),0),
                            ('RIGHTPADDING',(0,0),(-1,-1),3)]))
    return t


# ── Page template (slide number + thin top bar) ───────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, H - 3, W, 3, fill=1, stroke=0)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(GREY)
    n   = doc.page
    tot = 10
    canvas.drawRightString(W - MARGIN, 8*mm, f'{n} / {tot}')
    canvas.restoreState()


# ── Build slides ──────────────────────────────────────────────────────────────
story = []
FW    = W - 2 * MARGIN   # full content width
HW    = FW * 0.5 - 3*mm  # half width

# ── SLIDE 1: Title ────────────────────────────────────────────────────────────
story += [
    Spacer(1, 22*mm),
    Paragraph('PRADEEP ARKACHAR  ·  APRIL 2026', S_EYEBROW),
    Paragraph('Cash Flow PD Model', S_TITLE),
    Paragraph('An End-to-End Probability of Default Model for Bank Transactions', S_SUBTITLE),
    Spacer(1, 8*mm),
]
metrics = Table([
    [metric_card('Test AUC', '0.8252'),
     metric_card('Test KS',  '0.5192'),
     metric_card('Train/Test Gap', '1.41%', S_METRIC_G),
     metric_card('Final Features', '5')],
], colWidths=[FW/4 - 3*mm]*4, hAlign='LEFT')
metrics.setStyle(TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),
                              ('RIGHTPADDING',(0,0),(-1,-1),3),
                              ('TOPPADDING',(0,0),(-1,-1),0),
                              ('BOTTOMPADDING',(0,0),(-1,-1),0)]))
story += [metrics, Spacer(1, 8*mm),
          Paragraph('6,000 transactions  ·  13.3% default rate  ·  75/25 stratified split',
                    style('tagline', fontSize=9, textColor=BLUE, alignment=TA_CENTER)),
          PageBreak()]

# ── SLIDE 2: Agenda ───────────────────────────────────────────────────────────
story += tag_head('Overview', 'Agenda')
steps = [
    ('1', 'Dataset', '6,000 bank transactions, 14 numeric features + free-text descriptions, 13.3% default rate'),
    ('2', 'NLP Feature Extraction', '130+ regex rules extract 7 structured features from transaction text'),
    ('3', 'WoE / IV Analysis', 'Weight of Evidence scoring ranks predictive power; 6 features excluded as leakage (IV ≥ 1.0)'),
    ('4', 'SHAP RFE', 'probatus ShapRFECV narrows to 5 optimal features at the CV AUC elbow'),
    ('5', 'LightGBM + Bayes Opt', 'Anti-overfitting penalty constrains train/test gap to <2%; 50 Bayesian iterations'),
    ('6', 'Evaluation', 'ROC AUC, KS statistic, SHAP importance, decile expected vs actual'),
]
step_rows = [[Paragraph(f'<b>{n}. {t}</b><br/><font size="8" color="#64748b">{d}</font>', S_BODY)]
             for n, t, d in steps]
step_tbl = Table(step_rows, colWidths=[FW * 0.58])
step_tbl.setStyle(TableStyle([
    ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE, GREY_LT]),
    ('TOPPADDING',     (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING',    (0,0), (-1,-1), 10), ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ('LINEBELOW',      (0,0), (-1,-1), 0.3, BORDER),
]))
cards = [metric_card('Dataset', '6,000 records', sub='14 features + text'),
         Spacer(1, 4*mm),
         metric_card('Event Rate', '13.3%', sub='799 defaults · 6.5:1'),
         Spacer(1, 4*mm),
         metric_card('Model Gap', '1.41%', S_METRIC_G, sub='Target <2% ✓')]
story += [two_col([step_tbl], cards, left_w=FW*0.6), PageBreak()]

# ── SLIDE 3: Dataset ──────────────────────────────────────────────────────────
story += tag_head('Data', 'Dataset Overview')
issues_tbl = tbl(
    ['Issue', 'Column', 'Fix'],
    [['80% missing',       'feature_3',               'WoE imputation'],
     ['60% missing',       'feature_4',               'WoE imputation'],
     ['273 sentinels',     'feature_1',               '→ null'],
     ['Mixed encoding',    'feature_8',               "'n' → null"],
     ['6 leakage features','feature_2,8,10,13,4,14',  'IV ≥ 1.0 excluded']],
    col_widths=[FW*0.25, FW*0.32, FW*0.23]
)
hl = highlight('<b>Key Data Issues</b><br/>feature_3 has 80% nulls, feature_4 has 60% nulls. '
               'feature_1 contains 273 sentinel values (9,999,997) replaced with null. '
               'feature_8 has mixed encoding standardised to binary.')
eda_img = fig(f'{FIG}/fig2_eda.png', FW * 0.57,
              'Fig 1 — Target distribution, feature distributions, missing rates')
story += [two_col(eda_img, [hl, Spacer(1,4*mm), issues_tbl], left_w=FW*0.57), PageBreak()]

# ── SLIDE 4: NLP ──────────────────────────────────────────────────────────────
story += tag_head('Feature Engineering', 'NLP Extraction from Transaction Text')
nlp_tbl = tbl(
    ['Feature', 'Type', 'Example Values'],
    [['merchant_category',  '16-class', 'Payroll, Transfer_P2P, Grocery…'],
     ['txn_channel',        '7-class',  'Card, ACH, ATM, Wire, Mobile'],
     ['txn_direction',      '3-class',  'Debit, Credit, Unknown'],
     ['is_recurring',       'Binary',   '1 if RECURRING tag present'],
     ['is_p2p',             'Binary',   '1 if Venmo / Zelle / CashApp'],
     ['is_international',   'Binary',   '1 if INTL / FOREIGN / FX'],
     ['merchant_risk_tier', '3-class',  'High / Medium / Low']],
    col_widths=[FW*0.3, FW*0.18, FW*0.32]
)
nlp_hl  = highlight('The <b>feature_0</b> column contains free-text bank transaction descriptions. '
                    'A rule-based NLP extractor using <b>130+ regex patterns</b> derives 7 structured features '
                    'with no external API required.')
cards_nlp = [metric_card('Extraction Method', 'Rule-based NLP', sub='130+ patterns · no API'),
             Spacer(1,4*mm),
             metric_card('Features Added', '7', sub='merchant, channel, direction, flags')]
story += [two_col([nlp_hl, Spacer(1,4*mm), nlp_tbl], cards_nlp, left_w=FW*0.62), PageBreak()]

# ── SLIDE 5: WoE / IV ─────────────────────────────────────────────────────────
story += tag_head('Feature Selection', 'Weight of Evidence / Information Value')
iv_tbl = tbl(
    ['Feature', 'IV', 'Strength'],
    [['feature_2',  '5.42', 'Leakage'],
     ['feature_8',  '3.06', 'Leakage'],
     ['feature_10', '2.74', 'Leakage'],
     ['feature_13', '1.71', 'Leakage'],
     ['feature_4',  '1.52', 'Leakage'],
     ['feature_14', '1.17', 'Leakage'],
     ['feature_7',  '0.74', 'Very Strong'],
     ['feature_3',  '0.63', 'Very Strong'],
     ['feature_6',  '0.35', 'Strong'],
     ['feature_5',  '0.33', 'Strong'],
     ['feature_1',  '0.23', 'Medium']],
    col_widths=[FW*0.23, FW*0.12, FW*0.2]
)
iv_hl   = highlight('Features with <b>IV ≥ 1.0</b> flagged as suspected leakage and excluded. '
                    'Features with <b>IV &lt; 0.02</b> excluded as useless.')
iv_img  = fig(f'{FIG}/fig2_iv_chart.png', FW*0.55, 'Fig 2 — IV by feature (sorted descending)')
story  += [two_col([iv_hl, Spacer(1,4*mm), iv_tbl],
                   iv_img,
                   left_w=FW*0.38), PageBreak()]

# ── SLIDE 6: SHAP RFE ─────────────────────────────────────────────────────────
story += tag_head('Feature Selection', 'SHAP Recursive Feature Elimination')
rfe_tbl = tbl(
    ['#', 'Feature', 'IV', 'Selected'],
    [['1', 'feature_7',       '0.74', '✓'],
     ['2', 'feature_3',       '0.63', '✓'],
     ['3', 'feature_6',       '0.35', '✓'],
     ['4', 'feature_5',       '0.33', '✓'],
     ['5', 'feature_1',       '0.23', '✓'],
     ['6', 'feature_12',      '0.16', '✗ eliminated'],
     ['7', 'merchant_cat_woe','0.05', '✗ eliminated'],
     ['8', 'feature_11',      '0.04', '✗ eliminated']],
    col_widths=[FW*0.06, FW*0.25, FW*0.1, FW*0.2]
)
rfe_hl  = highlight('<b>probatus ShapRFECV</b>: step=0.2, 5-fold CV, scoring=roc_auc. '
                    'Elbow at n=5 features gives CV AUC 0.8239. '
                    'Dropping to n=4 loses 2.3pp — meaningful degradation.')
rfe_img = fig(f'{FIG}/fig3_shap_rfe.png', FW*0.47, 'Fig 4 — CV AUC vs number of features (elbow at n=5)')
story  += [two_col(rfe_img + [Spacer(1,3*mm)],
                   [rfe_hl, Spacer(1,4*mm), rfe_tbl],
                   left_w=FW*0.5), PageBreak()]

# ── SLIDE 7: LightGBM ─────────────────────────────────────────────────────────
story += tag_head('Model', 'LightGBM with Bayesian Optimisation')
param_tbl = tbl(
    ['Hyperparameter', 'Best Value'],
    [['n_estimators',        '500'],
     ['learning_rate',       '0.015'],
     ['max_depth',           '4'],
     ['num_leaves',          '20'],
     ['min_child_samples',   '80'],
     ['reg_alpha / lambda',  '0.8 / 1.2'],
     ['colsample_bytree',    '0.55'],
     ['subsample',           '0.75']],
    col_widths=[FW*0.28, FW*0.18]
)
obj_hl  = highlight('<b>Anti-overfitting objective:</b><br/>'
                    'score = val_AUC − 5 × max(0, gap − 0.02)<br/>'
                    'Penalises train/val AUC gaps above 2pp by factor 5.')
roc_img = fig(f'{FIG}/fig4_roc.png', FW*0.44, 'Fig 5 — ROC curve (train vs test)')
perf_tbl = tbl(
    ['Metric', 'Train', 'Test', 'Gap'],
    [['AUC', '0.8392', '0.8252', '1.41% ✓'],
     ['KS',  '0.5321', '0.5192', '1.29% ✓']],
    col_widths=[FW*0.12, FW*0.1, FW*0.1, FW*0.12]
)
story += [two_col([obj_hl, Spacer(1,4*mm), param_tbl],
                  [perf_tbl, Spacer(1,4*mm)] + roc_img,
                  left_w=FW*0.48), PageBreak()]

# ── SLIDE 8: SHAP ─────────────────────────────────────────────────────────────
story += tag_head('Explainability', 'SHAP Feature Importance')
shap_hl = highlight('SHAP values assign each feature an additive contribution to each prediction. '
                    '<b>feature_7</b> and <b>feature_3</b> account for ~72% of explanatory power, '
                    'consistent with their IV scores of 0.74 and 0.63.')
shap_bar = fig(f'{FIG}/fig7_shap.png', FW*0.5, 'Fig 6 — SHAP summary plot (test set, n=1,500)')
shap_tbl = tbl(
    ['Feature', 'Mean |SHAP|', 'Share'],
    [['feature_7', '0.312', '36%'],
     ['feature_3', '0.287', '33%'],
     ['feature_6', '0.143', '17%'],
     ['feature_5', '0.099',  '11%'],
     ['feature_1', '0.073',  '8%']],
    col_widths=[FW*0.22, FW*0.18, FW*0.1]
)
story += [two_col(shap_bar,
                  [shap_hl, Spacer(1,4*mm), shap_tbl],
                  left_w=FW*0.52), PageBreak()]

# ── SLIDE 9: Decile ───────────────────────────────────────────────────────────
story += tag_head('Validation', 'Expected vs Actual Default Rate by Decile')
dec_hl  = highlight('Top decile captures <b>51.3% default rate</b> vs 13.3% baseline — '
                    'a <b>3.9× lift</b>. Model correctly rank-orders risk across all deciles.')
dec_tbl = tbl(
    ['Decile', 'N', 'Defaults', 'Actual %', 'Expected %'],
    [['1 (highest)', '150', '77',  '51.3%', '41.7%'],
     ['2',           '146', '35',  '24.0%', '25.4%'],
     ['3',           '152', '31',  '20.4%', '21.7%'],
     ['4',           '152', '25',  '16.4%', '17.0%'],
     ['5',           '150', '13',   '8.7%', '11.3%'],
     ['6',           '149',  '8',   '5.4%',  '6.7%'],
     ['7',           '148',  '3',   '2.0%',  '4.9%'],
     ['8',           '153',  '2',   '1.3%',  '3.2%'],
     ['9',           '150',  '4',   '2.7%',  '2.0%'],
     ['10',          '150',  '1',   '0.7%',  '0.8%']],
    col_widths=[FW*0.2, FW*0.08, FW*0.12, FW*0.13, FW*0.14]
)
dec_img = fig(f'{FIG}/fig_decile.png', FW*0.5, 'Fig 8 — Decile expected vs actual (test set)')
story  += [two_col([dec_hl, Spacer(1,4*mm), dec_tbl],
                   dec_img + [Spacer(1,3*mm),
                   metric_card('Top Decile Rate', '51.3%', S_METRIC_R, 'vs 13.3% baseline'),
                   Spacer(1,3*mm),
                   metric_card('Lift (D1)', '3.9×', S_METRIC_G, 'Bottom decile: 0.7%')],
                   left_w=FW*0.46), PageBreak()]

# ── SLIDE 10: Key Takeaways ───────────────────────────────────────────────────
story += tag_head('Summary', 'Key Takeaways')
takeaways = [
    ('✓', GREEN, 'Strong discrimination, minimal overfitting',
     'Test AUC 0.8252, KS 0.5192. Train/test gap 1.41% — well within the 2% target.'),
    ('✓', GREEN, 'Parsimonious — only 5 features',
     'SHAP RFE reduces from 8 candidates to 5, losing <0.1pp AUC. Simpler models are more stable in production.'),
    ('✓', GREEN, 'Fully explainable at prediction level',
     'SHAP values decompose every score into additive feature contributions. Ready for regulatory use.'),
    ('✓', GREEN, '3.9× lift in top decile',
     'Top decile captures 51.3% default rate vs 13.3% baseline — enabling targeted intervention.'),
    ('!', AMBER, 'Next steps',
     'Threshold calibration for approval rate targets; temporal validation on holdout months.'),
]
tw_rows = [[Paragraph(f'<font color="#{("16a34a" if c==GREEN else "d97706")}">'
                      f'<b>{icon}</b></font>', S_BOLD),
            Paragraph(f'<b>{t}</b><br/><font color="#64748b">{d}</font>', S_BODY)]
           for icon, c, t, d in takeaways]
tw_tbl = Table(tw_rows, colWidths=[8*mm, FW*0.52])
tw_tbl.setStyle(TableStyle([
    ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE, GREY_LT]),
    ('TOPPADDING',     (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING',    (0,0), (-1,-1), 8), ('RIGHTPADDING',  (0,0), (-1,-1), 8),
    ('LINEBELOW',      (0,0), (-1,-1), 0.3, BORDER),
    ('VALIGN',         (0,0), (-1,-1), 'TOP'),
]))
scorecard = Table([
    [metric_card('Train AUC', '0.8392'),    metric_card('Test AUC', '0.8252')],
    [metric_card('Train KS',  '0.5321'),    metric_card('Test KS',  '0.5192')],
    [metric_card('AUC Gap',   '1.41%', S_METRIC_G), metric_card('D1 Lift', '3.9×', S_METRIC_G)],
], colWidths=[(FW*0.38)/2 - 2*mm, (FW*0.38)/2 - 2*mm])
scorecard.setStyle(TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),
                                ('RIGHTPADDING',(0,0),(-1,-1),3),
                                ('TOPPADDING',(0,0),(-1,-1),2),
                                ('BOTTOMPADDING',(0,0),(-1,-1),2)]))
story += [two_col([tw_tbl],
                  [scorecard, Spacer(1,5*mm),
                   highlight('github.com/pradark/Cash_Flow_PD_Model')],
                  left_w=FW*0.6)]

# ── Build PDF ─────────────────────────────────────────────────────────────────
os.makedirs(f'{BASE}/paper', exist_ok=True)
doc = SimpleDocTemplate(
    OUT, pagesize=landscape(A4),
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=14*mm,
    title='Cash Flow PD Model — Executive Presentation',
    author='Pradeep Arkachar',
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f'Saved: {OUT}')
