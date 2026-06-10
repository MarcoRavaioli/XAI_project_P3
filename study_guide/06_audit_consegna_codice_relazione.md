# Audit completo — Consegna ↔ Letteratura ↔ Codice ↔ Relazione

> Obiettivo di questo file: incrociare **quattro fonti indipendenti** —
> (1) i documenti di consegna (`XAI_00b_project_presentation.pdf` +
> la scheda P3 in `XAI_Projects_2026.pdf`), (2) gli 8 paper della letteratura,
> (3) il codice in `src/` (ciò che viene *davvero eseguito*), e (4) la vostra
> relazione in `paper/Your_Paper_Title_Here/Chapters/` — e segnalare ogni punto
> in cui due di queste fonti si contraddicono. **La relazione è l'unica delle
> quattro che potete ancora modificare**, quindi ogni discrepanza qui sotto si
> traduce in un'azione concreta su uno dei file `.tex`.
>
> Legenda: 🔴 = da correggere prima della consegna (rischio concreto in
> discussione) · 🟡 = da migliorare/rifinire · 🟢 = punto di forza, tenerlo a mente.

---

## Executive summary — i 5 punti più urgenti

1. 🔴 **La relazione narra risultati che non esistono ancora nel repository.**
   `results.tex` è uno scheletro vuoto (solo TODO e tabelle commentate), ma
   `abstract.tex` e `conclusion.tex` riportano già **numeri, percentuali e nomi
   di concetti specifici** ("eyes, snouts, background foliage", "6,144
   features", "a significant fraction"). Non esiste, da nessuna parte nel
   repo, un artefatto reale (`out/`, CSV, PNG) a supporto. → **§D.3**
2. 🔴 **Il dataset descritto in `methodology.tex`/`abstract.tex` ("ImageNet-1k
   validation set, covering all 1000 classes") quasi certamente non è quello
   che ha prodotto gli esempi citati nella conclusione.** Il termine
   "**snout**" — citato testualmente in `conclusion.tex` come esempio di
   concetto scoperto — compare **solo** nel vocabolario `imagewoof` (un
   sottoinsieme di 10 razze di cani), non in quello `imagenet`. → **§D.2**
3. 🔴 **L'abstract sovra-rappresenta la portata della valutazione CLIP.**
   Dice "Across 2 monitored layers, the SAEs recover dictionaries of 6,144
   features, of which a significant fraction are successfully mapped" — ma il
   codice etichetta con CLIP **solo le 10 feature più attive per layer**
   (`num_features=10` in `run_pipeline.py:528`), cioè lo **0.16%** del
   dizionario, non un "significant fraction" del totale. → **§D.4**
4. 🟡 **Incoerenza terminologica tra capitoli sulla profondità dei layer**:
   `conclusion.tex` chiama il layer 6 "**early layers**", mentre
   `methodology.tex` (come l'ho appena corretto) e il `README.md` del progetto
   lo chiamano "**mid-network**" — sono la stessa cosa chiamata in due modi
   diversi, e un layer 6 su 12 è letteralmente il *centro* della rete, non
   l'inizio. → **§D.1**
5. 🟡 **Lunghezza**: la consegna chiede esplicitamente un *"short document,
   2/3 pages"*; i soli capitoli di testo (escludendo figure/tabelle dei
   risultati, ancora da aggiungere) totalizzano già **~4.000 parole**
   (~7-9 pagine in formato `article` 11pt, margini 1 pollice) — **2-3 volte**
   più lungo del previsto. → **§A.3**

---

## A. Relazione vs. Consegna — STRUTTURA (`XAI_00b_project_presentation.pdf`)

### A.1 — La struttura dei capitoli rispetta quella richiesta? ✅ Sì, perfettamente

La diapositiva 4 di `XAI_00b` ("Structure of the slides and of the (brief)
recap report") richiede: *Introduction → Related work (Literature review) →
Research gap discussion (overview + identificazione del gap da affrontare) →
Methodology and implementation (overview della soluzione proposta) → Results
and analysis (presentazione + analisi dei risultati) → Conclusion (brief)*.

| Richiesto da XAI_00b | File `.tex` corrispondente | Stato |
|---|---|---|
| Introduction | `introduction.tex` | 🟢 presente, scritto |
| Related work / Literature review | `related_work.tex` | 🟢 presente, scritto, ben strutturato in 4 sotto-sezioni tematiche |
| Research gap discussion (overview + gap specifico) | `research_gap.tex` | 🟢 presente — **struttura esemplare**: 3 sotto-sezioni di "overview gaps" + una sezione finale "The Research Gap We Address" che li sintetizza. Rispecchia *esattamente* la richiesta "Overview research gaps / Identification of the research gap(s) to address" |
| Methodology and implementation (overview soluzione) | `methodology.tex` | 🟢 presente, scritto (e ora aggiornato — vedi §C) |
| Results and analysis (presentazione + analisi) | `results.tex` | 🔴 **scheletro vuoto** — solo TODO e codice commentato (vedi §D.3) |
| Conclusion (brief) | `conclusion.tex` | 🟢 presente, scritto — ma "brief" è discutibile: 347 parole è tra le più lunghe conclusioni che si vedano in un "recap report" di 2-3 pagine |

> 🟢 **Punto di forza da menzionare esplicitamente in discussione**: la
> struttura di `research_gap.tex` è particolarmente ben fatta — segue alla
> lettera lo schema "overview dei gap → gap specifico che affrontiamo" che la
> consegna richiede esplicitamente, e lo fa con tre sotto-gap ben distinti che
> convergono in una sintesi finale. Se il docente cerca un capitolo-modello da
> indicare come esempio, questo potrebbe essere quello.

### A.2 — I 4 criteri di valutazione (diapositiva 8) — coperti?

| Criterio di valutazione (XAI_00b, slide 8) | Coperto? | Dove / Note |
|---|---|---|
| Literature review, Research gaps, Methodology **and Assessment** | 🟡 parzialmente | Literature review e Research gaps: solidi. **"Assessment" è la parte più debole**: la metodologia di valutazione è ben descritta (ora anche la verifica causale, dopo la mia modifica), ma non esiste ancora *nessun risultato valutato* — vedi §D.3 |
| Originality/Novelty | 🟢 | Il framing "SAE + CLIP esterno per ViT puri" è genuinamente originale rispetto alla letteratura citata (lo dimostra bene `research_gap.tex` §4, "The Research Gap We Address") |
| Discussion and Analysis | 🔴 | Non ancora presente — dipende interamente dal completamento di `results.tex` |
| Clarity | 🟢 | La prosa dei capitoli scritti è chiara, ben argomentata, con transizioni logiche esplicite (specialmente in `related_work.tex` §4, che chiude ogni sotto-sezione collegandola al gap) |

### A.3 — 🟡 Lunghezza: "short document, 2/3 pages"

La diapositiva 2 di `XAI_00b` è esplicita: *"Recap document: support your
discussion [...] **Short document, 2/3 pages** (we will provide a template)"*.
I capitoli di testo attuali (esclusi figure/tabelle dei risultati, che
mancano ancora) pesano:

| Capitolo | Parole |
|---|---|
| `abstract.tex` | 176 |
| `introduction.tex` | 361 |
| `related_work.tex` | 844 |
| `research_gap.tex` | 780 |
| `methodology.tex` | ~1.640 (dopo la mia aggiunta della sezione causale) |
| `results.tex` | 153 (solo TODO) |
| `conclusion.tex` | 347 |
| **Totale (senza risultati)** | **~4.300** |

A 11pt, margini 1 pollice, una colonna (`\documentclass[11pt]{article}`,
`\geometry{margin=1in}` — verificato in `template.tex`), questo corrisponde
grosso modo a **7-9 pagine di solo testo**, prima ancora di aggiungere figure,
tabelle e l'analisi dei risultati che porterebbero probabilmente a **10+
pagine** — *3-4 volte* più lungo del "2-3 pagine" richiesto.

> ⚠️ Non è detto che sia un problema bloccante — la consegna dice anche *"we
> will provide a template"*, quindi è possibile che il template ufficiale
> abbia un formato diverso (due colonne? font più piccolo?) che renda 2-3
> pagine più capienti di quanto sembri. **Ma è un punto da chiarire con i
> docenti di riferimento prima della consegna**: chiedere esplicitamente se
> questo documento, nella sua forma attuale (stile "paper" completo con
> abstract, citazioni, equazioni), è ciò che si aspettano per il "recap
> document", o se invece serve una versione condensata. Meglio scoprirlo ora
> che il giorno della consegna.

---

## B. Relazione/Codice vs. Consegna — CONTENUTO (scheda P3 in `XAI_Projects_2026.pdf`, pagg. 7-8)

Confronto punto-per-punto con i 4 blocchi di "Required analysis,
implementation, and evaluation" della scheda specifica del progetto P3:

| Richiesto dalla scheda P3 | Coperto? | Dove / Note |
|---|---|---|
| **Literature Review**: "Review the ViT architecture and pre-training paradigms (supervised ViT, DINO). Study mechanistic interpretability for transformers, covering transformer circuits, SAEs, and automated circuit discovery. Survey existing work on interpreting ViT internals." | 🟢 quasi completo | ViT (Dosovitskiy), circuiti (Elhage), SAE (Bricken/Cunningham), circuit discovery (Conmy) — tutti coperti in `related_work.tex`. **Unico punto debole**: "pre-training paradigms (supervised ViT, **DINO**)" — DINO viene *menzionato* solo come lavoro futuro nella conclusione, non *recensito* in `related_work.tex` come la scheda chiederebbe esplicitamente. Il codice però *supporta già* DINOv2 (`--model facebook/dinov2-base`)! → 🟡 due righe in più in `related_work.tex` su DINO (anche solo per dire "useremo il backbone supervisionato; DINO, allenato self-supervised, rappresenterebbe un interessante termine di paragone — vedi conclusioni") chiuderebbero questo gap a costo quasi zero |
| **Identification of Research Gaps**: "Identify how the spatial structure of ViTs [...] creates opportunities and challenges relative to language models. Highlight the absence of systematic SAE and circuit analyses for vision models as a key gap." | 🟢 completo, anzi sviluppato oltre il richiesto | `research_gap.tex` copre sia "spatial structure" (§2) sia "absence of systematic SAE analyses" (§1) — e aggiunge un terzo gap originale, il "cross-modal labelling problem" (§3), che la scheda non menzionava esplicitamente: è un valore aggiunto genuino |
| **Implementation**: "implement one of the following directions: *Sparse Autoencoders* [...] or *Circuit analysis* (activation patching)" | 🟢 | Direzione SAE scelta e implementata in pieno (`sae.py`, `caching_and_training.py`, `run_pipeline.py`). Coerente con la scelta di **non** implementare Circuit analysis/ACDC — purché questo resti sempre presentato come "scelta consapevole tra due opzioni equivalenti", **mai** come "li abbiamo solo citati e basta" (vedi anche `04_conmy_acdc.md` nello study guide) |
| **Evaluation**: "Assess the interpretability of SAE-learned features [...] through automated labeling via CLIP or an LLM, **and through human evaluation**. Measure circuit faithfulness (if applicable) by ablating identified components [...]. Analyse how features or circuits vary across layers [...]" | 🟡 parzialmente, e con un dettaglio da non perdere | • CLIP auto-labeling: ✅ implementato<br>• Causal verification via ablation: ✅ implementato (`causal_eval.py`)<br>• Cross-layer analysis: ✅ implementato (confronto layer 6 / 11)<br>• **"and through human evaluation"**: 🔴 questo pezzo della richiesta **non risulta implementato né menzionato** da nessuna parte (codice o relazione). La consegna lo chiede esplicitamente, in coordinata con l'automated labeling ("via CLIP **or** an LLM, **and** through human evaluation") |

> 🔴 **Azione consigliata sul punto "human evaluation"**: è probabilmente il
> gap più concreto e facilmente colmabile rispetto alla scheda ufficiale del
> progetto. Non richiede nuovo codice pesante: basta che, durante l'analisi
> delle feature scoperte (quando finalmente avrete dei risultati), **un
> membro del gruppo guardi "alla cieca" i ritagli delle top-K patch e dia un
> giudizio umano indipendente**, da confrontare poi con l'etichetta assegnata
> da CLIP — e che questo confronto/disaccordo venga riportato in
> `results.tex` come parte della "valutazione". È esattamente il tipo di
> "human evaluation" leggera che la scheda si aspetta, e rafforza pure
> l'argomento "la nostra etichettatura non è solo un numero di cosine
> similarity, l'abbiamo anche verificata a occhio". **Non costa quasi nulla
> da fare, e chiude un buco esplicito nei requisiti.**

---

## C. Codice ↔ Methodology — riepilogo delle correzioni già applicate in questa sessione

Questa parte è già stata sistemata (vedi i messaggi precedenti per il
dettaglio completo) — la riporto qui solo come **riepilogo per l'audit**,
perché fa parte del quadro generale di "quanto la relazione rispecchia il
codice realmente eseguito":

| # | Discrepanza trovata | Stato |
|---|---|---|
| 1 | Layer monitorati: `{4,8,12}` nel testo vs. `[5,10]` (layer 6 e 11) nel codice | ✅ corretto in `methodology.tex` |
| 2 | Dizionario SAE: `m = 4d = 3072` nel testo vs. `expansion_factor=8 → m = 6144` nel codice | ✅ corretto |
| 3 | Top-K esemplari: "K=16" nel testo vs. `k=5` nel codice, e "immagini" vs. "ritagli contestuali di patch" | ✅ corretto |
| 4 | Vocabolario: "ImageNet class names + descrittori" nel testo vs. liste curate a mano per dataset (`dataset_concepts`) + prompt template nel codice | ✅ corretto |
| 5 | "Le feature polisemantiche vengono filtrate ed escluse automaticamente" — **non esiste alcun filtro nel codice** (zero occorrenze di soglie/filtri in `run_pipeline.py`/`interpretability.py`) | ✅ corretto — riformulato come valutazione qualitativa |
| 6 | L'intero **stadio di verifica causale** (`causal_eval.py`: ablation, steering, RLD, dose-response) non era descritto in nessuna sottosezione di `methodology.tex`, nonostante sia uno stadio reale e centrale della pipeline | ✅ aggiunta una nuova sottosezione "Causal Verification via Targeted Interventions" |

> 🟡 **Un dettaglio in più che potreste aggiungere** (non bloccante, ma utile
> per la precisione): il `README.md` del progetto rivela che il valore di
> *scaling factor* effettivamente usato per gli esperimenti di steering è
> **S = 5.0** ("Steered Logit Increase (%): Logit shift margin under
> $5.0\times$ feature scaling"). Se questo è davvero il valore con cui
> riporterete i risultati di steering, vale la pena menzionarlo esplicitamente
> in `methodology.tex` (es. "in our experiments we use $s=5$ for steering")
> — rende la sezione completamente autosufficiente e verificabile.

---

## D. Coerenza INTERNA tra i capitoli della relazione — discrepanze nuove, non ancora segnalate

Questa è la parte più delicata dell'audit: confrontando `abstract.tex`,
`methodology.tex`, `results.tex` e `conclusion.tex` *tra loro* (più che
ciascuno con il codice), emergono alcune incoerenze che — proprio perché sono
*interne* alla relazione — sono il tipo di cosa che un lettore attento (o un
docente in sede di discussione) nota per prima, perché non richiede nemmeno
di aprire il codice: basta leggere due paragrafi della stessa relazione e
confrontarli.

### D.1 — 🟡 Terminologia incoerente sulla profondità dei layer

- `conclusion.tex` (riga 1): *"early layers (**Layer 6**) captured low-level
  visual properties [...] while deeper layers (**Layer 11**) encoded [...]"*
- `methodology.tex` (dopo la correzione, riga 11) e `README.md` (riga 57):
  *"**Layer 6 (Mid-Network)**"* / *"**mid-network and late-network depths**"*

Layer 6 su 12 è, numericamente, **il centro esatto** della rete — chiamarlo
"early" in un capitolo e "mid-network" in un altro è un'inconsistenza
terminologica facile da notare e facile da correggere. **Suggerimento**:
uniformare ovunque su "mid-network" / "metà rete" (è anche l'inquadramento più
corretto, ed è quello già usato nel `README.md` e ora in `methodology.tex`).

📍 **Azione**: in `conclusion.tex` riga 1, sostituire *"early layers (Layer
6)"* con *"mid-network depths (Layer 6)"* o equivalente — due parole, ma
elimina una contraddizione visibile a chiunque legga l'abstract e la
conclusione di fila.

### D.2 — 🔴 Il dataset descritto non sembra essere quello che ha prodotto gli esempi citati

Questo è probabilmente il punto più delicato dell'intero audit, quindi vale
la pena spiegarlo per esteso.

**Cosa dice la relazione** (`methodology.tex`, §Backbone Model and Dataset):
> *"For activation extraction we use a subset of the **ImageNet-1k validation
> set, covering all 1000 classes**."*

**Cosa dice la conclusione**, come esempio concreto delle feature scoperte:
> *"deeper layers (Layer 11) encoded progressively more semantic, object-level
> concepts (like **eyes, snouts, or background foliage**)"*

Ora, la parola chiave è **"snout"** (muso): ho controllato i quattro
vocabolari di concetti candidati in `run_pipeline.py` (`dataset_concepts`,
righe 349-431) — quelli che CLIP può effettivamente assegnare come etichetta
— e **"snout" compare *esclusivamente* nel vocabolario `imagewoof`**
(un sottoinsieme ImageNet di **sole 10 razze di cani**, righe 350-364: *fur,
eye, nose, ear, tongue, **snout**, paw, tail, collar, spotted pattern, grass,
collie, labrador*). Non compare né in `imagenet`, né in `imagenette`, né in
`cifar10`.

> 🧩 **Perché questo conta**: CLIP può assegnare *solo* etichette presenti nel
> vocabolario passato in input (è zero-shot, ma su un set chiuso di candidati
> — vedi [`02_concetti/04_clip_e_valutazione_crossmodale.md`](02_concetti/04_clip_e_valutazione_crossmodale.md)
> §3.3). Se "snout" è uscito come etichetta di una feature, **il run che ha
> prodotto quell'osservazione doveva necessariamente usare
> `--dataset imagewoof`** — non "ImageNet-1k validation set covering all 1000
> classes" come dichiara `methodology.tex`.

Questo non è (necessariamente) un errore grave — anzi, usare `imagewoof` (un
sottoinsieme piccolo, scaricabile al volo, con classi semanticamente vicine —
tutte razze di cani) è una **scelta sperimentale del tutto sensata** per un
primo run pilota, specialmente con risorse di calcolo limitate: produce
immagini con parti anatomiche ben definite (musi, orecchie, occhi, pelo) che
sono *l'ideale* per testare se un SAE scopre feature monosemantiche legate a
parti di oggetti. **Il problema non è la scelta del dataset — è che la
relazione ne descrive uno diverso da quello che sembra aver prodotto i
risultati discussi.**

📍 **Azioni possibili** (scegliete quella che riflette la realtà del vostro
lavoro):
1. Se avete davvero usato `imagewoof` (anche solo per un run pilota/di
   sviluppo): **aggiornate `methodology.tex`** per dirlo esplicitamente — e
   anzi *argomentate* perché è una scelta valida ("un sottoinsieme con classi
   visivamente affini permette di verificare se il SAE distingue concetti
   *fini*, come parti anatomiche, piuttosto che solo categorie macroscopiche
   diverse tra loro)". Questo trasforma una "incongruenza da nascondere" in
   un "dettaglio metodologico da rivendicare".
2. Se invece avete intenzione di rieseguire la pipeline su ImageNet-1k vero e
   proprio prima della consegna: allora **gli esempi specifici nella
   conclusione ("eyes, snouts, ...") sono probabilmente placeholder/anteprime
   da un run di sviluppo**, e andranno sostituiti con le etichette
   *effettivamente* prodotte dal run finale (che — su un vocabolario
   `imagenet` — non includerà mai "snout", perché quella parola non è nel
   vocabolario candidato di quel dataset).
3. In ogni caso: **prima della consegna, verificate che ogni esempio concreto
   citato nella relazione (parole tra virgolette, numeri, percentuali) sia
   effettivamente riproducibile dal run che descrivete** — è il singolo
   controllo di coerenza più importante che potete fare, perché è anche il
   primo che farebbe un revisore esterno.

### D.3 — 🔴 La relazione descrive risultati che non esistono ancora nel repository

Una semplice lettura in sequenza di `results.tex` rivela il problema:

```
% TODO: fill in once experiments are done. Three subsections below.
\subsection{Interpretability of Discovered Features}
% Gallery of top-K patches per feature [...]
\subsection{Feature Variation Across Layers}
% How features change from layer 4 to layer 12.       <- nota: ANCHE qui i layer sbagliati!
\subsection{Quantitative Metrics}
% [tabella con valori "--" da riempire]
```

**Eppure**, `abstract.tex` e `conclusion.tex` — che logicamente dovrebbero
*riassumere* ciò che si trova in `results.tex` — contengono già affermazioni
molto specifiche e quantitative:

- *"the SAEs recover dictionaries of **6,144** features, of which **a
  significant fraction** are successfully mapped to distinct semantic
  concepts"* (abstract)
- *"The discovered features exhibit **a consistent progression** from
  low-level visual properties in early layers to semantic, object-level
  concepts in deeper layers"* (abstract)
- *"We observed a consistent progression across depth: [...] (like **eyes,
  snouts, or background foliage**)"* (conclusione)

E ho verificato che **non esiste, in nessuna parte del repository**, alcun
artefatto reale: niente cartella `out/`, nessun CSV, nessun PNG, nessun file
con numeri concreti. L'unico posto dove compaiono numeri di esempio è la
tabella "_Example output_" nel `README.md` del progetto (righe 100-101) — ma
quei valori (`R² = 0.0379` e perfino **`R² = -0.4434` negativo**, `L0 ≈
2.380` su 6.144 — cioè quasi **il 39% delle feature attive per token, lontano
dall'essere "sparse"**) descrivono esplicitamente un *"Test Run"* con
`--subset_size 10 --epochs 1` (riga 78 del README): un run-lampo di verifica,
non un esperimento da riportare in un paper.

> 🧩 **In altre parole**: l'abstract e la conclusione sembrano scritti
> "in anticipo", presumibilmente come traccia/scaletta di cosa *ci si aspetta*
> di trovare (un'ipotesi più che ragionevole, sostenuta da Raghu et al. — vedi
> [`03_paper/06_raghu_vit_vs_cnn.md`](03_paper/06_raghu_vit_vs_cnn.md)) — ma
> al momento sono **promesse non ancora mantenute dal capitolo che dovrebbe
> dimostrarle**. Questo è esattamente il tipo di cosa che un revisore (o un
> docente in sede di esame) nota per primo: legge l'abstract, si aspetta di
> trovare la prova nei risultati, e la trova vuota.

📍 **Azione — la più urgente di tutto questo audit**: prima di ogni altra
rifinitura stilistica, **eseguite la pipeline con parametri "da esperimento
vero"** (il `README.md` suggerisce `--device cuda --subset_size 500
--epochs 5` come "Full Run", riga 85), salvate gli artefatti generati
(`layer_comparison_summary.md`, `discovered_features_summary.csv`,
`multi_feature_exemplar_grid.png`, `dose_response_curve.png`), e **scrivete
`results.tex` a partire dai numeri reali**. Solo a quel punto rileggete
`abstract.tex` e `conclusion.tex` e **allineateli ai risultati realmente
ottenuti** — anche se questi risultati dovessero risultare meno "puliti" di
quanto l'abstract promette ora (es. R² basso, alcune feature non
etichettabili in modo netto): un'analisi onesta di risultati imperfetti vale
*molto* di più, agli occhi di un valutatore XAI, di un'affermazione
trionfalistica che i dati non supportano. Trovate il template del file
`05_glossario_e_cheatsheet.md` §D pensato apposta per raccogliere questi
numeri mano a mano che li ottenete.

### D.4 — 🟡 L'abstract sovrastima la portata della valutazione CLIP

Riprendendo la frase dell'abstract: *"Across 2 monitored layers, the SAEs
recover dictionaries of 6,144 features, **of which a significant fraction**
are successfully mapped to distinct semantic concepts by the CLIP-based
filter"*.

Il modo in cui è scritta lascia intendere che **una porzione consistente
dell'intero dizionario di 6.144 feature** venga valutata e che molte di esse
risultino interpretabili. Ma, leggendo `run_pipeline.py` riga 528
(`get_top_active_features(..., num_features=10)`), emerge che la pipeline
seleziona ed etichetta con CLIP **solo le 10 feature più attive per layer** —
cioè lo **0,16% del dizionario totale** (10 / 6.144). Non è un dettaglio da
poco: la frase, così com'è scritta, implica una valutazione sistematica
dell'intero dizionario che semplicemente non avviene (e che, con 6.144
feature × 2 layer, sarebbe comunque computazionalmente proibitiva da fare per
intero — il che è perfettamente comprensibile, ma va *dichiarato*, non
lasciato intuire).

📍 **Azione**: riformulare la frase per essere precisi sulla portata reale,
ad es.: *"For each layer, we select and evaluate the $K=10$ most active
features — a representative sample rather than an exhaustive census of the
dictionary — and find that [X out of 10 / X out of 20] receive a clear,
consistent CLIP label"* (con X = il numero reale, una volta disponibile).
Una dichiarazione di scopo onesta ("abbiamo esaminato un campione
rappresentativo") è scientificamente più solida — e molto più difendibile in
sede di domande — di un'affermazione che suona più ampia di quanto sia.

### D.5 — 🟡 La definizione operativa di "monosemantico" differisce leggermente tra capitoli

`conclusion.tex` (riga 5) definisce così la monosemanticità usata nel
progetto: *"a feature is labelled monosemantic if **CLIP assigns a consistent
label** to its top activating images"*.

Ma, leggendo il codice (`interpretability.py` / `run_pipeline.py`), il
processo reale è: per ciascuna feature si recuperano i suoi $K=5$ ritagli
top-attivanti, si calcola **un'unica** etichetta aggregata (quella con la
massima similarità coseno *media* sui 5 ritagli), e questa singola etichetta
viene riportata nel CSV — non viene mai verificato (né salvato) se le 5
etichette "individuali" (una per ritaglio) **concordano** tra loro. In altre
parole, il codice produce *un'unica etichetta di consenso*, non una *misura
di accordo tra etichette*; la "consistenza" che la conclusione menziona è, di
fatto, **un giudizio qualitativo che un umano dà guardando i ritagli** (è
quello che ho descritto nella sezione causale di `methodology.tex` come
valutazione "qualitativa... durante l'analisi"), non qualcosa che CLIP
"assegna" in modo automatico e misurabile.

📍 **Azione**: la formulazione più fedele al codice — e che tra l'altro lega
perfettamente questo paragrafo alla nuova sottosezione su monosemanticità che
ho appena scritto in `methodology.tex` — sarebbe qualcosa come: *"a feature
is labelled monosemantic when visual inspection of its top-$K$ activating
crops confirms that they consistently depict the concept identified by
CLIP's aggregate label"* — sposta il soggetto della "consistenza" da "CLIP"
(che non la misura) a "noi, guardando i ritagli" (che è ciò che davvero
succede, ed è anche un'occasione per agganciare il punto "human evaluation"
mancante di cui parla §B).

### D.6 — 🟡 Il template commentato di `results.tex` cita ancora i layer vecchi

Riga 17: *"% How features change from layer **4 to layer 12**."* — stesso
refuso `{4, 8, 12}` già corretto altrove. Riga 30-32: la tabella di esempio
elenca ancora *"4 / 8 / 12"* come righe. Sono commenti, quindi non
compaiono nel PDF compilato — ma quando andrete a scrivere i risultati veri,
**sostituiteli con 6 e 11** per coerenza (e perché probabilmente userete
quello scheletro come canovaccio).

---

## E. Uso della letteratura — gli 8 paper sono rappresentati fedelmente?

Ho incrociato ciò che `related_work.tex`/`research_gap.tex` dicono di ciascun
paper con i riassunti dettagliati che abbiamo scritto in
[`03_paper/`](03_paper/) (a loro volta basati sui PDF originali, dove
disponibili). Risultato: **la rappresentazione della letteratura nella
relazione è solida e accurata** — qui sotto solo le note a margine.

| Paper | Uso nella relazione | Valutazione |
|---|---|---|
| Elhage 2021 | Framework residual stream/circuiti — base concettuale di tutto | 🟢 fedele, ben sintetizzato |
| Bricken 2023 | SAE, monosemanticità — citato insieme a Cunningham come base diretta dell'architettura | 🟢 fedele |
| Cunningham 2024 | Validazione SAE su LLM, pre-processing/centratura attivazioni | 🟢 fedele — la citazione specifica sulla centratura (`related_work.tex` riga 13 di `methodology.tex`) è precisa e pertinente |
| Conmy 2023 (ACDC) | Presentato come "circuit discovery, dimostrato su LLM, in principio architecture-agnostic ma non validato sui ViT" | 🟢 **distinzione netta e corretta** tra ciò che fa ACDC e ciò che fate voi — esattamente il framing che evita di dare l'impressione di "averlo usato" quando non è così |
| Dosovitskiy 2021 | Architettura ViT-B/16 | 🟢 fedele |
| Raghu 2021 | Motivazione del confronto multi-layer, "rappresentazioni uniformi nei ViT" | 🟢 fedele — e ora ben collegato anche alla scelta dei layer 6/11 in `methodology.tex` |
| Gandelsman 2024 (SPLICE) | Termine di paragone per il "cross-modal labelling problem": funziona perché CLIP ha già uno spazio condiviso testo-immagine | 🟢 argomentazione solida — ma ⚠️ **ricordate**: il PDF di questo paper non è scaricato localmente (vedi nota in [`03_paper/07_gandelsman_splice.md`](03_paper/07_gandelsman_splice.md)) — se possibile recuperatelo prima della discussione, in caso di domande tecniche fini |
| Haque 2026 (MedConcept) | Secondo precedente di "modello esterno come giudice/traduttore", in ambito medico | 🟢 argomentazione coerente — ⚠️ **stessa nota**: PDF non disponibile localmente, e paper troppo recente per essere stato verificato a fondo (vedi [`03_paper/08_haque_medconcept.md`](03_paper/08_haque_medconcept.md)) |

> 🟢 **Punto di forza da sottolineare**: `related_work.tex` §4 (righe 16-22)
> chiude la rassegna della letteratura con un paragrafo che **anticipa
> esplicitamente il gap** ("Both of these contributions rest on the same
> prerequisite: a shared embedding space [...] The question of how to
> identify and label interpretable features in a purely visual model [...]
> remains open, and it is exactly the challenge this project addresses") — è
> *esattamente* la struttura "a imbuto" che rende una related work persuasiva:
> non un elenco di paper, ma un argomento che converge naturalmente nel gap.
> Vale la pena rileggerlo ad alta voce prima della presentazione: è il
> paragrafo-chiave da saper riproporre quasi a memoria.

---

## F. Cose fatte BENE — un riepilogo esplicito (da portare con sicurezza in discussione)

1. **Framing del gap genuinamente originale e ben argomentato a imbuto**
   (§related_work → research_gap): non è un elenco di limitazioni, è un
   argomento che costruisce, paper dopo paper, il bisogno della vostra
   soluzione specifica.
2. **Architettura sperimentale coerente con un requisito esplicito della
   consegna**: il confronto a due profondità (layer 6 / 11) risponde
   direttamente alla richiesta P3 "Analyse how features or circuits vary
   across layers" — un argomento più forte di "lo abbiamo fatto perché Raghu
   lo suggeriva" (anche se *anche* questo è vero, e li avete citati entrambi).
3. **Pipeline tecnicamente solida e completa**: tutti e quattro gli stadi
   (estrazione attivazioni → SAE → CLIP → verifica causale) sono
   implementati, non solo abbozzati — e la simmetria ablation/steering, il
   trattamento del `[CLS]`, e il meccanismo di hook con callback sono scelte
   di design eleganti che vale la pena spiegare a fondo se richiesto.
4. **Scelta consapevole tra le due direzioni di implementazione proposte
   dalla scheda P3** (SAE vs. circuit analysis/ACDC): l'avete presa, e
   l'avete motivata con un argomento di principio (mancanza di allineamento
   testuale nei ViT puri), non per esclusione/comodità.
5. **Supporto per due backbone** (`google/vit-base-patch16-224` e
   `facebook/dinov2-base`) già presente nel codice — anche se non ancora
   sfruttato nella relazione, è un "asso nella manica" pronto all'uso, e
   risponde proprio al cenno della scheda P3 su "supervised ViT, DINO".

---

## G. Checklist delle azioni — in ordine di priorità

- [ ] 🔴 **#1 — Eseguire un run "vero" della pipeline** (parametri da
      esperimento, es. `--subset_size 500 --epochs 5` o simili — quelli che
      decidete essere adeguati al tempo/risorse a disposizione) e **salvare
      gli artefatti generati**. È il prerequisito di tutto il resto.
- [ ] 🔴 **#2 — Scrivere `results.tex`** a partire dai numeri reali, seguendo
      lo scheletro a 3 sotto-sezioni già presente (Interpretability of
      Discovered Features / Feature Variation Across Layers / Quantitative
      Metrics) — sostituendo "4/8/12" con "6/11" anche nei commenti.
- [ ] 🔴 **#3 — Allineare `abstract.tex` e `conclusion.tex` ai risultati
      reali**: sostituire gli esempi specifici ("eyes, snouts, background
      foliage", "6,144... significant fraction") con quelli effettivamente
      osservati — anche se meno "puliti" di quanto promesso ora.
- [ ] 🔴 **#4 — Decidere e dichiarare il dataset reale usato** (vedi §D.2):
      o argomentare `imagewoof` come scelta valida, o rieseguire su
      `imagenet` e aggiornare gli esempi di conseguenza. **Non lasciare la
      contraddizione attuale tra "ImageNet-1k, 1000 classi" e "snout" (parola
      che esiste solo nel vocabolario imagewoof).**
- [ ] 🟡 **#5 — Aggiungere una breve "human evaluation"** (§B): un membro
      del gruppo dà un giudizio indipendente sui ritagli, da confrontare con
      l'etichetta CLIP — chiude un requisito esplicito della scheda P3 a
      costo quasi nullo.
- [ ] 🟡 **#6 — Uniformare la terminologia "mid-network" / "early layers"**
      (§D.1): un find-and-replace di due minuti in `conclusion.tex`.
- [ ] 🟡 **#7 — Riformulare la frase sulla "significant fraction" di feature
      mappate** (§D.4) per riflettere che si valuta un campione (top-10), non
      l'intero dizionario.
- [ ] 🟡 **#8 — Rivedere la definizione di "monosemantico" in
      `conclusion.tex`** (§D.5) per attribuire la valutazione di consistenza
      all'ispezione visiva umana, non a CLIP.
- [ ] 🟡 **#9 — Aggiungere 1-2 frasi su DINO in `related_work.tex`** (§B):
      la scheda P3 lo richiede esplicitamente come "pre-training paradigm" da
      recensire, e il codice lo supporta già.
- [ ] 🟡 **#10 — Verificare con i docenti di riferimento la lunghezza attesa**
      del documento (§A.3): "2-3 pagine" vs. le ~9 pagine attuali (destinate
      a crescere). Meglio un chiarimento ora che una sorpresa il giorno della
      consegna.
- [ ] 🟢 **#11 — Opzionale ma a costo zero**: aggiungere in `methodology.tex`
      il valore reale dello scaling factor di steering (`S = 5.0`, dal
      `README.md`) per rendere la sezione completamente autosufficiente.

---

## Una nota di chiusura

Non fatevi spaventare dalla lunghezza di questo audit: **la maggior parte dei
punti elencati sono sintomi di un solo problema a monte — i risultati
sperimentali non sono ancora stati prodotti e trascritti**. Una volta
completato il punto #1 (un run vero della pipeline) e il punto #2 (scrivere
`results.tex` con i numeri reali), una buona parte degli altri punti (§D.2,
§D.3, §D.4, parzialmente §D.5) si **risolve quasi da sola**, perché smettono
di essere "promesse da verificare" e diventano "fatti da riportare". Il
lavoro concettuale e argomentativo che sta dietro alla relazione — gap,
letteratura, metodologia — è già solido; quello che manca è "chiudere il
cerchio" con i dati che lo dimostrano.
