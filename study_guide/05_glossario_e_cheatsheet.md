# Glossario unificato + Cheat Sheet — il "bigliettino" da consultare al volo

> Pensato per essere il file che riapri 10 minuti prima della presentazione/discussione:
> tutte le definizioni, formule e numeri-chiave in un posto solo, senza dover
> ricostruire il filo del discorso. Per le spiegazioni complete, ogni voce rimanda al
> file di approfondimento corrispondente.

## A. Glossario alfabetico (con rimando all'approfondimento)

| Termine | Definizione in una riga | Approfondisci in |
|---|---|---|
| **`[CLS]` token** | Token speciale appreso, in testa alla sequenza, che aggrega informazione globale e da solo guida la classificazione finale | [`02_concetti/01`](02_concetti/01_vision_transformer.md) §1.1, §2.3 |
| **ACDC / Activation Patching** | Algoritmo di scoperta automatica di circuiti tramite potatura iterativa di un grafo computazionale, basata su "trapianti" di attivazioni tra run | [`03_paper/04`](03_paper/04_conmy_acdc.md) |
| **Ablation (ablazione)** | Rimozione chirurgica del contributo di una feature dal residual stream: `x − α·f_j·W_dec_j` | [`02_concetti/05`](02_concetti/05_interventi_causali.md) §2.1 |
| **CKA (Centered Kernel Alignment)** | Metrica per confrontare quanto si somigliano due rappresentazioni interne di reti (anche con architetture diverse) | [`03_paper/06`](03_paper/06_raghu_vit_vs_cnn.md) |
| **CLIP** | Modello vision-language allenato a proiettare immagini e testo in uno spazio di embedding condiviso, usato qui come "traduttore esterno" | [`02_concetti/04`](02_concetti/04_clip_e_valutazione_crossmodale.md) §2 |
| **Dose-response curve** | Grafico dell'effetto causale (logit drop) in funzione dell'intensità dell'intervento — una curva monotona è la prova causale più forte | [`02_concetti/05`](02_concetti/05_interventi_causali.md) §4 |
| **Encoder/Decoder del SAE** | `f = ReLU(W_enc·(x−b_dec)+b_enc)` / `x̂ = W_dec·f + b_dec` — proiezione in spazio sovracompleto e ricostruzione | [`02_concetti/03`](02_concetti/03_sparse_autoencoder.md) §2.2-2.3 |
| **Feature monosemantica** | Una direzione appresa che corrisponde — in modo affidabile e coerente — a un solo concetto interpretabile | [`02_concetti/02`](02_concetti/02_interpretabilita_meccanicistica.md) §2.3 |
| **Hook (forward hook)** | Meccanismo PyTorch che intercetta l'output di un modulo durante il forward pass — usato sia per catturare sia per modificare attivazioni | [`04_codice_spiegato`](04_codice_spiegato.md) §1.2 |
| **L0 norm** | Numero medio di feature attive simultaneamente per token — misura diretta della sparsità raggiunta | [`02_concetti/03`](02_concetti/03_sparse_autoencoder.md) §3.2 |
| **L1 sparsity penalty** | Termine di loss `λ·‖f‖₁` che penalizza la somma dei valori assoluti delle attivazioni, incoraggiando la sparsità | [`02_concetti/03`](02_concetti/03_sparse_autoencoder.md) §2.5 |
| **Mechanistic Interpretability** | Reverse-engineering degli algoritmi interni di una rete neurale, fino a un livello "leggibile come codice sorgente" | [`02_concetti/02`](02_concetti/02_interpretabilita_meccanicistica.md) §1 |
| **Monosemanticità / Polisemanticità** | Proprietà di una direzione/neurone di rappresentare uno / molti concetti scollegati | [`02_concetti/02`](02_concetti/02_interpretabilita_meccanicistica.md) §2.3 |
| **MLP sub-block** | Trasformazione non-lineare per-token (espande e ricomprime `d_model`); il sito che monitoriamo con i nostri hook | [`02_concetti/01`](02_concetti/01_vision_transformer.md) §2.2 |
| **Multi-Head Self-Attention** | Meccanismo che fa scambiare informazione tra le patch, pesata per rilevanza reciproca, in più "canali" paralleli | [`02_concetti/01`](02_concetti/01_vision_transformer.md) §2.1 |
| **Patch embedding** | Proiezione lineare che trasforma una patch 16×16×3 in un vettore 768-dimensionale ("token") | [`02_concetti/01`](02_concetti/01_vision_transformer.md) §1.1 |
| **Prompt template ("a photo of...")** | Tecnica che incornicia un concetto in una frase naturale per allineare l'input alla distribuzione di training di CLIP | [`02_concetti/04`](02_concetti/04_clip_e_valutazione_crossmodale.md) §3.3 |
| **R² Score** | Varianza spiegata dalla ricostruzione del SAE — misura la fedeltà della decomposizione | [`02_concetti/03`](02_concetti/03_sparse_autoencoder.md) §3.2 |
| **Relative Logit Drop (RLD)** | `(L_baseline − L_ablated) / |L_baseline|` — quanto crolla, in proporzione, il logit della classe target dopo ablazione | [`02_concetti/05`](02_concetti/05_interventi_causali.md) §3 |
| **Residual stream** | Il "canale condiviso" a cui ogni componente del modello legge e scrive, sommando il proprio contributo | [`02_concetti/02`](02_concetti/02_interpretabilita_meccanicistica.md) §2.1 |
| **Ritaglio contestuale** | Porzione di immagine più ampia della singola patch, centrata su di essa, usata per dare contesto a CLIP | [`02_concetti/04`](02_concetti/04_clip_e_valutazione_crossmodale.md) §3.2 |
| **SAE (Sparse Autoencoder)** | Modello ausiliario che riscrive le attivazioni come combinazione sparsa di direzioni in uno spazio sovracompleto | [`02_concetti/03`](02_concetti/03_sparse_autoencoder.md) |
| **Similarità coseno** | Misura di quanto due vettori "puntano nella stessa direzione" — prodotto scalare di vettori normalizzati a norma 1 | [`02_concetti/04`](02_concetti/04_clip_e_valutazione_crossmodale.md) §3.3 |
| **Spazio sovracompleto** | Spazio di rappresentazione con più dimensioni (`m = 8·d`) di quante ne abbia l'originale — dà "spazio" a ogni concetto | [`02_concetti/03`](02_concetti/03_sparse_autoencoder.md) §1 |
| **Steering** | Amplificazione artificiale dell'attivazione di una feature: `x + (S−1)·f_j·W_dec_j` | [`02_concetti/05`](02_concetti/05_interventi_causali.md) §2.2 |
| **Superposition** | Strategia con cui una rete "comprime" più concetti di quanti neuroni possieda, sfruttandone la scarsità | [`02_concetti/02`](02_concetti/02_interpretabilita_meccanicistica.md) §2.3 |
| **Top-activating exemplars** | Le K patch/esempi che attivano maggiormente una feature — il metodo principale per interpretarla | [`02_concetti/04`](02_concetti/04_clip_e_valutazione_crossmodale.md) §3.1 |
| **Vincolo di norma unitaria** | Normalizzazione delle colonne del decoder a norma 1, per impedire alla L1 di essere aggirata "gonfiando" i pesi | [`02_concetti/03`](02_concetti/03_sparse_autoencoder.md) §2.6 |
| **ViT-B/16** | Vision Transformer Base, patch 16×16 — il backbone analizzato (`google/vit-base-patch16-224`) | [`02_concetti/01`](02_concetti/01_vision_transformer.md) §3 |

## B. Le equazioni-chiave, tutte insieme

```
SAE encode:           f(x) = ReLU( W_enc · (x − b_dec) + b_enc )
SAE decode:           x̂    = W_dec · f + b_dec
SAE loss:             L    = MSE(x, x̂) + λ · ‖f‖₁
Vincolo decoder:      ‖W_dec[:, j]‖₂ = 1   ∀j

Ablation:             x_ablated = x_patches − α · f_j(x) · W_dec[:, j]      (α = 1 − scaling_factor)
Steering:             x_steered = x_patches + (S − 1) · f_j(x) · W_dec[:, j] (S = scaling_factor)
Relative Logit Drop:  RLD = (L_baseline − L_ablated) / |L_baseline|

Coordinate patch:     row = (spatial_idx // grid_size) · patch_size
                      col = (spatial_idx %  grid_size) · patch_size
```

## C. La "scheda anagrafica" del modello e della pipeline

| Parametro | Valore |
|---|---|
| Backbone | `google/vit-base-patch16-224` (ViT-B/16) |
| `d_model` (dimensione attivazioni) | 768 |
| Patch / griglia | 16×16 px → griglia 14×14 = 196 patch |
| Lunghezza sequenza | 197 (196 patch + 1 `[CLS]`) |
| Layer totali | 12 (indici 0-based 0..11) |
| Layer monitorati | **6 e 11** (indici 0-based: 5 e 10) |
| Sotto-modulo agganciato | output del blocco **MLP** |
| Espansione SAE | `expansion_factor = 8` → `hidden_dim = 6144` |
| Loss SAE | `MSE + λ·L1`, vincolo norma unitaria sul decoder |
| Modello CLIP | `openai/clip-vit-base-patch32` |
| Top-K esemplari per etichettatura | K = 5 |
| Context patches per ritaglio contestuale | 2 (→ ritaglio fino a 80×80 px) |

## D. I "nostri numeri" — da riempire dopo aver eseguito la pipeline 📌

> Esegui `uv run python run_pipeline.py [...]` (vedi [`README.md`](../README.md) per i
> parametri), poi riporta qui i valori reali letti da `out/layer_comparison_summary.md`
> e `out/discovered_features_summary.csv`. Avere questi numeri a memoria (anche
> approssimati) ti renderà molto più sicuro/a in sede di discussione.

| Layer | R² Score | L0 Norm | Mean Logit Drop | Note / osservazioni |
|---|---|---|---|---|
| Layer 6  | _da compilare_ | _da compilare_ | _da compilare_ | |
| Layer 11 | _da compilare_ | _da compilare_ | _da compilare_ | |

Feature più interessanti da ricordare (3-5, scelte tra quelle con drop/score più alti):

| Feature idx | Layer | Etichetta CLIP | Score | Logit drop | Cosa mostrano i ritagli |
|---|---|---|---|---|---|
| _____ | _____ | _____ | _____ | _____ | _____ |
| _____ | _____ | _____ | _____ | _____ | _____ |
| _____ | _____ | _____ | _____ | _____ | _____ |

## E. Mappa "chi cita chi" — per non confondersi mai più tra i papers

```
Elhage 2021 ("framework matematico")
   │  fornisce: residual stream, circuiti, polisemanticità — ma si ferma davanti alle MLP
   ▼
Bricken 2023 ("Towards Monosemanticity")  ──┐
   │  propone: SAE su un modello giocattolo   │  ENTRAMBI = fondamento diretto
   ▼                                          │  della nostra architettura SAE
Cunningham 2024 ("SAE find interpretable")  ─┘  e della valutazione causale
   │  valida: SAE su LLM di scala reale, con metriche rigorose
   │
   ├──────────────────────────────────────────────────┐
   │                                                    │
   ▼ (soggetto da studiare)                            ▼ (come SI etichetta in dominio
Dosovitskiy 2021 ("ViT")                                  visivo, quando il modello "parla")
   │  definisce l'architettura ViT-B/16             Gandelsman 2024 (SPLICE, su CLIP)
   ▼                                                Haque 2026 (MedConcept, su VLM medici)
Raghu 2021 ("Do ViT see like CNN?")                      │
   │  motiva il confronto layer 6 vs 11                  │  PROBLEMA: i loro metodi richiedono
   ▼                                                      │  un modello che "parla" già —
Conmy 2023 (ACDC — citato, non implementato)              │  il nostro ViT puro non lo fa!
   "prossimo passo naturale" per circuiti                 ▼
                                              ════> IL NOSTRO GAP: SAE + CLIP esterno <════
```

## F. Le 5 domande che, se sai rispondere bene, coprono il 90% della discussione

1. **"Cos'è la polisemanticità e perché è un problema?"** → Un neurone risponde a
   concetti scollegati perché la rete deve comprimere più concetti di quanti neuroni
   possieda (superposition); questo rende i neuroni grezzi illeggibili uno per uno.
   ([`02_concetti/02`](02_concetti/02_interpretabilita_meccanicistica.md) §2.3)
2. **"Come funziona un SAE, e perché funziona?"** → Proietta in uno spazio
   sovracompleto e sparso (encoder+ReLU), ricostruisce (decoder), bilancia fedeltà e
   sparsità (loss MSE+L1); funziona perché sfrutta la stessa scarsità che la rete
   originale sfruttava per comprimere. ([`02_concetti/03`](02_concetti/03_sparse_autoencoder.md))
3. **"Perché serve CLIP, e perché non basta il SAE da solo?"** → Il SAE trova
   *direzioni*, non *nomi*; CLIP è un traduttore esterno indipendente che sa
   "parlare" sia immagini sia testo — un ponte che il ViT puro non possiede.
   ([`02_concetti/04`](02_concetti/04_clip_e_valutazione_crossmodale.md) §1-2)
4. **"Come dimostrate che le feature non sono solo 'decorative'?"** → Interventi
   causali (ablation/steering) e dose-response curve — non basta la correlazione
   ("si attiva su X"), serve la prova che manipolarla cambia l'output in modo
   prevedibile e graduale. ([`02_concetti/05`](02_concetti/05_interventi_causali.md))
5. **"Qual è il vostro contributo rispetto alla letteratura esistente?"** → Applicare
   una tecnica nata per il testo (SAE) a un dominio (ViT puro) dove emergono due
   problemi nuovi mai affrontati insieme: struttura spaziale 2D bidirezionale, e
   assenza di un "linguaggio" nativo per nominare i concetti scoperti — risolto
   tramite CLIP come valutatore esterno disaccoppiato dal modello analizzato.
   ([`01_il_progetto_spiegato.md`](01_il_progetto_spiegato.md) §3-4)
