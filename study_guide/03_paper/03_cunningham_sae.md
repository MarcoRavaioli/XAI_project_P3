# Paper 3 — Cunningham et al. 2024, "Sparse Autoencoders Find Highly Interpretable Features in Language Models"

**Citazione**: Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L. (2024).
*Sparse Autoencoders Find Highly Interpretable Features in Language Models*. ICLR 2024.
(`Cunningham2024` in [`references.bib`](../../paper/Your_Paper_Title_Here/references.bib))

> Ruolo nel nostro progetto: insieme a Bricken et al. 2023, è il **secondo pilastro**
> della nostra metodologia SAE — ma con un ruolo complementare e ben distinto: dove
> Bricken et al. *propone e illustra* l'idea su un caso minimale, Cunningham et al. la
> **valida rigorosamente su scala più ampia**, con metriche quantitative e confronti
> sistematici. È il paper che ci dà "il permesso scientifico" di fidarci del metodo.

## 1. Il problema che affronta

Bricken et al. avevano mostrato che i SAE *sembrano* funzionare su un modello
giocattolo. Ma una dimostrazione su un singolo modello piccolo lascia aperte domande
serie: **funziona anche su modelli più grandi e realistici? Le feature scoperte sono
*davvero* più interpretabili dei neuroni grezzi (e non solo "diverse"), in modo
misurabile e non solo aneddotico? E hanno davvero un ruolo causale verificabile?**
Cunningham et al. affrontano queste domande con un disegno sperimentale molto più
rigoroso, su modelli linguistici di dimensioni "vere" (Pythia).

## 2. L'idea centrale — validare, non solo proporre

Il contributo distintivo di questo paper non è un'idea architetturale nuova (l'SAE è
sostanzialmente lo stesso di Bricken et al.), ma una **metodologia di valutazione
sistematica e quantitativa**:

- **Confronto diretto, misurato, tra feature SAE e neuroni grezzi**: non basta dire
  "le feature sembrano più pulite a occhio" — il paper costruisce metriche e protocolli
  (incluso l'uso di **giudici LLM** per valutare automaticamente, su larga scala,
  quanto coerentemente una feature/un neurone si attiva su un determinato concetto) per
  *misurare* l'interpretabilità in modo riproducibile.
- **Verifica causale sistematica**: non solo case study isolati, ma una valutazione su
  larga scala di quanto le feature scoperte abbiano un effetto causale prevedibile e
  coerente con la loro interpretazione, tramite interventi di ablazione.

> 🔑 Il messaggio di fondo: *"non fidatevi solo dell'intuizione — misurate
> l'interpretabilità con metriche riproducibili, e verificate la causalità con
> esperimenti controllati."* È esattamente lo spirito che cerchiamo di portare nel
> nostro progetto con le metriche R²/L0 (qualità della decomposizione SAE) e Relative
> Logit Drop / dose-response curve (rilevanza causale delle feature).

## 3. I risultati chiave — e cosa significano per noi

- **Le feature SAE sono sistematicamente più interpretabili dei neuroni grezzi**, sia
  secondo valutatori umani sia secondo giudici automatici (LLM). Questo è il risultato
  "headline" del paper: conferma su scala che il problema della polisemanticità non è
  un ostacolo insormontabile — può essere aggirato con questo approccio.
- **Le feature catturano concetti più fini e specifici** di quanto facciano i neuroni —
  spesso concetti che sarebbe stato impossibile isolare guardando le attivazioni
  grezze, perché "nascosti" nella sovrapposizione.
- **Il ruolo causale delle feature è verificabile**: ablare una feature interpretata
  come rappresentante un certo concetto produce, in modo sistematico, effetti coerenti
  con quell'interpretazione sul comportamento successivo del modello — non solo in
  alcuni casi isolati, ma come tendenza generale, misurabile.
- **Robustezza su scala**: il metodo regge — con gli opportuni adattamenti di
  iperparametri — anche su modelli linguistici di dimensioni significativamente
  maggiori rispetto al modello giocattolo di Bricken et al., un segnale incoraggiante
  sulla generalità della tecnica.

## 4. Perché questo paper conta — direttamente — per il nostro progetto

Questo paper ci fornisce **due cose preziose**:

1. **La fiducia metodologica** che vale la pena tentare: se il metodo SAE regge la
   prova di una valutazione rigorosa su modelli linguistici "veri" (non solo
   giocattolo), è ragionevole — anche se non garantito — aspettarsi che possa
   funzionare anche nel dominio visivo. Questo è precisamente il "salto di fiducia"
   che il nostro progetto compie, ed è esplicitamente menzionato come motivazione nel
   nostro `research_gap.tex`.
2. **Il modello di valutazione causale** che adattiamo: la nostra fase
   `causal_eval.py` (Relative Logit Drop, dose-response curve) è concettualmente
   un adattamento — al dominio visivo e a una singola feature alla volta — della
   stessa logica di "verifica causale tramite ablazione sistematica" proposta qui.
   Anche il file [`sae.py`](../../src/sae.py) cita esplicitamente questo paper nel suo
   docstring, accanto a Bricken et al.

## 5. Tre frasi/idee da avere pronte per la discussione

1. *"Mentre Bricken et al. propone e illustra l'idea dei SAE su un modello minimale,
   Cunningham et al. la sottopone a una valutazione molto più rigorosa e quantitativa
   — su modelli linguistici di scala realistica — confermando sia l'interpretabilità
   superiore delle feature scoperte sia, soprattutto, la loro **rilevanza causale
   verificabile**."*
2. *"È il paper che ci dà fiducia metodologica nel tentare il trasferimento al dominio
   visivo: se il metodo regge una valutazione rigorosa su LLM di scala reale, è
   ragionevole provarlo — pur sapendo che emergono problemi nuovi — su un Vision
   Transformer."*
3. *"Le nostre metriche di valutazione causale (Relative Logit Drop, dose-response
   curve) sono un adattamento al dominio visivo della stessa filosofia di valutazione
   sistematica proposta qui — non interventi aneddotici, ma misure riproducibili
   dell'effetto causale di ciascuna feature."*
