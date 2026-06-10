# Il nostro progetto P3, spiegato per bene

> Obiettivo di questo file: che tu possa **raccontare il progetto a memoria**, seguendo
> esattamente la struttura richiesta da `XAI_00b_project_presentation.pdf`
> (Introduction → Related work → Research gap → Methodology → Results → Conclusion),
> capendo *perché* ogni pezzo c'è e come si incastra con gli altri.

Titolo del progetto: **"Mechanistic Interpretability of Vision Transformers using Sparse
Autoencoders"** — corso *Explainable and Trustworthy AI*, Polito 2025/2026, docenti
referenti Gabriele Ciravegna ed Eliana Pastor.

Tienilo a mente come **frase-riassunto da 20 secondi**, quella che useresti se qualcuno ti
fermasse per strada:

> "Abbiamo preso un Vision Transformer già allenato, e abbiamo cercato di capire **cosa
> 'pensa'** guardando dentro le sue attivazioni interne. Il problema è che i singoli
> neuroni sono confusi — rispondono a un mucchio di cose diverse insieme (sono
> *polisemantici*). Allora abbiamo allenato un modellino ausiliario, lo Sparse
> Autoencoder, che riscrive quelle attivazioni come combinazione sparsa di 'concetti'
> più puliti (*feature monosemantiche*). Poi, siccome il ViT non sa parlare (non è
> allenato con testo), abbiamo usato CLIP come 'traduttore esterno' per dare un nome a
> ognuno di questi concetti. Infine abbiamo verificato che questi concetti non siano
> solo decorativi, ma contino davvero per la decisione del modello: li abbiamo
> letteralmente *spenti* (ablation) o *amplificati* (steering) dentro la rete e
> guardato come cambia la previsione."

Tutto il resto di questo documento è lo sviluppo "ufficiale" di questa frase, sezione per
sezione, in modo che tu sappia rispondere sia a "in due parole, di cosa parla il
progetto?" sia a "perché avete scelto proprio questa metrica/questa architettura/questo
layer?".

---

## 1. Introduction — il quadro generale e perché interessa a qualcuno

### 1.1 Da dove si parte: l'opacità delle reti neurali

Le reti neurali oggi prendono decisioni in contesti delicati (diagnosi mediche, veicoli
autonomi, sistemi di credito...). Il problema è che sono delle **scatole nere**: sappiamo
*che cosa* rispondono, ma non *come* arrivano a quella risposta. Questo è un problema di
fiducia, sicurezza e allineamento (*trust, safety, alignment* — i tre concetti-chiave
citati in apertura nell'introduzione del nostro paper, vedi
[`introduction.tex`](../paper/Your_Paper_Title_Here/Chapters/introduction.tex)).

La branca dell'XAI che prova ad **aprire la scatola nera fino in fondo** — non solo
spiegare "quali pixel hanno contato" (come fa una saliency map), ma letteralmente
**reverse-engineerizzare** i meccanismi interni in algoritmi comprensibili — si chiama
**Mechanistic Interpretability** (vedi [`02_concetti/02_interpretabilita_meccanicistica.md`](02_concetti/02_interpretabilita_meccanicistica.md)
per un'immersione completa). L'idea-guida (Elhage et al. 2021) è: un transformer non è
un'unica scatola opaca, ma un insieme di componenti (attention head, blocchi MLP) che
leggono e scrivono su un canale condiviso (il *residual stream*) — e in linea di
principio si può "decompilare" quel meccanismo proprio come si decompila un binario in
codice sorgente leggibile.

### 1.2 Perché proprio i Vision Transformer (e non le CNN)?

Per anni la visione artificiale è stata dominio delle CNN. Di recente i **Vision
Transformer (ViT)** (Dosovitskiy et al. 2021 — "An Image is Worth 16x16 Words") hanno
raggiunto prestazioni comparabili o superiori, ma **lavorando in modo strutturalmente
diverso**: niente convoluzioni, niente "bias induttivi" (località, invarianza
traslazionale) incorporati nell'architettura. Tutto ciò che il modello sa sulla
struttura spaziale dell'immagine, lo ha dovuto **imparare da zero dai dati** tramite
self-attention. Raghu et al. (2021, "Do Vision Transformers See Like CNNs?") hanno
mostrato che infatti i ViT "vedono" in modo qualitativamente diverso: aggregano
informazione globale **fin dal primo layer**, e le loro rappresentazioni interne sono
molto più "uniformi" lungo la profondità rispetto a una ResNet (che invece attraversa
fasi nettamente distinte, dal locale al globale).

Questo solleva una domanda naturale e ancora poco esplorata: **come risolvono i ViT i
loro compiti visivi, internamente?** È esattamente la domanda che apre la nostra
introduzione (vedi `introduction.tex`, riga 3).

### 1.3 L'ostacolo principale: la polisemanticità

Per rispondere a quella domanda servirebbe poter "leggere" i singoli neuroni interni
come si legge un circuito elettrico componente per componente. Il problema è che i
neuroni di un transformer sono tipicamente **polisemantici**: lo stesso neurone si
attiva per concetti completamente scorrelati (l'esempio da manuale, citato anche da
Bricken et al.: un neurone di un piccolo modello linguistico che risponde insieme a
"citazioni accademiche", "dialoghi in inglese", "richieste HTTP" e "testo coreano").
L'ipotesi che spiega *perché* questo accade si chiama **superposition**: la rete deve
rappresentare più concetti di quanti neuroni possieda, e lo fa "comprimendoli" in
combinazioni sovrapposte di direzioni nello spazio delle attivazioni — un trucco che
funziona perché i concetti sono *sparsi* nei dati (raramente attivi insieme).
Per i dettagli con analogie concrete, vedi
[`02_concetti/03_sparse_autoencoder.md`](02_concetti/03_sparse_autoencoder.md) §1.

### 1.4 La soluzione proposta in letteratura: Sparse Autoencoders

Recentemente (Cunningham et al. 2024, Bricken et al. 2023) si è scoperto che si può
**districare** la superposition addestrando un piccolo modello ausiliario — uno **Sparse
Autoencoder (SAE)** — a ricostruire le attivazioni passando per un collo di bottiglia
*sovracompleto* (più "feature" candidate che neuroni originali) e *sparso* (poche feature
attive alla volta). Le direzioni che il SAE impara a usare risultano **monosemantiche**:
ciascuna corrisponde — sorprendentemente bene — a un singolo concetto coerente. Questa è
la tecnologia-chiave del nostro progetto, e la spieghiamo in dettaglio (con tanto di
equazioni e codice) in [`02_concetti/03_sparse_autoencoder.md`](02_concetti/03_sparse_autoencoder.md).

### 1.5 Il problema: questi strumenti sono nati per il testo, non per le immagini

Qui arriva il punto di svolta dell'introduzione (e l'origine dell'intero progetto):
**quasi tutto** il lavoro di mechanistic interpretability con SAE è stato fatto su
**Large Language Models**. Applicarlo a un Vision Transformer "puro" (cioè allenato solo
con etichette di classificazione, senza alcuna supervisione testuale) apre **due
problemi nuovi e non banali**, che diventano i due "research gap" del progetto (vedi
§3 più sotto):

1. **The Spatial Challenge**: i token di un LLM sono parole in sequenza, con un ordine
   naturale (causale, da sinistra a destra); i token di un ViT sono **patch 2D** di
   un'immagine, con attenzione **bidirezionale** e nessuna direzione privilegiata di
   flusso dell'informazione. Tracciare "chi influenza chi" è strutturalmente più
   complesso.
2. **Cross-Modal Evaluation**: in un LLM, una volta scoperta una feature interpretabile,
   basta guardare *quali parole* la attivano per capire cosa rappresenta — il modello
   "parla" già la stessa lingua dell'interprete umano. In un ViT puro, le feature
   scoperte sono solo **direzioni in uno spazio ad alta dimensione**: si attivano su
   certe patch d'immagine, ma il modello stesso non offre alcun "vocabolario" per
   descrivere cosa hanno in comune quelle patch. Serve un meccanismo *esterno* di
   grounding linguistico.

> **In una riga**: la nostra introduzione conclude che "in questo progetto affrontiamo
> questi due gap applicando tecniche di mechanistic interpretability — SAE — a un Vision
> Transformer pre-addestrato, per scomporre le sue rappresentazioni interne e capire le
> strutture computazionali che guidano le sue predizioni visive."

---

## 2. Related Work — la letteratura su cui ci appoggiamo

Questa sezione (vedi [`related_work.tex`](../paper/Your_Paper_Title_Here/Chapters/related_work.tex))
fa quattro cose, in quest'ordine logico — e ognuna prepara il terreno alla successiva:

### 2.1 Mechanistic Interpretability nei Transformer (fondamenta teoriche)

- **Elhage et al. 2021** ("A Mathematical Framework for Transformer Circuits") introduce
  il concetto di **residual stream** come canale di comunicazione condiviso tra i
  componenti del modello, e mostra che — almeno per i blocchi di attenzione — si possono
  "decompilare" i pesi in *circuiti* interpretabili (QK = "dove guardo", OV = "cosa
  faccio col contenuto"). **Punto cruciale per noi**: lo stesso paper ammette
  esplicitamente di non avere "presa" sulle MLP, proprio per via della polisemanticità
  dei neuroni — lasciando un problema aperto che il paper successivo risolve.
- **Conmy et al. 2023** (ACDC, "Towards Automated Circuit Discovery") **automatizza** il
  passo successivo — la scoperta di circuiti — tramite potatura iterativa del grafo
  computazionale basata su *activation patching*. È importante per noi come "vocabolario
  concettuale" (cos'è un circuito, cos'è l'activation patching) e come precedente
  metodologico per gli interventi causali, **ma non è quello che implementiamo**: noi
  non scopriamo circuiti interi, facciamo interventi mirati su singole feature già
  identificate. Approfondimento in [`03_paper/04_conmy_acdc.md`](03_paper/04_conmy_acdc.md).

### 2.2 Sparse Dictionary Learning e Monosemanticità (la "soluzione" al problema)

- **Bricken et al. 2023** ("Towards Monosemanticity") e **Cunningham et al. 2024**
  ("Sparse Autoencoders Find Highly Interpretable Features") dimostrano — su modelli
  linguistici — che addestrare un SAE sulle attivazioni MLP produce migliaia di feature
  monosemantiche, molto più interpretabili dei neuroni grezzi (misurato sia da umani sia
  da LLM-judge), e che queste feature hanno un **ruolo causale verificabile** (se le
  ablate, cambia il comportamento del modello in modo coerente con la loro
  interpretazione). **Questi due paper sono il fondamento diretto della nostra
  architettura SAE e della nostra metodologia di valutazione causale** — vedi
  [`03_paper/02_bricken_monosemanticity.md`](03_paper/02_bricken_monosemanticity.md) e
  [`03_paper/03_cunningham_sae.md`](03_paper/03_cunningham_sae.md): l'equazione di loss,
  il vincolo di norma unitaria sul decoder, la sottrazione di centratura `x - b_dec`,
  le metriche L0/R², l'ablazione causale... sono **tutte prese pari pari da qui**.

### 2.3 Rappresentazioni interne dei ViT (il "soggetto" che studiamo)

- **Dosovitskiy et al. 2021** definisce l'architettura ViT che useremo (in particolare
  la variante **ViT-B/16**: 12 layer, hidden size 768, patch 16×16, 196 patch per
  immagine 224×224 organizzate in griglia 14×14).
- **Raghu et al. 2021** mostra *empiricamente* che le rappresentazioni dei ViT cambiano
  qualitativamente con la profondità (attenzione via via più globale, ma con
  informazione spaziale **preservata** fino all'ultimo layer) — il fondamento teorico
  diretto della nostra scelta di confrontare layer 6 vs layer 11 e di poter mappare ogni
  token a un ritaglio di pixel anche in profondità. Vedi
  [`03_paper/06_raghu_vit_vs_cnn.md`](03_paper/06_raghu_vit_vs_cnn.md).

### 2.4 Concept Discovery nei modelli vision-language (il "trucco" che ci manca... e che dobbiamo aggirare)

- **Gandelsman et al. 2024** (SPLICE) decompone le rappresentazioni di **CLIP** come
  combinazioni sparse di direzioni testo-immagine — funziona perché CLIP è già allenato
  ad allineare immagini e testo, quindi ogni direzione "parla" naturalmente in linguaggio
  umano.
- **Haque et al. 2026** (MedConcept) fa concept discovery non supervisionato su VLM
  medici, e usa un LLM come "giudice esterno" per valutare quantitativamente
  l'allineamento semantico dei concetti scoperti.
- **Il filo conduttore — e il punto di svolta dell'intera related work**: *entrambi*
  questi lavori funzionano **perché esiste già uno spazio di embedding condiviso
  testo-immagine** nel modello che analizzano. Un ViT puro (allenato solo su etichette
  di classificazione, o self-supervised come DINO) **non ha questo spazio**: una feature
  scoperta è solo una direzione anonima. "La domanda di come identificare ed etichettare
  feature interpretabili in un modello puramente visivo, allenato senza alcuna
  supervisione testuale, resta aperta — ed è esattamente la sfida che questo progetto
  affronta" (`related_work.tex`, ultima riga). Approfondimenti in
  [`03_paper/07_gandelsman_splice.md`](03_paper/07_gandelsman_splice.md) e
  [`03_paper/08_haque_medconcept.md`](03_paper/08_haque_medconcept.md).

> **Come raccontarla in un colpo d'occhio**: "Sappiamo *come* aprire la scatola nera nei
> modelli linguistici (SAE). Sappiamo *come* dare un nome ai concetti scoperti, ma solo
> nei modelli che già parlano la lingua delle immagini E del testo (CLIP, VLM medici).
> Quello che manca è: come fare *entrambe* le cose — apertura della scatola nera **e**
> assegnazione di un nome comprensibile ai concetti — in un modello che vede ma non
> parla (un ViT puro)? È lì che ci infiliamo."

---

## 3. Research Gap — cosa manca, esattamente, e cosa abbiamo scelto di affrontare

Questa è la sezione di "cerniera" tra letteratura e contributo: bisogna mostrare (a)
*una panoramica dei gap* e (b) *quale gap, in particolare, attacchiamo*. Il nostro
[`research_gap.tex`](../paper/Your_Paper_Title_Here/Chapters/research_gap.tex) li
articola così:

### 3.1 Overview dei gap

1. **Assenza di analisi SAE sistematiche su modelli visivi puri.** Tutto il lavoro SAE
   esistente è su LLM: lì i token sono parole, le feature emergenti sono concetti
   linguistici, e verificare l'interpretabilità equivale a "leggere i token che attivano
   la feature". In un ViT, i token sono patch d'immagine, le attivazioni mescolano
   statistiche visive di basso e alto livello, e non c'è modo diretto di "leggere" una
   direzione sparsa appresa. **Se i SAE recuperino strutture significative anche in
   spazi di attivazione puramente visivi è quindi una domanda empirica aperta.**
2. **La struttura spaziale dei ViT introduce sfide analitiche nuove.** Nei transformer
   linguistici il flusso causale dell'informazione è relativamente facile da tracciare
   (token precedenti → successivi). In un ViT, invece, (a) l'attenzione è
   **bidirezionale** su patch 2D senza direzione privilegiata — più difficile localizzare
   "chi instrada cosa verso dove" — e (b) la predizione finale dipende **esclusivamente**
   dal token `[CLS]`, che riceve contributi da *tutti* i token tramite attenzione:
   tracciare come l'evidenza da regioni spaziali specifiche si propaghi e si comprima in
   quel singolo token richiede strumenti la cui efficacia pratica nel dominio visivo non
   è stata ancora validata.
3. **Il problema dell'etichettatura cross-modale.** Quando Gandelsman et al. e Haque et
   al. identificano concetti visivi, possono nominarli in linguaggio naturale perché il
   modello analizzato è stato allenato su dati immagine-testo accoppiati: lo spazio di
   embedding condiviso è un meccanismo "gratuito" e incorporato per tradurre direzioni
   visive in descrizioni linguistiche. **In un ViT allenato puramente su etichette di
   classificazione (o self-supervised via DINO), questo spazio non esiste**: una feature
   scoperta da un SAE è solo una direzione anonima — si attiva fortemente su certe patch,
   ma il modello stesso non offre alcun vocabolario per descrivere cosa quelle patch
   abbiano in comune. Assegnare un'etichetta richiede un **meccanismo di grounding
   esterno**. Questo "collo di bottiglia di valutazione" non viene riconosciuto nella
   maggior parte dei lavori di mechanistic interpretability sui ViT, e rende le
   affermazioni di interpretabilità nel dominio visivo più difficili da sostanziare
   rispetto al dominio linguistico.

### 3.2 Il gap che abbiamo scelto di affrontare

Le tre osservazioni precedenti convergono su **un'unica domanda di ricerca aperta**:

> *"Possono gli Sparse Autoencoder recuperare feature interpretabili e monosemantiche
> dalle attivazioni interne di un Vision Transformer puramente visivo, e come si possono
> valutare rigorosamente queste feature in assenza di una supervisione linguistica
> incorporata?"*

E qui arriva la nostra proposta concreta — il "ponte" diretto verso la sezione di
metodologia:

> Alleniamo SAE sulle attivazioni MLP-output di layer selezionati di un ViT-B/16
> pre-addestrato, e usiamo **CLIP come valutatore cross-modale ESTERNO**: date le
> immagini che attivano massimamente ciascuna feature scoperta, CLIP produce una
> descrizione linguistica facendo zero-shot matching contro un ampio vocabolario
> testuale. Questo **disaccoppia** l'analisi di interpretabilità dal modello backbone,
> evitando il bisogno di un'architettura vision-language e al tempo stesso ancorando le
> feature a concetti comprensibili dall'uomo.

I due gap concreti che affrontiamo sono dunque: **(a)** l'assenza di analisi SAE
sistematiche su modelli puramente visivi, e **(b)** la mancanza di una metodologia di
valutazione validata quando non c'è allineamento testo-immagine incorporato.

> 💡 **Trucco mnemonico**: pensa al gap come a un'equazione mancante:
> `SAE (sa aprire la scatola nera) + CLIP (sa nominare le cose) = la nostra ricetta`
> per farlo *anche* dove nessuno dei due ingredienti, da solo, basterebbe
> (SAE da solo non sa nominare i concetti visivi; CLIP da solo non apre la scatola nera
> del ViT).

---

## 4. Methodology and Implementation — cosa abbiamo costruito, passo dopo passo

Qui descriviamo *il nostro contributo concreto*: la pipeline implementata in `src/`.
Per il dettaglio riga-per-riga del codice, vedi
[`04_codice_spiegato.md`](04_codice_spiegato.md); qui ci concentriamo sul **perché**
ogni passaggio è fatto in quel modo, collegandolo alla letteratura appena vista.

L'approccio si articola in **tre stadi sequenziali** (esattamente come descritto in
apertura di [`methodology.tex`](../paper/Your_Paper_Title_Here/Chapters/methodology.tex)):
estrazione delle attivazioni → addestramento del SAE → valutazione cross-modale via CLIP
(+ valutazione causale).

### 4.1 Stadio 1 — Backbone e estrazione delle attivazioni

- **Modello**: `google/vit-base-patch16-224` — un ViT-B/16 supervisionato, pre-allenato
  su ImageNet-1k. Architettura: immagine 224×224 → 196 patch 16×16 → embedding 768-dim
  → 12 blocchi Transformer → token `[CLS]` finale → testa di classificazione lineare.
  > 🔍 **Se questa frase ti suona ancora un po' criptica**: è del tutto normale, è
  > parecchia roba condensata in una riga. Trovi ogni singolo pezzo di questa pipeline
  > — la "patchification", l'embedding lineare, il token `[CLS]`, cosa succede *dentro*
  > ai 12 blocchi Transformer (attenzione + MLP, residual stream...) — smontato,
  > spiegato con esempi numerici concreti (es. "dove sta, fisicamente nell'immagine, la
  > patch numero 83?") e collegato riga per riga al codice, in
  > [`02_concetti/01_vision_transformer.md`](02_concetti/01_vision_transformer.md).
  > Vale la pena leggerlo per intero prima di andare avanti: tutto il resto del
  > progetto — SAE, CLIP, interventi causali — presuppone di avere ben chiaro *cosa
  > sia* un'attivazione "al layer 6, posizione spaziale 83, sotto-blocco MLP".
- **Dataset**: un sottoinsieme di ImageNet-1k *validation* (non training!) — scelta
  deliberata per evitare ogni "confondimento" tra le feature scoperte dal SAE e le
  immagini specifiche usate per allenare il backbone ViT.
- **Dove "ascoltiamo"**: registriamo dei **forward hook** (vedi
  `model_loader.ActivationHook`, ispirato esplicitamente al framework di Elhage:
  "i layer leggono e scrivono sul residual stream, possiamo intercettare e modificare
  questi flussi") sull'output del **sotto-blocco MLP** ai **layer 4, 8, 12** (nel paper)
  / **6 e 11** (nell'implementazione effettiva — vedi nota sotto). Per ogni immagine,
  l'attivazione catturata ha forma `(197, 768)` — 196 patch + 1 `[CLS]`.
- **Pulizia**: scartiamo il token `[CLS]` (è un aggregatore globale, non corrisponde a
  nessuna posizione fisica — non avrebbe senso interpretarlo "spazialmente"), centriamo
  le attivazioni sottraendo la media a livello di dataset, e le salviamo su disco.


### 4.2 Stadio 2 — Addestramento dello Sparse Autoencoder

Per ciascun layer monitorato, alleniamo un SAE **separato** (questo permette di
confrontare il "carattere" delle feature scoperte a profondità diverse). L'architettura
e la loss sono **prese pari pari** da Bricken et al. / Cunningham et al.:

```
x̄ = x - b_dec                     # centratura: sottrazione del bias del decoder
f = ReLU(W_enc · x̄ + b_enc)       # encoder: proietta in uno spazio sovracompleto (m = 8·d)
x̂ = W_dec · f + b_dec             # decoder: ricostruisce l'attivazione originale
L = MSE(x, x̂) + λ · ||f||₁        # loss = fedeltà di ricostruzione + penalità di sparsità
```

con `d = 768` (dimensione attivazioni ViT) e `m = 8 · d = 6144` (fattore di espansione
F=8 — un dizionario "sovracompleto", con più feature candidate che dimensioni originali).
Le colonne del decoder sono **vincolate a norma unitaria** dopo ogni step di
ottimizzazione (per evitare che il modello aggiri la penalità L1 semplicemente
"gonfiando" la scala delle feature anziché spegnerle). I dati arrivano tramite un
**buffer streaming di token** (`TokenActivationBuffer`) che mescola le attivazioni tra
batch diversi (per evitare correlazioni intra-batch) senza esaurire la RAM.

Durante il training monitoriamo due metriche-spia:
- **R² Score** (varianza spiegata dalla ricostruzione — quanto bene il SAE "ricorda"
  l'attivazione originale)
- **L0 Norm** (numero medio di feature attive simultaneamente — la misura diretta di
  *sparsità*: più è basso, più ogni attivazione è spiegata da poche feature "pulite")

Approfondimento completo, con ogni riga d'equazione spiegata e collegata al codice, in
[`02_concetti/03_sparse_autoencoder.md`](02_concetti/03_sparse_autoencoder.md).

### 4.3 Stadio 3a — Etichettatura cross-modale via CLIP (il "ponte linguistico")

Una volta addestrato il SAE, ogni sua feature `k` corrisponde a una direzione
`d_k = W_dec[:, k]` nello spazio delle attivazioni originali. Per scoprire *cosa
rappresenta*:

1. Cerchiamo nel dataset le **K patch che attivano di più** quella feature
   (`get_top_activating_patches` — un esempio diretto di "ricerca di esemplari"
   identica nello spirito ai "top-activating dataset examples" di Bricken et al., solo
   che invece di leggere *parole* leggiamo *ritagli d'immagine*).
2. Mappiamo l'indice "piatto" della patch `spatial_idx ∈ [0, 195]` in coordinate riga/
   colonna nella griglia 14×14, e da lì in coordinate pixel — riconvertendo l'astrazione
   "token" in un ritaglio visivo concreto (16×16 pixel, o un'area contestuale più ampia).
3. Diamo questi ritagli in pasto a **CLIP** (`openai/clip-vit-base-patch32`), insieme a
   una lista di concetti candidati testuali ("fur", "eye", "wheel", "red color", ...), e
   misuriamo la **similarità coseno** tra embedding immagine e embedding testo.
   L'etichetta vincente è il concetto con la similarità media più alta sui K esemplari.

Questo è **esattamente** l'aggiramento del gap di etichettatura cross-modale descritto
sopra: CLIP non fa parte del ViT che stiamo studiando, è un **valutatore esterno e
indipendente** — esattamente come previsto dal nostro `research_gap.tex`. Una feature è
**monosemantica** se i suoi esemplari condividono un'etichetta coerente; se gli
esemplari "spaziano" su categorie scorrelate, la feature è **polisemantica** e viene
filtrata.

### 4.4 Stadio 3b — Verifica causale (ablation & steering)

Trovare una feature che "sembra" rappresentare un concetto non basta: bisogna dimostrare
che quella feature **conta davvero** per la decisione del modello — non solo
correlazionalmente, ma **causalmente**. Qui entra in gioco l'idea (di nuovo, mutuata da
Elhage/Bricken/Cunningham) dell'**intervento chirurgico** sul residual stream:

- **Ablation** — "spegniamo" la feature: sottraiamo dal residual stream esattamente la
  sua proiezione, `x_ablated = x_patches - f_j(x) · W_dec[:, j]`. Se la feature
  rappresenta davvero "occhio di cane" e la classe target è "cane", spegnerla dovrebbe
  far **calare** il logit della classe.
- **Steering** — la operiamo all'inverso: amplifichiamo artificialmente la sua attivazione,
  `x_steered = x_patches + (S - 1) · f_j(x) · W_dec[:, j]`, e osserviamo se il logit
  **sale**.
- **Relative Logit Drop** — la metrica quantitativa che riassume l'effetto:
  `(logit_baseline - logit_ablated) / |logit_baseline|`.
- **Dose-response curve** — variando con continuità l'intensità dell'ablazione (0% → 100%)
  e tracciando il logit drop risultante, otteniamo una "curva di risposta alla dose":
  se è monotona e coerente, è un'evidenza forte che la feature ha un **ruolo causale
  reale e graduale**, non un artefatto.

Tutto questo è realizzato come **un singolo forward hook con callback**
(`ActivationHook` + `intervention_callback` in `causal_eval.py`): si intercetta
l'output del sotto-blocco MLP a runtime, si isola il `[CLS]` (lasciato intatto), si
proietta il resto nello spazio del SAE, si applica la modifica chirurgica, e si
ricompone il flusso. Spiegato riga per riga in
[`04_codice_spiegato.md`](04_codice_spiegato.md) e concettualmente in
[`02_concetti/05_interventi_causali.md`](02_concetti/05_interventi_causali.md).

> **Da ricordare per la discussione**: questo NON è "circuit discovery" alla ACDC (che
> esplora e pota automaticamente l'intero grafo computazionale del modello). È un
> intervento **mirato e chirurgico** su una singola feature **già identificata e già
> etichettata**: più simile, in scala, ai case study di ablazione di Bricken/Cunningham
> che a una scoperta sistematica di circuiti. Sapere distinguere questi due livelli ti
> mette al riparo da una domanda-trabocchetto molto probabile in sede di discussione
> ("ma allora perché citate ACDC se non lo usate?").

---

## 5. Results and Analysis — cosa ci aspettiamo / cosa abbiamo trovato

> ⚠️ **Stato attuale**: al momento in cui scrivo, [`results.tex`](../paper/Your_Paper_Title_Here/Chapters/results.tex)
> contiene ancora dei TODO — gli esperimenti vanno eseguiti e i numeri/figure inseriti.
> Questa sezione del file ti aiuta a capire **cosa cercare** e **come leggerlo**, in modo
> da poter scrivere i risultati man mano che escono dalla pipeline (`uv run python
> run_pipeline.py ...`, vedi [`README.md`](../README.md)).

Gli **artefatti** che la pipeline produce (cartella `out/`) sono:

| Artefatto | Cosa mostra | Cosa guardare |
|---|---|---|
| `layer_comparison_summary.md` | Tabella R² / L0 / Mean Logit Drop per ciascun layer | Il SAE ricostruisce bene (R² alto)? È sparso (L0 basso)? Le feature contano causalmente (logit drop alto)? |
| `multi_feature_exemplar_grid*.png` | Griglia 5×5: per 5 feature rappresentative, i 5 ritagli che le attivano di più, l'immagine intera con la patch evidenziata, e una heatmap di attivazione spaziale | Le feature sono *visivamente coerenti*? L'etichetta CLIP "ha senso" guardando i ritagli? |
| `discovered_features_summary.csv` | Tabella dettagliata per ogni feature: layer, etichetta CLIP, confidenza, logit baseline/ablato, drop relativo, incremento da steering | I dati grezzi per costruire grafici/statistiche di sintesi |
| `dose_response_curve.png` | Logit drop relativo in funzione dell'intensità di ablazione (0–100%) | La curva è monotona? È "graduale" (proporzionale alla forza dell'intervento) o a soglia? |
| `sae_training_curves.png` | Convergenza dell'addestramento del SAE (loss, R², L0) per ciascun layer | Il training è stabile? Quale layer converge meglio/più in fretta? |
| `feature_activation_heatmap.png` | Heatmap spaziale 14×14 della feature più causalmente rilevante sul layer 11, sovrapposta all'immagine | Dove "guarda" la feature più importante, fisicamente, dentro l'immagine? |

### 5.1 Interpretabilità delle feature scoperte

Cosa cercare: per ciascun layer, prendi 6-8 feature rappresentative (le più attive, o
quelle con il logit drop più alto) e per ognuna documenta: l'etichetta CLIP assegnata,
il punteggio di confidenza (similarità coseno media), e una breve descrizione visiva
basata sui ritagli (es. "Feature 4049: si attiva su superfici rosse uniformi — etichetta
CLIP 'red color', score 0.31"). Segna anche se la classifichi come monosemantica o
polisemantica, e perché.

### 5.2 Variazione delle feature lungo la profondità (layer 6 vs layer 11)

Questa è probabilmente la parte più interessante da raccontare, perché collega
direttamente alla letteratura (Raghu et al.): **ti aspetti** che le feature del layer 6
(metà rete) catturino proprietà visive di basso livello — texture, colori, pattern
locali ("fur", "scale pattern", "metal texture", "red color"...) — mentre quelle del
layer 11 (tardo) catturino concetti più semantici/a livello di oggetto ("eye", "snout",
"background foliage"...). Se i tuoi risultati confermano questo gradiente, è una
**bellissima conferma indipendente** — ottenuta con uno strumento diverso (SAE +
CLIP-labeling) — di un fenomeno già osservato con tecniche diverse (CKA, attention
distance) da Raghu et al. Se *non* lo confermano, è altrettanto interessante da
discutere (magari il modello supervisionato su ImageNet-1k, più piccolo di JFT-300M,
sviluppa rappresentazioni intermedie meno mature — vedi
[`03_paper/06_raghu_vit_vs_cnn.md`](03_paper/06_raghu_vit_vs_cnn.md) §4 punto 6).

### 5.3 Metriche quantitative

Riempi la tabella `layer_comparison_summary` con i numeri reali e commentali: un R²
basso/negativo segnala che il SAE fatica a ricostruire le attivazioni a quel layer
(magari serve più training, o un λ diverso); un L0 alto significa che le feature non
sono "abbastanza sparse" da essere ben isolate; un logit drop medio alto indica che,
in media, le feature scoperte sono causalmente rilevanti per la predizione.

> 📌 Quando avrai i numeri reali, aggiorna anche [`05_glossario_e_cheatsheet.md`](05_glossario_e_cheatsheet.md)
> (sezione "I nostri numeri") così li hai sempre a portata di mano per la discussione.

---

## 6. Conclusion — il messaggio che vogliamo lasciare

La conclusione (vedi [`conclusion.tex`](../paper/Your_Paper_Title_Here/Chapters/conclusion.tex))
chiude il cerchio aperto nell'introduzione, in tre mosse:

1. **Cosa abbiamo dimostrato**: che gli strumenti di mechanistic interpretability nati
   per il linguaggio (SAE) *si trasferiscono* — con gli adattamenti giusti — a un
   Vision Transformer puramente visivo: si possono recuperare feature sparse e
   interpretabili dalle sue rappresentazioni, e CLIP funziona bene come "traduttore"
   esterno per nominarle senza richiedere un'architettura vision-language. Si osserva un
   gradiente di astrazione coerente con la profondità (texture/colore in early layer →
   concetti a livello di oggetto in late layer) — analogo a quanto noto nelle CNN, ma
   ora caratterizzato a livello di **singole direzioni sparse nel residual stream** di
   un transformer.
2. **I limiti, con onestà intellettuale**: il vocabolario CLIP è finito e sbilanciato
   verso concetti stile-ImageNet (rischio che feature semanticamente distinte ricevano
   etichette simili); l'analisi è limitata agli output MLP (non copre residual stream o
   head di attenzione — altri "siti" di calcolo); la nozione di "monosemanticità" usata
   è operativa/proxy (CLIP assegna un'etichetta coerente), non una prova formale, e
   beneficerebbe di validazione umana su scala più ampia.
3. **Le naturali estensioni future**: applicare *circuit analysis* tramite activation
   patching (qui si ricollega ad ACDC — non lo abbiamo fatto, ma è il passo logico
   successivo una volta che si dispone di un dizionario di feature interpretabili e
   causalmente verificate) per tracciare *come* le feature contribuiscono insieme a una
   decisione; e confrontare i dizionari SAE di un ViT supervisionato vs. uno
   self-supervised DINO-pretrained — due regimi di training noti per produrre
   rappresentazioni interne qualitativamente diverse (Raghu et al.).

> 💬 **Una riga ad effetto da tenere in tasca per chiudere la presentazione**: "Abbiamo
> mostrato che si può aprire la scatola nera di un modello che *vede* ma non *parla* —
> usando un secondo modello che invece sa fare entrambe le cose come ponte linguistico
> esterno, senza dover toccare né riallenare il modello originale."

---

## 7. Domande-trabocchetto plausibili (e come rispondere)

| Possibile domanda | Come rispondere in breve |
|---|---|
| "Perché non avete usato direttamente CLIP come backbone, visto che già allinea testo e immagini?" | Perché il punto del progetto è proprio dimostrare che si può interpretare un modello *che non ha* questo allineamento — è il gap che colmiamo. Usare CLIP come backbone avrebbe reso il problema "facile" come in Gandelsman et al., non avrebbe richiesto alcuna soluzione cross-modale nuova. |
| "Come fate a essere sicuri che l'etichetta CLIP sia 'corretta' e non un artefatto del vocabolario che avete scelto?" | Non ne siamo sicuri al 100% — è un limite dichiarato (vedi conclusione): la nozione di monosemanticità è "operativa" (proxy), il vocabolario è finito; serve validazione umana su scala. Lo controlliamo guardando se gli esemplari (i ritagli) sono visivamente coerenti tra loro, non solo se CLIP dà un punteggio alto. |
| "Perché ablare/scalare proprio nello spazio del SAE e non nel residual stream grezzo?" | Perché vogliamo isolare l'effetto di *una singola feature monosemantica* — un concetto ben definito — non di una direzione qualunque del residual stream (che sarebbe quasi certamente polisemantica). Sottraendo `f_j(x) · W_dec[:, j]` rimuoviamo "esattamente quella quantità di quel concetto", bypassando il rumore di ricostruzione. |
| "Ma allora fate circuit discovery come ACDC?" | No — ACDC scopre automaticamente sottoreti dell'intero modello pottando migliaia di connessioni; noi facciamo interventi mirati e chirurgici su singole feature già identificate ed etichettate. È un livello di analisi più semplice ma più mirato — e un prerequisito concettuale per un'eventuale applicazione futura di ACDC al "grafo delle feature". |
| "Perché layer 6 e 11 e non altri?" | Rappresentano rispettivamente metà rete e quasi-fine rete (su 12 layer totali) — la dicotomia "mid vs late network" che, secondo Raghu et al., corrisponde a un cambio qualitativo nel tipo di informazione integrata (più locale/mista a metà, più globale/semantica verso la fine). |
| "Avete confrontato anche con DINOv2?" | Il codice supporta entrambi i backbone (`model_loader.py` gestisce sia `ViTForImageClassification` sia `Dinov2Model`), ma il confronto sistematico supervisionato-vs-self-supervised è indicato come *direzione futura* nella conclusione, non come parte del contributo attuale. |
