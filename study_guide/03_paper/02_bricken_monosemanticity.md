# Paper 2 — Bricken et al. 2023, "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"

**Citazione**: Bricken, T., Templeton, A., Batson, J., et al. (2023). *Towards
Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic /
Transformer Circuits Thread.
(`Bricken2023` in [`references.bib`](../../paper/Your_Paper_Title_Here/references.bib))

> Ruolo nel nostro progetto: è — insieme a Cunningham et al. 2024 — **il fondamento
> diretto della nostra architettura SAE**. Quasi ogni scelta in [`sae.py`](../../src/sae.py)
> (centratura `x - b_dec`, vincolo di norma unitaria sul decoder, loss MSE+L1...) viene
> da qui, ed è citata esplicitamente nei docstring del codice.

## 1. Il problema che affronta

Riprendendo esattamente da dove si era fermato Elhage et al. 2021: i singoli neuroni
MLP di un modello linguistico sono polisemantici e quindi illeggibili uno per uno. Il
paper si chiede: **possiamo trovare, all'interno di quello stesso spazio di
attivazioni, un insieme *diverso* di direzioni — non i neuroni grezzi, ma combinazioni
apprese di essi — che siano invece monosemantiche?** E se sì, *come* le troviamo?

## 2. L'idea centrale — "decomprimere" sfruttando la scarsità

Il paper formalizza l'ipotesi della **superposition**: una rete neurale, avendo meno
neuroni che concetti da rappresentare, "comprime" più concetti in sovrapposizione nello
stesso piccolo spazio — un trucco che funziona perché, in pratica, i concetti sono
*sparsi* (raramente attivi insieme in uno stesso input). La proposta è allora:
costruiamo un modello ausiliario — uno **Sparse Autoencoder** — che proietta quelle
attivazioni compresse in uno spazio **molto più ampio** (sovracompleto), forzando però
**solo poche direzioni alla volta a essere attive**. Se l'ipotesi di superposition è
corretta, questo "spazio extra, usato con parsimonia" dovrebbe permettere a ciascun
concetto di "scrollarsi di dosso" la sovrapposizione e ottenere una direzione propria,
pulita — una **feature monosemantica**.

> 🧩 Riprendendo l'analogia già vista in [`02_concetti/03_sparse_autoencoder.md`](../02_concetti/03_sparse_autoencoder.md):
> è come passare da un cassetto condiviso fra mille vestiti a mille piccoli scomparti,
> con la regola "ne usi solo 2-3 alla volta" — improvvisamente ogni capo può avere il
> suo posto pulito e identificabile.

## 3. I risultati tecnici chiave — e quanto sono "trapiantati" nel nostro codice

Il paper non si limita a proporre l'idea: la mette in pratica su un piccolo modello
linguistico (un transformer a un solo layer) e descrive in dettaglio l'architettura, la
procedura di addestramento e — soprattutto — un ricco repertorio di **scoperte
empiriche** sul comportamento delle feature risultanti. Ecco i punti che ritroverai,
**identici**, in [`sae.py`](../../src/sae.py) e [`caching_and_training.py`](../../src/caching_and_training.py):

- **Architettura encoder-ReLU-decoder con dizionario sovracompleto**: esattamente la
  struttura `encode`/`decode` di `SparseAutoencoder`.
- **Centratura delle attivazioni sottraendo il bias del decoder** prima della
  proiezione (`x - b_dec`): è una raccomandazione esplicita del paper, e il commento
  nel codice la cita testualmente: *"centering activation by subtracting decoder bias,
  as described in Bricken et al. [2023]"* ([`sae.py:63`](../../src/sae.py#L63)).
- **Vincolo di norma unitaria sulle colonne del decoder**, riapplicato dopo ogni step
  di ottimizzazione — per impedire alla rete di "barare" abbassando artificialmente la
  penalità L1 gonfiando i pesi del decoder e rimpicciolendo le attivazioni.
- **Top-activating dataset examples come metodo di interpretazione**: per capire cosa
  rappresenta una feature, il paper guarda *quali esempi del dataset la attivano di
  più* — è esattamente la logica di `get_top_activating_patches`, solo che lì gli
  "esempi" sono frammenti di testo, qui sono ritagli di immagine.
- **Verifica causale tramite ablazione**: il paper conferma le interpretazioni
  trovate manipolando attivamente le feature scoperte e osservando l'effetto sul
  comportamento del modello — l'ispirazione diretta della nostra fase
  `causal_eval.perform_causal_intervention`.

## 4. Le scoperte più sorprendenti (utili da citare a voce)

- Le feature scoperte spesso **non corrispondono a singole parole**, ma a concetti
  semantici sfumati e specifici — alcune rappresentano relazioni sintattiche, altre
  registri linguistici, altre ancora concetti molto astratti (es. "testo che esprime
  esitazione", "riferimenti a basi militari"). Questo livello di granularità sarebbe
  **invisibile** guardando i neuroni grezzi.
- Le feature mostrano **proprietà "finite-state" interessanti**: spesso si attivano in
  modo molto specifico al contesto, e la loro attivazione può essere "spiegata" con
  notevole precisione dal contenuto locale del testo — un'evidenza forte che siano
  davvero rappresentazioni di concetti, non rumore statistico.
- L'aumentare la dimensione del dizionario (più feature candidate) tende a produrre
  feature **più fini e specifiche** — un fenomeno che il paper chiama "splitting": una
  feature ampia in un dizionario piccolo si "divide" in più feature specializzate in
  un dizionario più grande. Un parallelo diretto con la nostra scelta di un
  `expansion_factor = 8`: più ampio è il dizionario, più probabile è ottenere feature
  fini e monosemantiche — ma anche più costoso l'addestramento.

## 5. Perché questo paper conta — direttamente — per il nostro progetto

È, in tutta franchezza, **il "manuale di istruzioni"** da cui abbiamo copiato
l'architettura del SAE. Ogni volta che in [`02_concetti/03_sparse_autoencoder.md`](../02_concetti/03_sparse_autoencoder.md)
trovi una scelta progettuale spiegata con un "perché si fa così", la risposta — quasi
sempre — è "perché lo raccomanda Bricken et al.". Il nostro contributo non sta
nell'inventare una nuova architettura SAE, ma nel **trasportarla in un dominio nuovo**
(immagini anziché testo) e affrontare i problemi che emergono in quel trasferimento
(struttura spaziale, assenza di un linguaggio nativo per etichettare le feature — i due
gap descritti in [`research_gap.tex`](../../paper/Your_Paper_Title_Here/Chapters/research_gap.tex)).

## 6. Tre frasi/idee da avere pronte per la discussione

1. *"Bricken et al. dimostrano empiricamente che, addestrando un SAE sulle attivazioni
   MLP di un piccolo modello linguistico, si ottengono feature monosemantiche e
   interpretabili — e che la loro rilevanza causale si può verificare tramite
   ablazione. La nostra intera architettura SAE — inclusi dettagli specifici come la
   centratura `x - b_dec` e il vincolo di norma unitaria sul decoder — viene presa
   direttamente da questo lavoro."*
2. *"Il paper mostra anche il fenomeno del 'feature splitting': dizionari più grandi
   tendono a produrre feature più specifiche e fini — è il motivo per cui, come loro,
   usiamo un dizionario sovracompleto (`expansion_factor = 8`, cioè 8 volte la
   dimensione originale)."*
3. *"La differenza fondamentale rispetto al nostro lavoro è il dominio: testo (dove le
   feature si possono interpretare leggendo le parole che le attivano) vs immagini
   (dove serve un passo aggiuntivo — l'etichettatura via CLIP — perché il modello non
   'parla' la lingua dell'interprete umano)."*
