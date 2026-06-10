# Concetto 1 — Vision Transformer (ViT): come funziona davvero, dentro

> Prerequisito per tutto il resto: il SAE e gli interventi causali operano *dentro* le
> attivazioni del ViT. Se non hai chiaro cosa significa "il token al layer 6, posizione
> spaziale 83, sotto-blocco MLP", il resto del progetto resta astratto. Questo file ti
> dà le coordinate per orientarti dentro al modello.

## 1. Il problema che il ViT risolve: trasformare un'immagine in una sequenza

Un Transformer (l'architettura di GPT, BERT, ecc.) lavora su **sequenze di vettori**.
Il testo è naturalmente una sequenza (parole/token in ordine). Un'immagine, invece, è
una griglia 2D di pixel — non una sequenza. Il contributo centrale di Dosovitskiy et al.
2021 ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale") è
una ricetta sorprendentemente semplice per *forzare* un'immagine in quella forma:

> **Taglia l'immagine in patch quadrate, srotola ogni patch in un vettore, trattalo
> come se fosse una "parola".**

### 1.1 La pipeline di tokenizzazione, passo per passo (con i numeri del nostro modello)

Il nostro backbone è **`google/vit-base-patch16-224`** (la variante "ViT-B/16"):

1. **Input**: immagine RGB ridimensionata a 224×224 pixel.
2. **Patchification**: la si divide in patch non sovrapposte di **16×16 pixel**.
   Lungo ciascun lato ci stanno `224 / 16 = 14` patch, quindi in totale
   `14 × 14 = 196` patch — organizzate in una **griglia 14×14**
   (questi numeri sono esattamente `model_wrapper.patch_size = 16` e
   `model_wrapper.grid_size = 14` in [`model_loader.py:108-109`](../../src/model_loader.py#L108-L109)).
3. **Linear projection ("patch embedding")**: ogni patch (16×16×3 = 768 numeri grezzi)
   viene proiettata linearmente in un vettore di **768 dimensioni** — la "dimensione
   nascosta" del modello, `d_model = 768` (`hidden_size` in
   [`model_loader.py:114`](../../src/model_loader.py#L114)). Da questo momento, ogni
   patch è un "token" come una parola lo è per un LLM.
4. **Token `[CLS]`**: in testa alla sequenza si aggiunge un token speciale "appreso"
   (non corrisponde a nessuna patch reale): il **classification token**. Il suo compito
   è aggregare informazione da tutta l'immagine, perché alla fine sarà *solo lui* — non
   le patch — a essere passato alla testa di classificazione finale.
5. **Positional embedding**: si somma a ogni token un vettore di posizione appreso, per
   "ricordare al modello" dove si trovava quella patch nella griglia originale (il
   Transformer di per sé è invariante all'ordine — senza questa aggiunta perderebbe ogni
   informazione spaziale).

> **Risultato**: una sequenza di `1 + 196 = 197` vettori a 768 dimensioni — esattamente
> la forma `(197, 768)` che vedi citata nel nostro `methodology.tex` e che
> `ActivationHook` cattura ad ogni layer.

### 1.2 Esempio concreto — "dove" sta il token numero 83?

Questo calcolo torna identico, parola per parola, nella funzione
`extract_patch_crop` di [`interpretability.py`](../../src/interpretability.py) — vale
la pena capirlo bene una volta per tutte:

```python
row = (spatial_idx // grid_size) * patch_size   # spatial_idx=83, grid_size=14, patch_size=16
col = (spatial_idx % grid_size)  * patch_size
# row = (83 // 14) * 16 = 5 * 16  = 80
# col = (83 % 14)  * 16 = 11 * 16 = 176
```

Cioè: il token "spaziale" numero 83 (ricorda: nella sequenza completa è all'indice 84,
perché il token 0 è il `[CLS]`) corrisponde alla patch che occupa i pixel
`[80:96, 176:192]` nell'immagine 224×224 originale — riga 5, colonna 11 della griglia
14×14. **Questa è la "traduzione" fondamentale** che rende possibile passare da "una
feature del SAE si attiva sul token 83" a "una feature del SAE si attiva su *quella zona
specifica dell'immagine*, lì in alto a destra" — ed è ciò che rende interpretabile, in
senso visivo, qualunque cosa scopra il SAE.

## 2. Cosa succede dentro un blocco Transformer (i 12 "piani" dell'edificio)

Il ViT-B ha **12 blocchi identici** impilati in sequenza (un "layer" = un blocco). Ogni
blocco ha la stessa struttura interna, **ripetuta identica 12 volte**:

```
ingresso (residual stream)
   │
   ├──► LayerNorm ──► Multi-Head Self-Attention ──► (+) ──┐
   │                                                       │
   └───────────────────────────────────────────────────► somma
                                                            │
   ┌────────────────────────────────────────────────────────
   │
   ├──► LayerNorm ──► MLP (2 layer lineari + GELU) ──► (+) ──┐
   │                                                          │
   └────────────────────────────────────────────────────────► somma ──► uscita
```

Questa è l'architettura **pre-norm con connessioni residuali**: ogni sotto-blocco
(attention, poi MLP) **legge** dal flusso principale, calcola qualcosa, e **scrive** la
sua "proposta di modifica" sommandola di nuovo al flusso. Non sostituisce mai il flusso:
lo arricchisce. Questa singola osservazione — che chiameremo **residual stream** — è la
chiave di volta concettuale di tutto il progetto, e la riprendiamo in dettaglio nel file
sulla [Mechanistic Interpretability](02_interpretabilita_meccanicistica.md), perché è
lì che Elhage et al. costruiscono l'intero loro framework.

### 2.1 Multi-Head Self-Attention — "ogni patch guarda tutte le altre"

L'attenzione è il meccanismo che permette a ogni token di **raccogliere informazione da
tutti gli altri token**, pesata per "quanto sono rilevanti per me". Concretamente, ogni
token produce tre vettori — Query (Q, "cosa sto cercando"), Key (K, "cosa offro"), Value
(V, "cosa comunico se vengo scelto") — e l'output di ogni token è una **media pesata**
dei Value di tutti i token, dove i pesi vengono dalla similarità tra la sua Query e le
Key altrui (softmax su `Q·Kᵀ / √d`). "Multi-head" significa che questo si fa in
parallelo con diversi sotto-spazi (head), ciascuno potenzialmente specializzato in un
tipo diverso di relazione (texture vicine, forma globale, contrasto di colore...).

> **Differenza cruciale rispetto a un LLM** (la trovi anche discussa nel nostro
> `introduction.tex` come primo "research gap"): in un LLM l'attenzione è *causale* —
> un token può guardare solo quelli che lo precedono (altrimenti "barerebbe" guardando
> il futuro durante il training). In un ViT l'attenzione è **bidirezionale** — ogni
> patch guarda *tutte* le altre, in entrambe le direzioni, senza alcun ordine
> privilegiato. Questo è proprio ciò che rende più complesso "tracciare chi influenza
> chi" in un ViT: non c'è una freccia del tempo da seguire.

### 2.2 MLP (il sotto-blocco che... interessa direttamente a noi!)

Dopo l'attenzione arriva il blocco **MLP** (Multi-Layer Perceptron): due trasformazioni
lineari con una non-linearità (GELU) in mezzo, che **espande** temporaneamente la
dimensione (768 → genericamente 4×768=3072 nel ViT-B) e poi la **ricomprime** a 768.
Se l'attenzione è il meccanismo che fa *muovere* l'informazione tra le posizioni, l'MLP
è (secondo l'intuizione classica della mechanistic interpretability, Elhage et al.) il
meccanismo che fa "elaborazione/ragionamento per-token": legge ciò che è nel residual
stream in quella posizione, applica una trasformazione non-lineare, e riscrive il
risultato.

> 🎯 **Perché è IL sotto-blocco che ci interessa**: la nostra intera pipeline registra
> i suoi hook proprio sull'**output del sotto-blocco MLP**
> (`get_submodule(layer_idx, "mlp")` in [`model_loader.py:142-163`](../../src/model_loader.py#L142-L163)).
> Il motivo non è arbitrario: è esattamente il sito dove Bricken et al. e Cunningham et
> al. hanno mostrato che si annida la polisemanticità più interessante — e dove i SAE
> sono stati storicamente più produttivi nello scoprire feature monosemantiche
> (vedi [`03_sparse_autoencoder.md`](03_sparse_autoencoder.md) §1).

### 2.3 Il token `[CLS]`: l'unico che "conta" alla fine

Dopo i 12 blocchi, **solo il token `[CLS]`** (posizione 0) viene passato alla testa di
classificazione finale (un singolo layer lineare che produce i logit sulle 1000 classi
di ImageNet). Le altre 196 posizioni vengono semplicemente scartate a quel punto. Ma
attenzione: questo non significa che le patch siano "inutili" — anzi, è proprio
attraverso l'attenzione, layer dopo layer, che l'informazione di tutte le 196 patch
**confluisce** nel `[CLS]`. È esattamente questo "imbuto" — tutta l'informazione
spaziale che si comprime in un solo vettore — il secondo aspetto del "Spatial
Challenge" descritto nel nostro `research_gap.tex`: tracciare *come* l'evidenza da
regioni specifiche dell'immagine si propaga e si concentra nel `[CLS]` non è affatto
banale.

> **Conseguenza pratica per noi**: quando alleniamo il SAE, **scartiamo deliberatamente
> il token `[CLS]`** (vedi `TokenActivationBuffer.fill_buffer`,
> [`caching_and_training.py:78-82`](../../src/caching_and_training.py#L78-L82) — commento
> esplicito: *"ViT and DINOv2 prepend a classification token [CLS] at seq index 0.
> Discard index 0 to focus on spatial patch tokens"*). Motivo: il `[CLS]` non corrisponde
> a nessuna posizione fisica nell'immagine — non avrebbe senso applicare
> `extract_patch_crop` su di lui, e mischiarlo con le patch "spaziali" introdurrebbe
> un'attivazione strutturalmente diversa (un aggregatore globale) dentro un dataset che
> vogliamo composto da "pezzi di immagine localizzabili".
>
> Allo stesso modo, durante gli interventi causali (`causal_eval.py`), il `[CLS]` viene
> esplicitamente **escluso da ogni manipolazione** — viene "tagliato fuori", modificato
> solo il resto, e poi ricucito: `cls_token = x[:, 0:1, :]` ... `x_modified =
> torch.cat([cls_token, patch_tokens_modified], dim=1)`
> ([`causal_eval.py:43-49,75`](../../src/causal_eval.py#L43)). Capire *perché* questa
> scelta è corretta — e non solo "una linea di codice" — è un punto su cui il docente
> potrebbe insistere: la risposta è che vogliamo isolare l'effetto della feature SAE
> *sulle rappresentazioni spaziali*, lasciando intatto il "riassunto" che il modello ha
> già costruito fino a quel punto, e osservare come la sua *propagazione successiva*
> (nei layer seguenti, fino al `[CLS]` finale) viene alterata.

## 3. ViT-B/16 in cifre — la "scheda anagrafica" da ricordare

| Parametro | Valore | Dove lo trovi nel codice |
|---|---|---|
| Risoluzione immagine | 224 × 224 px | config del modello pre-addestrato |
| Dimensione patch | 16 × 16 px | `model_wrapper.patch_size` |
| Numero di patch (griglia) | 14 × 14 = 196 | `model_wrapper.grid_size` |
| Lunghezza sequenza | 197 (= 196 patch + 1 `[CLS]`) | shape dell'attivazione catturata |
| Dimensione nascosta `d_model` | 768 | `model_wrapper.d_model` / `hidden_size` |
| Numero di blocchi (layer) | 12 | indice 0-based: 0..11 |
| Layer monitorati nel progetto | 6 e 11 (indici 0-based: 5 e 10) | `layers_to_compare = [5, 10]` in `run_pipeline.py` |
| Sotto-modulo agganciato | output del blocco MLP | `get_submodule(layer_idx, "mlp")` |

## 4. Cosa "vede" un ViT, e come cambia con la profondità

Dosovitskiy ci dice *com'è fatto* il ViT. Raghu et al. (2021, "Do Vision Transformers
See Like CNNs?") ci dicono *cosa succede dentro mentre lavora* — usando la metrica CKA
(Centered Kernel Alignment) per confrontare le rappresentazioni interne layer per layer
e tra ViT e ResNet. Le scoperte principali, rilevanti per noi:

- **I ViT "vedono globalmente" fin dal primo layer** (alcune teste di attenzione
  guardano già lontano nell'immagine fin dal layer 1), mentre le CNN costruiscono il
  campo recettivo gradualmente, layer dopo layer.
- **Le rappresentazioni dei ViT sono molto più "uniformi" in profondità**: nelle CNN ci
  sono fasi nettamente separate (early = bordi/texture, late = oggetti/parti); nei ViT
  la transizione è più graduale e i blocchi consecutivi si somigliano di più tra loro
  (effetto diretto delle connessioni residuali, che permettono all'informazione di
  "saltare" i blocchi).
- **L'informazione di localizzazione spaziale resta accessibile fino agli ultimi
  layer** nei ViT (a differenza delle CNN, dove il pooling la distrugge progressivamente).
  Questo è esattamente ciò che **rende possibile** la nostra analisi: possiamo prendere
  un token al layer 11 (quasi alla fine della rete) e ancora mapparlo, con
  `extract_patch_crop`, su una zona precisa dell'immagine originale — un'operazione che
  in una CNN profonda, a quel livello, non avrebbe più alcun significato spaziale chiaro.

> 🔗 Questo è il fondamento diretto della nostra scelta sperimentale di confrontare il
> **layer 6** (metà rete — "mid-network", dove ti aspetti concetti ancora abbastanza
> locali/di basso livello) con il **layer 11** (fine rete — "late-network", dove ti
> aspetti concetti più astratti/semantici). Approfondimento completo del paper in
> [`03_paper/05_raghu_vit_vs_cnn.md`](../03_paper/05_raghu_vit_vs_cnn.md).

## 5. Glossario rapido di questa sezione

- **Patch embedding**: proiezione lineare che trasforma una patch 16×16×3 in un vettore
  768-dimensionale ("token").
- **`[CLS]` token**: token speciale aggiunto in testa alla sequenza, che aggrega
  informazione globale e da solo guida la classificazione finale.
- **Positional embedding**: vettore appreso sommato a ogni token per codificarne la
  posizione nella griglia originale.
- **Residual stream**: il "flusso principale" a cui ogni sotto-blocco somma il proprio
  contributo (vedi [concetto 2](02_interpretabilita_meccanicistica.md) per i dettagli).
- **Multi-Head Self-Attention**: meccanismo che fa scambiare informazione tra le patch,
  pesata per rilevanza reciproca, in più "canali" (head) paralleli.
- **MLP sub-block**: trasformazione non-lineare per-token, espande e ricomprime la
  dimensione; è il sito che monitoriamo con i nostri hook.
- **`d_model` / hidden size**: la dimensione dei vettori che scorrono nel residual
  stream (768 per ViT-B).
