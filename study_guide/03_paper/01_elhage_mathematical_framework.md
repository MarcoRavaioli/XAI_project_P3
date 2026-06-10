# Paper 1 — Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits"

**Citazione**: Elhage, N., Nanda, N., Olsson, C., et al. (2021). *A Mathematical
Framework for Transformer Circuits*. Anthropic / Transformer Circuits Thread.
(`Elhage2021` in [`references.bib`](../../paper/Your_Paper_Title_Here/references.bib))

> Ruolo nel nostro progetto: è il paper "fondativo" — fornisce il **vocabolario e
> l'impalcatura teorica** (residual stream, circuiti, polisemanticità) su cui poggia
> tutto il resto della letteratura citata, e viene esplicitamente richiamato nel
> docstring della nostra classe `ActivationHook`.

## 1. Il problema che affronta

All'epoca (2021), i Transformer erano già lo stato dell'arte in NLP, ma restavano
sostanzialmente "scatole nere matematiche": si sapeva *come allenarli* (backprop su una
loss), ma non esisteva un linguaggio condiviso per descrivere *cosa calcolano
internamente*, in modo che un umano potesse leggerlo come si legge codice sorgente. Il
paper si propone di colmare esattamente questo vuoto: costruire un **framework
matematico** che permetta di "decompilare" un Transformer (almeno nei casi più semplici)
in componenti dal significato comprensibile.

## 2. L'idea centrale — l'analogia del reverse engineering

La frase-chiave del paper, diventata quasi un manifesto del campo:

> "Trying to understand transformer language models can feel like trying to reverse
> engineer a large compiled binary."

L'idea è che una rete neurale allenata sia, in un certo senso, **un programma compilato
in una forma incomprensibile** (matrici di pesi). Compito del mechanistic
interpretability researcher è "decompilarlo": trovare la struttura algoritmica
sottostante, esprimibile in termini comprensibili (circuiti, regole, pattern). Per
rendere questo possibile su un sistema così complesso, il paper inizia
*deliberatamente* dai casi più semplici possibili — transformer con uno o due blocchi
di sola attenzione (niente MLP) — esattamente come uno scienziato studierebbe prima un
sistema modello (un moscerino della frutta) prima di affrontare un sistema complesso
(un essere umano).

## 3. I tre concetti tecnici chiave (approfonditi anche in [`02_concetti/02_interpretabilita_meccanicistica.md`](../02_concetti/02_interpretabilita_meccanicistica.md))

### 3.1 Il Residual Stream come "canale di comunicazione condiviso"

L'osservazione centrale: grazie alle connessioni residuali (`x = x + f(x)`), ogni
componente del Transformer non *trasforma* il flusso principale — **legge** da esso
(proiezione lineare in ingresso) e **scrive** su di esso (somma in uscita). Il paper
formalizza questa idea dicendo che il residual stream ha una sorta di "base
privilegiata" e funge da bus di comunicazione condiviso fra tutte le componenti del
modello, attraverso tutti i layer.

> Citazione chiave: *"We think of the residual stream as having a kind of 'privileged
> basis'... it's the central object through which all parts of the model communicate."*

**Conseguenza pratica**: dato che ogni contributo è *sommato* (mai sovrascritto), in
linea di principio si può isolare e studiare il contributo di una singola componente —
o, ancora più potentemente, **rimuoverlo chirurgicamente** mantenendo tutto il resto
intatto. Questa è esattamente la base teorica della nostra fase di intervento causale
(`causal_eval.perform_causal_intervention`): sottrarre `f_j · W_dec[:, j]` dal residual
stream è un'operazione ben definita, con un significato preciso, **proprio perché** il
residual stream funziona per somma additiva di contributi.

### 3.2 I Circuiti — decomposizione QK / OV

Per i blocchi di sola attenzione, il paper mostra che si può scomporre la
trasformazione operata da una testa in due matrici composte interpretabili: la matrice
**QK** (Query-Key — determina *dove* la testa "guarda", in base a posizione/contenuto)
e la matrice **OV** (Output-Value — determina *cosa scrive* sul residual stream se
seleziona un certo token). Un **circuito** è quindi un insieme di componenti (head,
connessioni tra layer) che, composte, implementano un comportamento riconoscibile —
l'esempio più celebre del paper sono gli **induction heads**, circuiti a due head che
imparano la regola "se hai visto il pattern [A][B] in precedenza, e ora rivedi [A],
predici [B]".

### 3.3 La polisemanticità — il problema lasciato aperto

Qui arriva il punto più importante per noi. Il paper **ammette esplicitamente** che il
suo framework — così elegante per i blocchi di attenzione — **fatica con i blocchi
MLP**. Il motivo: i singoli neuroni MLP sono spesso **polisemantici**, cioè rispondono
a insiemi di concetti scollegati tra loro (l'esempio canonico, citato nel paper:
un neurone che si attiva insieme su *citazioni accademiche*, *dialoghi in inglese*,
*richieste HTTP* e *testo coreano*). Senza un modo per "districare" questa
sovrapposizione, è impossibile leggere i neuroni MLP uno per uno come si fa con le
matrici QK/OV — e il framework, su quel fronte, resta incompleto.

## 4. Perché questo paper conta — direttamente — per il nostro progetto

Questo paper **apre** la domanda a cui il nostro progetto, due livelli più in là nella
catena della letteratura, prova a dare una risposta pratica:

```
Elhage 2021:    "i neuroni MLP sono polisemantici → il nostro framework si ferma qui"
       │
       ▼
Bricken/Cunningham: "ecco una soluzione — i Sparse Autoencoder districano la
                     polisemanticità delle MLP, producendo feature monosemantiche"
       │
       ▼
NOI:            "applichiamo questa soluzione — non a un LLM, ma a un Vision
                Transformer — affrontando due problemi nuovi che emergono nel
                dominio visivo (struttura spaziale, assenza di un 'linguaggio' nativo)"
```

Concretamente, tre tracce dirette di questo paper nel nostro codice:
- `model_loader.ActivationHook` cita Elhage et al. nel suo docstring — l'idea di
  "leggere e scrivere sul residual stream tramite hook" è presa pari pari da qui.
- L'intera logica degli interventi causali (`causal_eval.py`) si fonda
  sull'assunzione — giustificata da questo paper — che il residual stream sia additivo
  e quindi "chirurgicamente modificabile".
- Il nostro `introduction.tex` apre proprio richiamando l'idea che "i transformer si
  possono capire studiando i loro pesi come si farebbe il reverse engineering di un
  binario compilato" — è una citazione diretta dello spirito di questo paper.

## 5. Tre frasi/idee da avere pronte per la discussione

1. *"Il residual stream è un bus condiviso a cui ogni componente legge e scrive per
   somma — questa è la proprietà che rende possibile, in linea di principio, isolare e
   manipolare chirurgicamente un singolo contributo."*
2. *"Il framework di Elhage funziona bene per l'attenzione (decomposizione QK/OV in
   circuiti) ma si ferma davanti alle MLP — proprio per la polisemanticità dei loro
   neuroni: è il problema aperto che il nostro progetto, attraverso i SAE, prova ad
   aggirare."*
3. *"Non implementiamo la 'circuit analysis' di questo paper — lavoriamo a un livello
   più elementare (singole feature monosemantiche, non intere sotto-reti) — ma
   un'eventuale circuit analysis sul ViT è indicata esplicitamente come direzione
   futura nella nostra conclusione."*
