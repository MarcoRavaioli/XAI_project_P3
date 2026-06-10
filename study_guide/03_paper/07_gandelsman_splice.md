# Paper 7 — Gandelsman et al. 2024, "Interpreting CLIP's Image Representation via Text-Based Decomposition" (SPLICE)

**Citazione**: Gandelsman, Y., Efros, A. A., Steinhardt, J. (2024). *Interpreting
CLIP's Image Representation via Text-Based Decomposition*. ICLR 2024 (oral).
(`Gandelsman2024` in [`references.bib`](../../paper/Your_Paper_Title_Here/references.bib))

> ⚠️ **Nota di trasparenza**: il PDF di questo paper non è presente localmente nella
> cartella `references/research_papers/` — questo riassunto è costruito a partire dalla
> descrizione che ne dà il nostro [`related_work.tex`](../../paper/Your_Paper_Title_Here/Chapters/related_work.tex)
> (sezione "Concept Discovery in Vision-Language Models") più conoscenza generale del
> lavoro. **Prima della consegna, sarebbe utile recuperare il PDF originale** (è
> liberamente reperibile, es. su arXiv) e verificare/arricchire questo riassunto con
> dettagli diretti — specialmente se pensi che possano farti domande tecniche fini su
> questo paper specifico.

## 1. Il problema che affronta

CLIP (vedi [`02_concetti/04_clip_e_valutazione_crossmodale.md`](../02_concetti/04_clip_e_valutazione_crossmodale.md)
§2) produce, per ogni immagine, un singolo vettore di embedding — una rappresentazione
compatta ma del tutto opaca: 512 (o 768) numeri che catturano "il significato"
dell'immagine secondo CLIP, ma senza alcuna struttura leggibile. Il paper si chiede:
**possiamo scomporre questo singolo vettore opaco in una combinazione di "pezzi"
ciascuno descrivibile in linguaggio naturale?**

## 2. L'idea centrale — sfruttare lo spazio condiviso che CLIP già possiede

La proposta — SPLICE (Sparse Linear Concept Embeddings, o nome simile a seconda della
versione del paper) — è concettualmente affine alla nostra: usare la **sparsità** per
decomporre una rappresentazione opaca in una combinazione di direzioni più
interpretabili. Ma c'è una differenza strutturale enorme rispetto al nostro caso: il
paper può costruire il proprio "dizionario di concetti" usando **direttamente
l'embedding testuale di CLIP** — cioè può rappresentare l'immagine come combinazione
sparsa di *direzioni che corrispondono già, per costruzione, a descrizioni testuali*
(es. "questa immagine = 0.6 × 'cane' + 0.3 × 'erba' + 0.1 × 'giorno di sole'). Il
"vocabolario" del dizionario sparso è — in un certo senso — **già scritto in linguaggio
umano**, perché CLIP è stato allenato proprio per allineare immagini e testo nello
stesso spazio.

> 🧩 **Analogia**: è come scomporre un colore RGB in una combinazione di colori
> "primari con nome" (rosso, blu, verde, ...) — puoi farlo direttamente, perché quei
> nomi *sono già* coordinate dello stesso spazio. Il nostro problema, invece, è più
> simile a dover descrivere un colore misurato da uno spettrometro che non conosce i
> nomi dei colori — devi prima trovare *un secondo strumento* (uno spettrometro
> "bilingue", che sa sia misurare sia nominare) per fare da ponte.

## 3. Cosa rende questo lavoro diverso dal nostro — e perché è proprio questa differenza il "varco" del nostro research gap

Questo è il punto-chiave da portare alla discussione, perché è esattamente l'argomento
con cui il nostro `related_work.tex` chiude la sezione e introduce il research gap:

> Il metodo di Gandelsman et al. **funziona perché lo spazio di embedding
> testo-immagine è già lì, incorporato nel modello che analizzano** (CLIP). La
> "traduzione" da direzione interna a descrizione testuale è — in un certo senso —
> *gratuita*, una conseguenza diretta di come CLIP è stato allenato.
>
> Un Vision Transformer puro (come il nostro `google/vit-base-patch16-224`, allenato
> solo su etichette di classificazione ImageNet, **senza alcuna supervisione
> testuale**) **non possiede questo spazio condiviso**. Una feature scoperta nelle sue
> attivazioni interne è — letteralmente — solo una direzione anonima: si attiva su
> certe patch, ma il modello stesso non ha alcun meccanismo, alcun "dizionario interno",
> per descrivere a parole cosa quelle patch abbiano in comune.

Ecco perché **non possiamo semplicemente applicare la tecnica di Gandelsman et al. al
nostro problema** — il suo prerequisito fondamentale (uno spazio di embedding
condiviso *dentro* il modello analizzato) semplicemente non esiste nel nostro caso.
Dobbiamo costruire un meccanismo di traduzione **esterno** — ed è esattamente lì che
entra in gioco CLIP nel nostro progetto: non come modello da analizzare (come in
Gandelsman et al.), ma come **valutatore indipendente**, "preso in prestito" dall'esterno
proprio per supplire a ciò che il nostro modello non possiede.

## 4. Perché questo paper conta — direttamente — per il nostro progetto

Questo paper è il "termine di paragone" che rende visibile, per contrasto, il nostro
research gap. Senza di esso, sarebbe difficile argomentare *perché* serva una soluzione
nuova per i ViT puri — Gandelsman et al. dimostra che "il problema della spiegazione
in linguaggio naturale" è già stato risolto **in un caso specifico** (modelli
vision-language come CLIP), rendendo molto più nitido **dove** il problema resta
aperto: nei modelli che vedono ma non parlano.

## 5. Tre frasi/idee da avere pronte per la discussione

1. *"Gandelsman et al. decompongono le rappresentazioni di CLIP come combinazioni
   sparse di direzioni che corrispondono — per costruzione — a descrizioni testuali:
   un'operazione resa possibile dal fatto che CLIP possiede già uno spazio di
   embedding condiviso testo-immagine."*
2. *"Il loro metodo funziona perché analizzano un modello che 'parla' la stessa lingua
   dell'interprete umano. Il nostro modello (un ViT puro) non lo fa — ed è esattamente
   questa differenza a definire il research gap che affrontiamo."*
3. *"Non possiamo applicare la loro tecnica direttamente, ma possiamo prendere in
   prestito CLIP stesso — non come oggetto di studio, ma come strumento esterno di
   traduzione — per colmare il vuoto che il nostro modello, da solo, non può colmare."*
