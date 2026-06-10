# Concetto 4 — CLIP come "traduttore esterno": dare un nome ai concetti scoperti

> Questo file copre la parte della pipeline che risolve direttamente il **secondo
> research gap** ("Cross-Modal Evaluation") descritto in
> [`01_il_progetto_spiegato.md`](../01_il_progetto_spiegato.md) §3: una volta che il SAE
> ha trovato 6144 direzioni candidate, *come facciamo a sapere cosa rappresentano*?

## 1. Il problema, ridetto in modo molto concreto

Immagina di aver finito l'addestramento del SAE. Hai in mano una matrice `W_dec` con
6144 colonne — 6144 direzioni nello spazio a 768 dimensioni delle attivazioni MLP del
ViT. Sai (dal training: R² alto, L0 basso) che queste direzioni "ricostruiscono bene e
in modo sparso". Ma se ti chiedessero **"e la feature numero 4049, cosa rappresenta?"**,
cosa risponderesti guardando solo i numeri della colonna 4049 di `W_dec`? Niente — sono
768 numeri in virgola mobile, completamente illeggibili a occhio.

In un LLM questo problema ha una soluzione "gratis": ogni token è una parola, quindi
basta guardare *quali parole* attivano la feature, e il significato emerge da solo
("si attiva su 'gatto', 'cane', 'criceto'... → probabilmente rappresenta 'animali
domestici'"). Ma un token ViT è una **patch di pixel**, non una parola — e il ViT stesso
non possiede alcun meccanismo per tradurre "questa zona di pixel" in linguaggio. Ecco
il **gap di valutazione cross-modale**: ci serve un "traduttore" che sappia fare il
ponte tra "schema visivo" e "descrizione testuale", **dall'esterno**, perché il modello
che stiamo studiando non lo sa fare da solo.

## 2. La soluzione: CLIP come "ponte linguistico" indipendente

**CLIP** (Radford et al. 2021, "Learning Transferable Visual Models From Natural
Language Supervision") è un modello allenato su centinaia di milioni di coppie
*immagine + didascalia testuale* trovate sul web, con un obiettivo molto semplice da
descrivere: imparare a proiettare immagini e testi in **uno spazio di embedding
condiviso**, in modo che un'immagine e la sua descrizione corretta finiscano "vicine"
(alta similarità coseno), mentre coppie sbagliate finiscano "lontane".

Il punto cruciale per noi: **CLIP non è il modello che stiamo studiando**. È un
osservatore esterno, indipendente, che "parla" entrambe le lingue (visiva e testuale)
per costruzione. Usarlo come "etichettatrice" è esattamente la strategia descritta nel
nostro `research_gap.tex`: *"questo disaccoppia l'analisi di interpretabilità dal
modello backbone, evitando il bisogno di un'architettura vision-language [nel modello
da analizzare] e ancorando le feature scoperte a concetti comprensibili dall'uomo."*

> 🧩 **Analogia**: è come studiare il comportamento di un animale che non parla la tua
> lingua (il ViT) chiedendo aiuto a un interprete bilingue (CLIP) che, guardando le
> stesse "scene" che hanno catturato l'attenzione dell'animale, ti dice in parole tue
> "ah, sembra che si stia concentrando su questo tipo di cosa". L'interprete non sa
> *cosa pensa* l'animale — ma sa descrivere *cosa vede*, ed è esattamente l'anello
> mancante di cui abbiamo bisogno.

## 3. La pipeline di etichettatura, passo per passo (con il codice)

Il flusso implementato in [`interpretability.py`](../../src/interpretability.py) si
articola in tre fasi sequenziali — esattamente quelle descritte nella sezione "CLIP
Cross-Modal Evaluation" del nostro `methodology.tex`:

### 3.1 Fase 1 — Trovare i "migliori esempi" di una feature: `get_top_activating_patches`

```python
patch_tokens = activation[:, 1:, :]
f = sae.encode(patch_tokens)
feature_activation = f[:, :, feature_idx]   # quanto si attiva la feature 'feature_idx'
                                             # su ogni patch di ogni immagine
```

Si scorre l'intero dataset di validazione, si calcola — per ogni patch di ogni immagine
— quanto la feature in esame si attiva, e si mantiene una lista delle **top-K patch**
(ordinata, lunga al massimo K) con l'attivazione più alta in assoluto
([`interpretability.py:138-140`](../../src/interpretability.py#L138)). È
concettualmente identico al meccanismo dei "top-activating dataset examples" usato da
Bricken et al. per interpretare le feature dei loro SAE su modelli linguistici — solo
che invece di "le 10 frasi che attivano di più questa feature", qui sono "le K patch
d'immagine che la attivano di più".

### 3.2 Fase 2 — Trasformare l'astrazione "token" in pixel reali: `extract_contextual_crop`

Una volta trovate le patch più attivanti, dobbiamo "mostrarle" a CLIP — ma una singola
patch 16×16 pixel è minuscola e spesso poco informativa fuori contesto (pensa a un
quadratino 16×16 che mostra solo "un pezzo di pelo marrone": fuori contesto potrebbe
essere pelliccia, terra, legno...). Per questo, l'etichettatura usa il **ritaglio
contestuale** — non solo la singola patch, ma un'area più ampia centrata su di essa:

```python
def extract_contextual_crop(image, spatial_idx, patch_size, grid_size, context_patches=2):
    row = spatial_idx // grid_size
    col = spatial_idx % grid_size
    row_start = max(0, row - context_patches) * patch_size
    row_end   = min(grid_size, row + context_patches + 1) * patch_size
    col_start = max(0, col - context_patches) * patch_size
    col_end   = min(grid_size, col + context_patches + 1) * patch_size
    return image[:, row_start:row_end, col_start:col_end]
```

Con `context_patches=2`, il ritaglio risultante può arrivare fino a `(2·2+1) × 16 = 80`
pixel di lato — circa **5 volte più grande** della singola patch, abbastanza per dare a
CLIP un contesto visivo sensato (es. non solo "un pezzo di pelo", ma "un pezzo di pelo
*su un muso di animale*"), pur restando centrato precisamente sulla zona che ha
attivato la feature. Le clausole `max(0, ...)` e `min(grid_size, ...)` gestiscono
correttamente i bordi dell'immagine (una patch nell'angolo non avrà 2 patch di contesto
disponibili in ogni direzione).

> Nota anche che le coordinate sono calcolate **due volte, in modo leggermente diverso**
> a seconda dell'uso: `extract_patch_crop` lavora in pixel assoluti
> (`row = (idx // grid) * patch_size`), `extract_contextual_crop` lavora prima in
> coordinate-griglia e converte in pixel solo per gli estremi del ritaglio
> (`row_start = max(0, row - ctx) * patch_size`). Stesso principio geometrico, due
> implementazioni adattate al loro scopo specifico (un singolo box vs un intervallo con
> clamping ai bordi).

### 3.3 Fase 3 — Chiedere a CLIP: "questi ritagli somigliano di più a quale concetto?"

Questo è il cuore di `CLIPAutoLabeler.label_feature`
([`interpretability.py:169-266`](../../src/interpretability.py#L169)). I passaggi:

1. **De-normalizzazione delle immagini**: le immagini nel dataset sono normalizzate con
   media/deviazione standard di ImageNet (`mean=[0.485,0.456,0.406]`,
   `std=[0.229,0.224,0.225]`) per il training del ViT — ma CLIP si aspetta immagini
   "naturali" (range [0,1] o [0,255]). Si inverte quindi la normalizzazione
   (`img * std + mean`) prima di passarle al suo processore.
2. **Costruzione dei prompt testuali** — qui c'è una scelta progettuale precisa, presa
   pari pari dal paper originale di CLIP:
   ```python
   formatted_texts = [f"a photo of {concept}" for concept in candidate_concepts]
   ```
   Questo "incorniciamento" del concetto in un **prompt template naturale** (*"a photo
   of {concetto}"* invece della sola parola nuda, es. "fur") non è decorativo: CLIP è
   stato allenato su didascalie naturali del web, che raramente sono singole parole
   isolate. Inquadrare il concetto in una frase realistica avvicina l'input alla
   distribuzione su cui CLIP è stato allenato, e produce embedding testuali più
   affidabili — è una tecnica nota come **prompt engineering per zero-shot
   classification**, descritta nello stesso paper di Radford et al.
3. **Embedding e normalizzazione su sfera unitaria**:
   ```python
   image_features = self.model.get_image_features(**image_inputs)
   text_features  = self.model.get_text_features(**text_inputs)
   image_features = image_features / image_features.norm(dim=-1, keepdim=True)
   text_features  = text_features  / text_features.norm(dim=-1,  keepdim=True)
   ```
   Normalizzare ogni embedding a **norma 1** è ciò che trasforma un semplice prodotto
   scalare in una **similarità coseno** — la metrica standard per confrontare direzioni
   (anziché intensità) in uno spazio di embedding, esattamente ciò che vogliamo: "questi
   due vettori puntano nella stessa direzione semantica?", non "hanno la stessa norma?".
4. **Matrice di similarità e media sugli esemplari**:
   ```python
   similarity_matrix  = torch.matmul(image_features, text_features.t())  # [K, num_concetti]
   mean_similarities  = similarity_matrix.mean(dim=0)                    # media su tutti i K esemplari
   best_idx     = np.argmax(mean_similarities)
   best_concept = candidate_concepts[best_idx]
   ```
   Si calcola la similarità coseno tra **ciascuno dei K ritagli** e **ciascun concetto
   candidato**, poi si fa la **media lungo gli esemplari**. Questa media è una scelta
   metodologicamente importante: non basta che *un singolo* ritaglio somigli al
   concetto candidato — vogliamo che la somiglianza sia **consistente su tutti i top-K
   esemplari**. È proprio questa consistenza (o la sua assenza) il segnale che useremo
   per distinguere monosemanticità da polisemanticità (vedi §4).

### 3.4 Il "vocabolario" dei concetti candidati

`candidate_concepts` è una lista di parole/espressioni scelte a priori (es. "fur",
"eye", "wheel", "red color", "scale pattern"...) — costruita per essere plausibilmente
rilevante rispetto al dataset usato (`run_pipeline.py` ne definisce set diversi a
seconda che si usi CIFAR-10, ImageWoof, Imagenette o ImageNet). Questo è anche
**il limite più importante e più onesto da dichiarare** (ed è esattamente ciò che dice
la nostra `conclusion.tex`): CLIP può solo scegliere **tra le opzioni che gli offriamo**.
Se il vero concetto rappresentato da una feature non è nella lista, CLIP sceglierà
comunque "il meno peggio" tra le alternative disponibili — un'etichetta plausibile ma
potenzialmente fuorviante. È un compromesso necessario (un vocabolario aperto
richiederebbe un setup molto più complesso, tipo *captioning* generativo), ma va
riconosciuto esplicitamente in fase di discussione dei risultati.

## 4. Come distinguiamo monosemanticità da polisemanticità, in pratica

Non basta un punteggio di similarità alto: bisogna guardare **se la coerenza si
mantiene su tutti i K esemplari**. Due scenari concreti, opposti:

- ✅ **Monosemantica**: i 5 ritagli più attivanti per la feature 4049 mostrano tutti
  superfici rosse uniformi (un peperone, una mela, un'insegna, un vestito, un mattone).
  Lo score CLIP per "red color" è alto e *consistente* (similarità simile su tutti e 5).
  → la feature rappresenta in modo affidabile "il concetto colore rosso".
- ❌ **Polisemantica** (un caso da segnalare e discutere, non da nascondere!): i 5
  ritagli mostrano un occhio di gatto, una ruota di bicicletta, una texture di corteccia,
  un riflesso su vetro, e una scritta. Nessun concetto candidato avrà uno score alto e
  consistente — i punteggi saranno bassi e "spalmati". → la feature **non** è stata
  ben districata dal SAE: probabilmente rappresenta ancora una sovrapposizione di
  concetti diversi (residuo di superposition non risolto), oppure il vocabolario di
  candidati non copre il vero concetto sottostante.

> 📌 **Suggerimento pratico per la presentazione**: prepara, se possibile, un esempio
> di entrambi i casi dai tuoi risultati reali (`multi_feature_exemplar_grid.png` ti dà
> esattamente questo: i 5 ritagli + le immagini intere + le heatmap, fianco a fianco).
> Mostrare *anche* un caso di fallimento (feature polisemantica non ben etichettata) è
> molto più convincente — e onesto — che mostrare solo i casi di successo: dimostra che
> avete capito i limiti del metodo, non solo i suoi punti di forza.

## 5. Le visualizzazioni — come "vedere" tutto questo

Due funzioni producono gli artefatti visivi principali:

- **`save_feature_grid_visualization`**: per un set di feature rappresentative, genera
  una griglia con **3 righe per feature**: (1) i 5 ritagli più attivanti ingranditi,
  (2) le immagini intere con un riquadro rosso che evidenzia *dove* si trova la patch
  attivante, (3) heatmap di attivazione spaziale sovrapposte. È l'artefatto
  `multi_feature_exemplar_grid.png` citato in
  [`01_il_progetto_spiegato.md`](../01_il_progetto_spiegato.md) §5 — probabilmente
  l'immagine più "parlante" da mostrare in presentazione.
- **`get_feature_activation_map` / `save_feature_activation_heatmap`**: calcolano e
  visualizzano, per una singola immagine, **l'intera mappa 14×14** di attivazione di
  una feature — non solo "dov'è il massimo", ma "come si distribuisce l'attivazione su
  tutta l'immagine". Da notare il piccolo accorgimento implementativo
  (`activation_cache`, [`interpretability.py:277,310-311`](../../src/interpretability.py#L277)):
  evita di rifare un intero forward pass del ViT per ogni feature da visualizzare sulla
  stessa immagine — se l'attivazione per quell'immagine/layer è già stata calcolata, la
  si riusa. Un dettaglio di efficienza, ma utile da menzionare se ti chiedono "quanto
  costa generare questi artefatti?".

## 6. Glossario rapido di questa sezione

- **CLIP**: modello vision-language allenato a proiettare immagini e testo in uno
  spazio di embedding condiviso, tramite contrastive learning su coppie immagine-testo.
- **Zero-shot classification**: classificare senza fine-tuning, semplicemente
  confrontando l'embedding dell'input con gli embedding di descrizioni testuali
  candidate, e scegliendo quella più simile.
- **Ritaglio contestuale (`extract_contextual_crop`)**: porzione di immagine più ampia
  della singola patch, centrata su di essa, usata per dare a CLIP contesto visivo
  sufficiente.
- **Prompt template ("a photo of {concept}")**: tecnica di formattazione del testo che
  avvicina l'input alla distribuzione di training di CLIP, migliorando l'affidabilità
  dell'embedding testuale.
- **Similarità coseno**: misura di quanto due vettori "puntano nella stessa direzione",
  ottenuta come prodotto scalare di vettori normalizzati a norma 1 — la metrica
  standard per confrontare embedding semantici.
- **Vocabolario di concetti candidati**: insieme predefinito di etichette testuali tra
  cui CLIP può scegliere — è anche il principale limite dichiarato del metodo.
