# Il codice, spiegato — un giro guidato attraverso `src/`

> Obiettivo: che tu possa aprire un qualunque file di `src/`, indicare una funzione e
> spiegare **cosa fa, perché esiste, e a quale concetto/scelta metodologica
> corrisponde** — senza dover rileggere il codice da zero. Questo file segue l'ordine
> *logico* di esecuzione della pipeline (non l'ordine alfabetico dei file): è lo stesso
> ordine in cui li racconteresti se qualcuno ti chiedesse "spiegami passo passo cosa
> succede quando lanci `run_pipeline.py`".

```
run_pipeline.py  ──orchestratore: usa tutto il resto, in sequenza──┐
                                                                     │
   model_loader.py  → backbone + meccanismo di hook (le "fondamenta")
   sae.py           → architettura del modello ausiliario (Stadio 2)
   caching_and_training.py → buffer streaming + training loop (Stadio 2)
   interpretability.py → ricerca esemplari + etichettatura CLIP + viz (Stadio 3a)
   causal_eval.py   → ablation/steering + metriche causali + plot (Stadio 3b)
```

---

## 1. `model_loader.py` — le fondamenta: il modello e il "bisturi" per osservarlo

### 1.1 `set_seed` — riproducibilità, prima di tutto

```python
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
La primissima riga eseguita da `main()`. In un progetto che fa addestramento
(SAE), campionamento (selezione del subset di dati), e confronti tra layer, la
riproducibilità non è un dettaglio: è ciò che permette a te (o al docente) di rieseguire
la pipeline e ottenere risultati comparabili. `cudnn.deterministic = True` /
`benchmark = False` sacrificano un po' di velocità in cambio di determinismo
nelle operazioni GPU — uno scambio sensato in un contesto di ricerca/valutazione,
molto meno in uno di produzione.

### 1.2 `ActivationHook` — il meccanismo unico che serve a TUTTO

```python
def hook_fn(self, module, inputs, outputs):
    actual_output = outputs[0] if isinstance(outputs, tuple) else outputs
    self.activation = actual_output.detach()          # SEMPRE: cattura
    if self.callback is not None:                      # SE fornito: modifica
        modified_output = self.callback(actual_output)
        return (modified_output,) + outputs[1:] if is_tuple else modified_output
    return outputs
```

Questa è probabilmente **la singola astrazione più importante di tutto il progetto** —
vale la pena saperla raccontare bene, perché spiega un principio di design elegante:
*"un solo meccanismo, due usi"*.

- **Uso 1 — solo lettura** (`callback=None`): catturiamo l'attivazione e la lasciamo
  passare inalterata. È così che `TokenActivationBuffer` raccoglie i dati per
  addestrare il SAE, e così che `get_top_activating_patches` cerca gli esemplari più
  attivanti — in entrambi i casi, "guardiamo senza toccare".
- **Uso 2 — lettura E scrittura** (`callback=intervention_callback`): catturiamo
  l'attivazione (sempre), MA **sostituiamo** l'output del modulo con il risultato della
  callback — che a sua volta legge l'attivazione, applica una trasformazione (ablation
  o steering, vedi `causal_eval.py`), e restituisce una versione modificata. È così
  che realizziamo gli interventi causali.

Il dettaglio `actual_output = outputs[0] if isinstance(outputs, tuple) else outputs`
gestisce un'inconsistenza reale dell'API di Hugging Face: alcuni moduli restituiscono
direttamente un tensore, altri restituiscono una tupla `(tensore, metadati...)` — il
codice si adatta a entrambi i casi senza differenziare il chiamante.

> 🔗 Il docstring cita esplicitamente Elhage et al. — è la traduzione in codice
> dell'idea "i componenti del modello leggono e scrivono sul residual stream" (vedi
> [`02_concetti/02_interpretabilita_meccanicistica.md`](02_concetti/02_interpretabilita_meccanicistica.md) §2.1).

### 1.3 `ViTModelWrapper` — un'interfaccia uniforme su due famiglie di modelli

```python
if "dinov2" in model_name:
    self.model = Dinov2Model.from_pretrained(model_name)
    self.model_type = "dinov2"; self.patch_size = 14; self.grid_size = 16
else:
    self.model = ViTForImageClassification.from_pretrained(model_name)
    self.model_type = "vit"; self.patch_size = 16; self.grid_size = 14
```

Il wrapper "nasconde" le differenze tra ViT supervisionato e DINOv2 self-supervised
dietro un'unica interfaccia (`get_layer`, `get_submodule`). Per il nostro progetto, la
riga interessante è `self.d_model = self.model.config.hidden_size` — leggiamo `768`
**direttamente dalla configurazione del modello pre-addestrato**, non da una costante
hard-coded: una buona pratica difensiva (se domani si cambia checkpoint, il codice
resta corretto).

```python
def get_submodule(self, layer_idx, target_type="mlp"):
    layer = self.get_layer(layer_idx)
    if target_type == "mlp":
        if hasattr(layer, "mlp"): return layer.mlp
        elif hasattr(layer, "output") and hasattr(layer.output, "dense"): return layer.output.dense
    elif target_type == "residual":
        return layer
```

Questa funzione è **il punto dove decidiamo, esplicitamente, "cosa stiamo guardando"**.
Passare `"mlp"` ci dà il sotto-modulo che computa la trasformazione MLP — la nostra
scelta primaria, motivata dalla letteratura SAE (vedi
[`02_concetti/01_vision_transformer.md`](02_concetti/01_vision_transformer.md) §2.2).
Passare `"residual"` darebbe accesso all'intero blocco — un'opzione architetturalmente
prevista ma non quella usata nella nostra analisi principale (utile da menzionare se ti
chiedono "il codice supporta anche altri siti di analisi?": sì, è già predisposto).

---

## 2. `sae.py` — l'architettura del modello ausiliario

Già spiegato equazione per equazione, riga per riga, in
[`02_concetti/03_sparse_autoencoder.md`](02_concetti/03_sparse_autoencoder.md) §2 — qui
solo la "mappa di navigazione rapida" per orientarti nel file durante una domanda alla
lavagna:

| Cosa cerchi | Dove sta | Equazione corrispondente |
|---|---|---|
| Dimensioni del dizionario | `__init__`, righe 16-18 | `d=768`, `m=8d=6144` |
| Centratura pre-encoder | `encode`, riga 64 | `x̄ = x - b_dec` |
| Encoder + ReLU (sparsità "strutturale") | `encode`, riga 65 | `f = ReLU(W_enc·x̄ + b_enc)` |
| Decoder + ricostruzione | `decode`, righe 79-81 | `x̂ = W_dec·f + b_dec` |
| Loss combinata | `compute_loss`, righe 114-118 | `L = MSE(x,x̂) + λ‖f‖₁` |
| Vincolo anti-scorciatoia | `normalize_decoder_weights`, righe 100-106 | `‖W_dec[:,j]‖₂ = 1 ∀j` |

> ⚠️ Ricorda la discrepanza da segnalare: `methodology.tex` parla di `m=4d=3072`, il
> codice usa `expansion_factor=8` → `m=6144`.

---

## 3. `caching_and_training.py` — come "diamo da mangiare" al SAE

### 3.1 `TokenActivationBuffer.fill_buffer` — passo per passo

```python
_ = self.model_wrapper.model(images)               # forward pass del ViT (l'hook cattura)
activation = self.hook.activation                   # [batch, 197, 768]
patch_activations = activation[:, 1:, :]            # scarta [CLS] → [batch, 196, 768]
flat_activations = patch_activations.reshape(-1, self.model_wrapper.d_model)  # [batch*196, 768]
self.buffer = torch.cat([self.buffer, flat_activations], dim=0)
# ... quando il buffer è pieno:
perm = torch.randperm(len(self.buffer))
self.buffer = self.buffer[perm]                     # mescola
```

Nota cosa NON facciamo: non alleniamo il SAE su *immagini intere*. Lo alleniamo su
**singoli token** — frammenti spaziali. Questo `reshape(-1, d_model)` è l'operazione
concettuale che "spacchetta" un batch di 8 immagini × 196 patch in 1568 esempi di
addestramento indipendenti — esattamente ciò che rende il SAE in grado di imparare
"questo pattern di attivazione, ovunque compaia in qualunque immagine, rappresenta il
concetto X", anziché qualcosa di legato a una singola immagine specifica.

Il mescolamento (`randperm`) è la risposta diretta a un rischio concreto: senza di
esso, un batch di addestramento del SAE sarebbe quasi interamente composto da patch
della stessa manciata di immagini (quelle appena passate nel ViT) — fortemente
correlate tra loro. Mescolando l'intero buffer (token di immagini diverse, raccolti in
momenti diversi), ogni batch di addestramento diventa statisticamente più
rappresentativo della popolazione complessiva.

### 3.2 `train_sae` — il loop, e la scelta dell'ordine delle operazioni

```python
optimizer.zero_grad()
x_hat, f = sae(batch_tokens)
loss_dict = sae.compute_loss(batch_tokens, x_hat, f, l1_coeff)
loss.backward()
optimizer.step()
sae.normalize_decoder_weights()      # <-- SUBITO dopo lo step, non prima
```

Nota *dove* è collocata `normalize_decoder_weights()`: **dopo** `optimizer.step()`,
non prima del calcolo della loss. Questo è intenzionale — la normalizzazione è una
*proiezione* sul vincolo (norma unitaria), applicata a posteriori sui pesi appena
aggiornati, non un termine che entra nel calcolo del gradiente. È una tecnica standard
in ottimizzazione vincolata ("projected gradient descent"): si lascia che il gradiente
"spinga" liberamente i pesi, e poi li si "riporta" sul vincolo dopo ogni passo.

```python
x_mean = torch.mean(batch_tokens, dim=0, keepdim=True)
total_ss = torch.sum((batch_tokens - x_mean) ** 2)
residual_ss = torch.sum((batch_tokens - x_hat) ** 2)
r2 = (1.0 - (residual_ss / (total_ss + 1e-8))).item()
l0 = torch.mean(torch.sum(f > 0, dim=-1).float()).item()
```
Qui si vede bene l'idea che **la qualità di una decomposizione si misura su due assi
indipendenti**: quanto bene ricostruisce (`r2`, calcolato esattamente come il
coefficiente di determinazione standard della statistica — varianza spiegata sul totale)
e quanto è sparsa (`l0`, conteggio diretto delle componenti strettamente positive — non
un proxy, il numero vero). Nessuna delle due da sola basterebbe: un SAE che attiva
sempre tutte le 6144 feature avrebbe probabilmente un R² altissimo ma un L0 enorme
(nessuna decomposizione utile); un SAE che ne attiva sempre zero avrebbe L0 minimo ma
R² pessimo (ricostruzione inutile). Il loro equilibrio è la "prova del nove" che il
training sta funzionando.

---

## 4. `interpretability.py` — dare un volto e un nome alle feature

### 4.1 `extract_patch_crop` vs `extract_contextual_crop` — due geometrie correlate

Già viste in dettaglio in [`02_concetti/01_vision_transformer.md`](02_concetti/01_vision_transformer.md) §1.2
e [`02_concetti/04_clip_e_valutazione_crossmodale.md`](02_concetti/04_clip_e_valutazione_crossmodale.md) §3.2.
Qui solo il punto da ricordare: sono **due conversioni indice→pixel coerenti tra loro**
ma con scopi diversi — la prima produce un riquadro fisso 16×16 (per la visualizzazione
"crop" nella griglia di esemplari), la seconda un'area più ampia e centrata, con
clamping ai bordi (per dare un contesto visivo sufficiente a CLIP).

### 4.2 `get_top_activating_patches` — la "ricerca degli esemplari" in azione

```python
patch_tokens = activation[:, 1:, :]
f = sae.encode(patch_tokens)
feature_activation = f[:, :, feature_idx]      # quanto si attiva LA feature scelta, ovunque
for b in range(images.shape[0]):
    for p in range(feature_activation.shape[1]):
        act_val = feature_activation[b, p].item()
        if act_val > 0.0:
            ... costruisci esemplare, inserisci, ordina, tronca a k ...
```

Da notare: si scorre l'**intero dataloader**, immagine per immagine, patch per patch —
e si mantiene una "classifica" delle top-K sempre aggiornata (lista ordinata, troncata
a `k` ad ogni inserimento: `top_exemplars.sort(...); top_exemplars = top_exemplars[:k]`).
È un algoritmo "top-K incrementale" molto semplice ma corretto — con un costo
computazionale che cresce linearmente con la dimensione del dataset esplorato (motivo
per cui, durante test rapidi, si usa `--subset_size` per limitare il numero di immagini
analizzate).

### 4.3 `CLIPAutoLabeler.label_feature` — il cuore dell'etichettatura

Già analizzato a fondo in
[`02_concetti/04_clip_e_valutazione_crossmodale.md`](02_concetti/04_clip_e_valutazione_crossmodale.md) §3.3.
Il dettaglio implementativo che vale la pena ricordare a memoria, perché è facile
dimenticarsene e produce immagini "sbagliate" se saltato: la **de-normalizzazione**

```python
mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
img_unnorm = img_np * std + mean
img_uint8 = (np.clip(img_unnorm, 0, 1) * 255.0).astype(np.uint8)
```

Le immagini nel dataset sono normalizzate (per essere date in pasto al ViT) con le
statistiche di ImageNet — valori che possono uscire dal range [0,1] o [0,255]. Prima di
mostrarle a CLIP (o di salvarle in un PNG), bisogna "invertire" questa trasformazione.
Dimenticarsene produrrebbe immagini con colori distorti — non un errore che blocca
l'esecuzione, ma che **comprometterebbe silenziosamente la qualità delle etichette**
(CLIP vedrebbe colori "sbagliati") e renderebbe le visualizzazioni fuorvianti.

### 4.4 `get_feature_activation_map` + `activation_cache` — un'ottimizzazione mirata

```python
cache_key = (layer_idx, id(image))
if cache_key in activation_cache:
    activation = activation_cache[cache_key]
```

Un dettaglio piccolo ma indicativo di attenzione all'efficienza: quando si genera la
griglia multi-feature (`save_feature_grid_visualization`), per ogni immagine si calcola
**una sola volta** il forward pass del ViT fino al layer di interesse, e si riusa
l'attivazione cacheata per calcolare le heatmap di *tutte* le feature da visualizzare
su quell'immagine — invece di rifarlo da capo per ciascuna. Con 5 feature × 5 esemplari
= 25 potenziali combinazioni per layer, questo accorgimento **riduce di parecchio** il
numero di forward pass necessari per generare un singolo artefatto grafico.

---

## 5. `causal_eval.py` — chiudere il cerchio: dimostrare la causalità

### 5.1 `intervention_callback` — il "bisturi" in 6 righe

```python
cls_token = x[:, 0:1, :]
patch_tokens = x[:, 1:, :]
f = sae.encode(patch_tokens)
f_j = f[:, :, feature_idx]
W_dec_j = sae.W_dec[:, feature_idx]
# ablation:  patch_tokens - α · f_j · W_dec_j      (α = 1 - scaling_factor)
# steering:  patch_tokens + (S - 1) · f_j · W_dec_j (S = scaling_factor)
x_modified = torch.cat([cls_token, patch_tokens_modified], dim=1)
```

Vale la pena notare la simmetria elegante delle due formule — sono letteralmente la
stessa identica operazione (sommare un multiplo di `f_j · W_dec_j` al residual stream),
solo con **segno e scala del coefficiente diversi**:
- ablation: coefficiente `−α ∈ [−1, 0]` (sottrae fino al 100% del contributo)
- steering: coefficiente `(S−1) ∈ [0, +∞)` (aggiunge un multiplo arbitrario del contributo)

Riconoscere questa simmetria è un buon modo per dimostrare di aver capito la logica
profonda — non solo letto il codice riga per riga: **entrambi gli interventi sono
casi speciali di un'unica operazione**, "modifica additivamente l'intensità con cui
questa feature compare nel residual stream", parametrizzata da un singolo coefficiente
con segno.

### 5.2 `evaluate_relative_logit_drop` — confronto controllato baseline vs. ablato

Si noti come il "baseline" venga calcolato con un **forward pass pulito, senza alcun
hook registrato** — non con `scaling_factor=1.0` (che pure, numericamente, lascerebbe
l'attivazione inalterata). È una scelta di igiene sperimentale: il baseline deve essere
"il modello come è sempre stato eseguito", non "il modello con un hook che capita di
non modificare nulla in questo caso specifico" — eliminando ogni possibile dubbio che
la sola presenza dell'hook (per quanto inerte) possa introdurre artefatti numerici
(es. per via di `detach()`, conversioni di tipo, ecc.).

### 5.3 `plot_dose_response` — costruire la curva, intensità per intensità

```python
ablation_strengths = np.linspace(0.0, 1.0, 6)            # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
percent_ablation = (1.0 - ablation_strengths) * 100      # [100%, 80%, 60%, 40%, 20%, 0%]
```

Nota l'inversione: `scaling_factor=0.0` produce ablazione del 100% (perché
`ablation_strength = 1.0 - scaling_factor = 1.0`), mentre `scaling_factor=1.0` produce
ablazione dello 0% (baseline). Il grafico finale, per essere leggibile in modo
intuitivo ("più vado a destra, più ablazione applico"), inverte questa relazione
nell'asse x — un dettaglio di presentazione che vale la pena notare se ti viene chiesto
di interpretare l'asse del grafico: l'asse x del `dose_response_curve.png` mostra
"percentuale di ablazione applicata", non il valore grezzo di `scaling_factor`.

### 5.4 `plot_training_curves` — confrontare la convergenza tra layer

Una funzione di servizio che produce `sae_training_curves.png`: tre pannelli
(loss, R², L0) con una curva per ciascun layer monitorato. Utile per rispondere — con
un grafico alla mano, non solo a parole — a domande come "il SAE del layer 6 converge
più velocemente/più stabilmente di quello del layer 11? hanno raggiunto livelli di
sparsità comparabili?".

---

## 6. `run_pipeline.py` — l'orchestratore: come tutto si mette insieme

`main()` è lungo, ma segue una logica lineare in 10 passi numerati nei commenti del
codice. Ecco i punti che meritano un'attenzione particolare — quelli che racchiudono
**decisioni**, non solo meccanica:

### 6.1 Il loop multi-layer — il cuore comparativo del progetto

```python
layers_to_compare = [5, 10]   # Layer 6 e Layer 11 (indici 0-based)
for i, layer in enumerate(layers_to_compare):
    ... costruisci buffer, SAE, allena, trova feature top, etichetta, valuta causalmente ...
```

Tutto — addestramento del SAE, ricerca degli esemplari, etichettatura CLIP, valutazione
causale — viene **ripetuto identico per ciascun layer**, con un SAE *separato* per
ciascuno (un dizionario di feature è specifico di un layer: le attivazioni a
profondità diverse hanno statistiche e "vocabolari di concetti" diversi). Questo è
ciò che rende possibile, alla fine, popolare `layer_comparison_summary` — la tabella
che permette il confronto diretto tra i due layer.

### 6.2 La selezione automatica della "feature rappresentativa" per il dose-response

```python
if layer == 10:
    if rel_drop > best_layer11_drop:
        best_layer11_drop = rel_drop
        best_layer11_feature_idx = f_idx
...
representative_feat = best_layer11_feature_idx if best_layer11_feature_idx is not None \
                      else (layer11_features[0] if layer11_features else 100)
```

Da notare: la pipeline **non sceglie a caso o per indice fisso** quale feature
sottoporre all'analisi dose-response più approfondita (e alla heatmap finale) — sceglie
automaticamente, fra le top-10 feature più attive del layer 11, **quella con il logit
drop relativo più alto in valore assoluto sull'ablazione totale** — cioè quella che ha
*già dimostrato* di avere il maggiore impatto causale misurabile. È una scelta
metodologicamente solida: concentriamo l'analisi più approfondita (la curva continua)
proprio sul candidato più promettente, quello con la storia causale più forte da
raccontare. Il valore di fallback (`else 100`) è puramente difensivo, per il caso limite
in cui nessuna feature attiva sia stata trovata.

### 6.3 `dataset_concepts` — il vocabolario CLIP è "su misura" per ciascun dataset

```python
dataset_concepts = {
    "imagewoof": ["fur", "eye", "nose", "ear", ... "collie", "labrador"],
    "imagenette": ["fish", "dog", "car", ... "metal texture", "wheel", "scale pattern", "red color"],
    "imagenet":   ["animal", "dog", "bird", ... "honeycomb pattern", "scale pattern"],
    "cifar10":    ["airplane", "automobile", ... "fur", "eye", "metal texture", "stripe"],
}
```

Si nota che ogni lista mescola **due tipi di concetti**: nomi di classi ad alto livello
("dog", "fish", "car"...) e proprietà visive di basso livello, trasversali alle classi
("fur", "scale pattern", "red color", "metal texture"...). Questa scelta non è casuale:
riflette esattamente l'ipotesi di ricerca discussa in
[`01_il_progetto_spiegato.md`](01_il_progetto_spiegato.md) §5.2 — alcune feature
(presumibilmente quelle dei layer più precoci/intermedi) potrebbero corrispondere a
proprietà visive locali, altre (presumibilmente quelle più tardive) a concetti più
vicini alle categorie semantiche finali. Avere **entrambi i tipi** nel vocabolario dà
a CLIP la possibilità di "scegliere il livello di astrazione giusto" per ciascuna
feature — un vocabolario sbilanciato verso un solo tipo avrebbe forzato etichette
fuorvianti per metà delle feature.

### 6.4 `get_top_active_features` — perché non analizziamo TUTTE le 6144 feature

```python
total_activations += f.sum(dim=(0, 1))
mean_activations = total_activations / total_tokens
top_values, top_indices = torch.topk(mean_activations, k=num_features)
```

Una scelta pragmatica e ben motivata: analizzare in dettaglio (esemplari, etichettatura
CLIP, valutazione causale) **tutte** le 6144 feature del dizionario sarebbe
proibitivamente costoso — e inutile, perché la stragrande maggioranza non si attiva mai
(o quasi mai) sui dati che usiamo. Si calcola quindi, per ciascuna feature,
**l'attivazione media su tutti i token osservati**, e si selezionano solo le 10 con la
media più alta — quelle che hanno effettivamente "un ruolo attivo e frequente" nel
comportamento del modello su questo dataset, e che quindi meritano l'analisi
approfondita successiva.

---

## 7. Una "frase di chiusura" per ogni file — da avere pronta a memoria

- **`model_loader.py`**: *"definisce l'interfaccia uniforme verso il backbone, e
  l'unico meccanismo — l'hook con callback — che useremo sia per osservare sia per
  intervenire chirurgicamente sul modello."*
- **`sae.py`**: *"implementa l'architettura encoder-decoder sovracompleta e sparsa,
  con tutti gli accorgimenti raccomandati da Bricken/Cunningham (centratura, vincolo
  di norma unitaria, loss MSE+L1) per garantire che le feature scoperte siano
  davvero monosemantiche."*
- **`caching_and_training.py`**: *"trasforma un flusso di immagini in un flusso di
  milioni di singoli token di attivazione, mescolati e bufferizzati per un
  addestramento stabile e scalabile, e gestisce il loop di ottimizzazione con le
  metriche-spia R²/L0."*
- **`interpretability.py`**: *"trova i migliori esemplari visivi di ciascuna feature,
  li traduce in ritagli di pixel reali, e delega a CLIP — un osservatore esterno e
  indipendente — il compito di tradurli in un'etichetta in linguaggio naturale,
  generando anche le visualizzazioni che rendono tutto questo ispezionabile a occhio."*
- **`causal_eval.py`**: *"trasforma 'questa feature sembra rappresentare un concetto'
  in 'questa feature ha dimostrabilmente un effetto causale, graduale e prevedibile,
  sulla decisione finale del modello' — il livello di evidenza più forte che il
  progetto può fornire."*
- **`run_pipeline.py`**: *"orchestra l'intera pipeline su due layer in parallelo
  concettuale (6 e 11), prendendo decisioni metodologicamente motivate (quali feature
  approfondire, quale vocabolario usare, quale candidato scegliere per l'analisi
  dose-response) e producendo tutti gli artefatti finali — tabelle, CSV, grafici —
  che diventano i 'risultati' del progetto."*
