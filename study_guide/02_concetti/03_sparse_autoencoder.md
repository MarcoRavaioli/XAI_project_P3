# Concetto 3 — Sparse Autoencoder (SAE): come "districare" la superposition

> Questo è il cuore tecnico del progetto — il pezzo che ti conviene saper spiegare alla
> lavagna, equazione per equazione, codice alla mano. Ogni riga di
> [`sae.py`](../../src/sae.py) qui sotto è collegata a una scelta progettuale precisa,
> presa dalla letteratura (Bricken 2023, Cunningham 2024).

## 1. L'intuizione di partenza: "decomprimere" invece di "leggere"

Hai visto in [`02_interpretabilita_meccanicistica.md`](02_interpretabilita_meccanicistica.md)
§2.3 che i neuroni di un transformer sono polisemantici perché la rete deve rappresentare
**più concetti di quanti neuroni possieda**, comprimendoli in sovrapposizione
(superposition). Questo "trucco di compressione" funziona solo perché i concetti sono
**sparsi**: in un dato input, quasi sempre solo una piccola minoranza di tutti i concetti
possibili è effettivamente presente/rilevante.

L'idea geniale dei SAE (resa popolare da Bricken et al. 2023 e validata su scala più
ampia da Cunningham et al. 2024) è: **se la compressione sfrutta la scarsità, possiamo
invertirla sfruttando la stessa scarsità**. Costruiamo un modello che:

1. Proietta le attivazioni (768-dim) in uno spazio **molto più ampio** — *sovracompleto*
   — dove c'è "spazio" per dare a ogni concetto la sua direzione individuale, pulita;
2. Forza, allo stesso tempo, **solo poche direzioni alla volta a essere attive** — la
   stessa scarsità presente nei dati originali.

Se questa scommessa funziona (ed empiricamente funziona, sorprendentemente bene), il
risultato è uno spazio dove ogni direzione appresa corrisponde — con buona
approssimazione — a **un solo concetto interpretabile**: una feature *monosemantica*.

> 🧩 **Analogia**: immagina di dover descrivere migliaia di sapori diversi avendo a
> disposizione solo 5 "categorie di gusto" di base (dolce, salato, acido, amaro, umami).
> Ogni sapore complesso sarà necessariamente una *miscela* sovrapposta di queste 5
> categorie — informazione "compressa" e difficile da isolare. Il SAE è come dare a
> quella stessa descrizione **5000 categorie** invece di 5, con la regola "puoi usarne
> solo 2-3 alla volta per descrivere ogni sapore". All'improvviso, ogni categoria può
> diventare specifica e pulita ("nota di vaniglia", "retrogusto di limone candito"...)
> invece di essere un'accozzaglia indistinta.

## 2. L'architettura — riga per riga, equazione e codice insieme

Il file [`sae.py`](../../src/sae.py) implementa esattamente questa idea. Vediamola
pezzo per pezzo, equazione e codice fianco a fianco.

### 2.1 I parametri dimensionali

```python
def __init__(self, d_model: int = 768, expansion_factor: int = 8, tied: bool = False):
    self.d_model = d_model                      # 768 — dimensione delle attivazioni ViT
    self.hidden_dim = d_model * expansion_factor # 768 * 8 = 6144 — spazio "sovracompleto"
```

`d = 768` è la dimensione delle attivazioni MLP del ViT (quella che vogliamo
analizzare); `m = hidden_dim = 6144` è la dimensione del "dizionario" di feature — il
nostro spazio sovracompleto, **8 volte più grande** dell'originale (`expansion_factor =
8`). Più feature candidate ci sono, più "spazio" c'è perché ciascun concetto trovi una
direzione propria, pulita, senza dover condividere con altri.

> ⚠️ **Discrepanza da segnalare** (già annotata in
> [`01_il_progetto_spiegato.md`](../01_il_progetto_spiegato.md) §4.1): il
> `methodology.tex` parla di un dizionario `m = 4d = 3072`, ma il codice realmente
> eseguito usa `expansion_factor = 8` → `m = 6144`. Da allineare prima della consegna.

### 2.2 Encoder — proiettare nello spazio sovracompleto

```python
self.encoder = nn.Linear(d_model, self.hidden_dim, bias=True)   # 768 → 6144
nn.init.zeros_(self.encoder.bias)
```
```python
def encode(self, x):
    x_centered = x - self.b_dec          # centratura (vedi §2.4)
    f = torch.relu(self.encoder(x_centered))
    return f
```

In notazione matematica: `f(x) = ReLU(W_enc · (x − b_dec) + b_enc)`

- `W_enc` ha forma `[6144, 768]`: prende un'attivazione a 768 dimensioni e produce
  6144 "punteggi grezzi" — uno per ogni feature candidata del dizionario.
- **Perché ReLU?** Forza ogni punteggio negativo a zero. Questo è ciò che produce
  davvero la *sparsità*: in un dato istante, solo le feature il cui punteggio grezzo
  supera la soglia implicita (determinata da `b_enc`) restano "accese"; tutte le altre
  sono esattamente zero — non "piccole", **zero**. È una scelta architetturale
  potentissima: la sparsità non è solo "incoraggiata" dalla loss (vedi §2.5), è
  *garantita strutturalmente* dalla non-linearità.

### 2.3 Decoder — ricostruire l'attivazione originale dalla combinazione sparsa

```python
self.decoder = nn.Linear(self.hidden_dim, d_model, bias=False)   # 6144 → 768
nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity="relu")
```
```python
def decode(self, f):
    x_reconstructed = self.decoder(f) + self.b_dec
    return x_reconstructed
```

In notazione matematica: `x̂ = W_dec · f + b_dec`

`W_dec` ha forma `[768, 6144]`: ogni sua **colonna** `W_dec[:, j]` è un vettore a 768
dimensioni — la **"direzione di significato"** della feature `j` nello spazio originale
delle attivazioni. Il decoder ricostruisce l'attivazione come **somma pesata di queste
direzioni**, dove i pesi sono le attivazioni sparse `f` (la maggior parte zero):

```
x̂ = Σⱼ f_j · W_dec[:, j] + b_dec
   = (somma di poche colonne "accese", ciascuna pesata dalla sua intensità)
```

> 🎯 **Questo è ESATTAMENTE ciò che usiamo, peso per peso, negli interventi causali**
> ([`causal_eval.py:55`](../../src/causal_eval.py#L55): `W_dec_j = sae.W_dec[:,
> feature_idx]`): "spegnere la feature j" significa letteralmente sottrarre dal residual
> stream il termine `f_j · W_dec[:, j]` — cioè rimuovere esattamente il contributo che
> quella feature stava aggiungendo alla ricostruzione. Vedi
> [`05_interventi_causali.md`](05_interventi_causali.md) per il dettaglio completo.

### 2.4 La centratura `x - b_dec`: un dettaglio che sembra cosmetico mai non lo è

Hai notato che `encode` sottrae `b_dec` PRIMA di proiettare, e `decode` lo riaggiunge
DOPO? Non è un capriccio stilistico — è una scelta esplicitamente raccomandata da
Bricken et al. (il codice lo cita: *"centering activation by subtracting decoder bias,
as described in Bricken et al. [2023]"*, [`sae.py:63`](../../src/sae.py#L63)).

**Perché serve**: le attivazioni di una rete neurale raramente sono centrate intorno
allo zero — hanno spesso una componente "media" sistematica e condivisa da (quasi) tutti
gli input (pensa a un bias generale del layer). Se non la rimuovessimo, l'encoder
sprecherebbe parte della sua capacità nel "rappresentare la media" anziché concentrarsi
sulle *variazioni specifiche* da un input all'altro — che sono ciò che effettivamente
distingue un concetto dall'altro. Sottraendo `b_dec` (che il modello impara essere
proprio questa componente media) prima di proiettare, l'encoder può concentrarsi su ciò
che conta: le deviazioni interpretabili. Riaggiungerlo nel decode "ripristina" la scala
e l'offset corretti per la ricostruzione finale.

### 2.5 La loss — l'equilibrio tra "ricorda tutto" e "usa poco"

```python
def compute_loss(self, x, x_hat, f, l1_coeff):
    mse_loss = torch.mean((x - x_hat) ** 2)
    l1_loss  = torch.mean(torch.sum(torch.abs(f), dim=-1))
    total_loss = mse_loss + l1_coeff * l1_loss
```

In notazione matematica: `L = MSE(x, x̂) + λ · ||f||₁`

Questa loss mette in tensione **due obiettivi contrastanti**, ed è proprio da questa
tensione che nasce l'interpretabilità:

- **Termine di ricostruzione (MSE)**: "la combinazione di feature che hai scelto deve
  ricostruire fedelmente l'attivazione originale". Da solo, questo termine spingerebbe
  il modello ad attivare *quante più feature possibile* (più informazione disponibile =
  ricostruzione più facile) — tendenza opposta alla sparsità.
- **Termine di sparsità (L1, penalità sulla somma dei valori assoluti delle
  attivazioni)**: "usa il minor numero possibile di feature attive, e con la minor
  intensità possibile". Da solo, questo termine spingerebbe a spegnere tutto
  (ricostruzione pessima, ma "costo zero").
- **`λ` (l1_coeff)**: il "cursore" che bilancia i due. Troppo basso → feature ancora
  polisemantiche (il modello "bara" attivando troppe direzioni insieme); troppo alto →
  ricostruzione povera e feature troppo rare per essere utili.

> 🧩 **Analogia**: è come scrivere un riassunto di un libro con un budget di parole
> limitato. Se hai infinite parole (nessuna penalità di sparsità), puoi essere fedele
> ma prolisso e ridondante — non impari a "isolare i concetti chiave". Se hai
> pochissime parole (penalità troppo alta), sei costretto a essere conciso ma perdi
> informazione essenziale. Il punto giusto ti costringe a trovare *le parole — i
> concetti — giuste*: non un compromesso qualunque, ma quello che cattura l'essenza.

### 2.6 Il vincolo di norma unitaria sul decoder — il "trucco anti-scorciatoia"

```python
@torch.no_grad()
def normalize_decoder_weights(self):
    norms = torch.norm(self.decoder.weight, p=2, dim=0, keepdim=True)
    self.decoder.weight.div_(norms.clamp(min=1e-8))
```

Chiamata dopo **ogni** step di ottimizzatore (`sae.normalize_decoder_weights()` in
[`caching_and_training.py:175`](../../src/caching_and_training.py#L175)). Risolve un
problema sottile ma fatale per la loss L1: il modello potrebbe "barare" rendendo le
colonne di `W_dec` molto **grandi** (alta norma) e, simmetricamente, le attivazioni `f`
molto **piccole** — ottenendo *la stessa ricostruzione* `f · W_dec` ma con una penalità
L1 artificialmente bassa (perché L1 guarda solo `f`, non `W_dec`!). Forzare ogni colonna
del decoder ad avere **norma euclidea = 1** elimina questa scorciatoia: l'unico modo per
abbassare la L1 resta davvero "usare meno feature, con meno intensità" — non
"camuffare" l'intensità nei pesi del decoder.

## 3. Il training — `TokenActivationBuffer` e `train_sae`

### 3.1 Perché serve un "buffer streaming" e non un dataset normale

Il SAE non viene allenato su *immagini*, ma su **singoli token di attivazione** — milioni
di vettori a 768 dimensioni, uno per ogni patch di ogni immagine analizzata, raccolti al
volo durante un forward pass del ViT. Tenerli tutti in RAM contemporaneamente sarebbe
proibitivo. La soluzione, `TokenActivationBuffer`
([`caching_and_training.py:13-120`](../../src/caching_and_training.py#L13)):

1. Passa un batch di immagini nel ViT, intercetta le attivazioni MLP via hook;
2. **Scarta il `[CLS]`** (`activation[:, 1:, :]` — vedi
   [`01_vision_transformer.md`](01_vision_transformer.md) §2.3 per il perché);
3. "Appiattisce" tutte le patch di tutte le immagini del batch in un'unica lista di
   token (`reshape(-1, d_model)`) e la accumula in un buffer fino a una capacità
   prefissata (`buffer_size = 65536` token);
4. **Mescola** (`torch.randperm`) il buffer — passo essenziale per evitare
   *correlazioni intra-batch*: senza questo, token consecutivi verrebbero quasi sempre
   dalla stessa immagine (e quindi statisticamente molto simili), il che
   destabilizzerebbe l'apprendimento e introdurrebbe bias artificiali;
5. Eroga il buffer a "fette" (`sae_batch_size = 1024`) man mano che servono,
   ricaricandolo quando si esaurisce.

> 💡 Questa è proprio l'astrazione "ricicla finché serve, scarta appena si esaurisce"
> — un classico pattern per gestire flussi di dati più grandi della memoria disponibile,
> anche descritto nel docstring come misura "to avoid unrecoverable OOM errors in
> automated runs".

### 3.2 Il loop di training, e cosa significano le metriche che logghiamo

```python
x_hat, f = sae(batch_tokens)
loss_dict = sae.compute_loss(batch_tokens, x_hat, f, l1_coeff)
loss.backward(); optimizer.step()
sae.normalize_decoder_weights()       # rinormalizza SUBITO dopo ogni update
```

Ad ogni step calcoliamo anche due metriche-spia, fondamentali per giudicare se
l'addestramento sta "andando nella direzione giusta" (le ritroverai nelle tabelle dei
risultati, [`01_il_progetto_spiegato.md`](../01_il_progetto_spiegato.md) §5):

- **R² (varianza spiegata)**:
  `r2 = 1 - (Σ(x - x̂)² / Σ(x - x̄)²)`
  — quanto bene la ricostruzione `x̂` "spiega" la variabilità dei dati originali `x`,
  rispetto a una previsione banale (la media `x̄`). Un R² vicino a 1 significa
  ricostruzione quasi perfetta; vicino a 0 (o negativo) significa che il SAE non sta
  imparando a ricostruire nulla di utile.
- **L0 (norma zero — sparsità effettiva)**:
  `l0 = media su tutti i token del numero di feature con f > 0`
  — è la misura *diretta* della sparsità che la loss L1 sta solo *incoraggiando
  indirettamente*. Un L0 = 40, ad esempio, significa "in media, per ogni token, solo 40
  delle 6144 feature disponibili sono attive" — uno spettacolare livello di
  decomposizione, se la ricostruzione resta comunque fedele (R² alto).

> 📌 **Il "punto debole" da saper raccontare con onestà**: R² alto + L0 basso insieme
> sono il segnale che il SAE ha trovato un buon compromesso — sta isolando pochi
> concetti puliti per token, e quei pochi bastano a ricostruire l'attivazione quasi
> perfettamente. Se uno dei due è "fuori scala" (R² basso o L0 troppo alto/basso), è il
> primo indizio di un problema di iperparametri (`l1_coeff`, `lr`, `epochs`) — e va
> discusso onestamente nei risultati, non nascosto.

## 4. Cosa abbiamo, alla fine: un "dizionario di concetti"

Al termine dell'addestramento, abbiamo (per ciascun layer monitorato) un dizionario di
**6144 direzioni candidate** nello spazio delle attivazioni MLP a quel layer, ciascuna
con la proprietà che, quando è attiva, tende a esserlo *da sola o in compagnia di poche
altre* — e a corrispondere (lo verifichiamo nello stadio successivo) a un concetto
visivo coerente. Da qui in poi, il problema diventa: **"queste 6144 direzioni
rappresentano *cosa*, esattamente?"** — la domanda a cui risponde
[`04_clip_e_valutazione_crossmodale.md`](04_clip_e_valutazione_crossmodale.md).

## 5. Glossario rapido di questa sezione

- **Spazio sovracompleto**: spazio di rappresentazione con più dimensioni (feature
  candidate) di quante ne abbia l'originale — `m = 8 · d` nel nostro caso.
- **Feature / direzione del dizionario**: una colonna `W_dec[:, j]`; una direzione
  appresa nello spazio delle attivazioni originali, idealmente corrispondente a un
  concetto.
- **Attivazione sparsa `f`**: il vettore di "intensità" con cui ciascuna feature è
  presente in un dato token — per costruzione, quasi tutte le sue componenti sono zero.
- **L1 sparsity penalty**: termine di loss che penalizza la somma dei valori assoluti
  delle attivazioni, spingendo verso la sparsità.
- **Vincolo di norma unitaria**: normalizzazione delle colonne del decoder a norma 1,
  necessaria per impedire che la rete aggiri la penalità L1 "gonfiando" i pesi.
- **R² / L0**: le due metriche-spia per giudicare, rispettivamente, la fedeltà di
  ricostruzione e il livello di sparsità effettivamente raggiunto.
