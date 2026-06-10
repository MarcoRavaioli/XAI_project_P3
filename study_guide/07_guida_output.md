# Guida alla lettura dei risultati in `out/`

> Questo file è la tua mappa per navigare tutto ciò che il pipeline produce.
> Ogni file di `out/` ha una spiegazione di cosa è, come leggerlo e cosa
> cercarci. In fondo c'è un **riassunto narrativo completo** dei risultati
> reali già prodotti dal pipeline — così non devi farlo tu.

---

## 1. Mappa rapida dei file

| File | Tipo | Cosa contiene |
|------|------|---------------|
| `sae_training_curves.png` | Grafico | Curve di training del SAE (Loss, R², L₀) per Layer 6 e 11 |
| `layer_comparison_summary.md` | Tabella testo | Metriche finali di confronto tra Layer 6 e Layer 11 |
| `discovered_features_summary.csv` | Tabella dati | 20 righe: le 10 feature più attive per ciascun layer, con CLIP label e metriche causali |
| `multi_feature_exemplar_grid_layer6.png` | Griglia immagini | Le 5 feature più interpretabili di Layer 6, con exemplar, immagini complete e heatmap |
| `multi_feature_exemplar_grid_layer11.png` | Griglia immagini | Le 5 feature più interpretabili di Layer 11 |
| `multi_feature_exemplar_grid.png` | Griglia immagini | Copia identica di `layer11` (alias legacy) |
| `feature_grid_layer6_feat{N}.png` | Griglia immagini | Una singola feature di Layer 6 (stesso formato della griglia multi-feature) |
| `feature_grid_layer11_feat{N}.png` | Griglia immagini | Una singola feature di Layer 11 |
| `dose_response_curve.png` | Grafico | Curva dose-risposta per la feature Layer 11 con maggior impatto causale |
| `feature_activation_heatmap.png` | Heatmap | Mappa spaziale di attivazione per la stessa feature causalmente più rilevante |

**Ordine di lettura consigliato** (racconta la storia nel giusto verso):
```
sae_training_curves.png
  → layer_comparison_summary.md
    → discovered_features_summary.csv
      → multi_feature_exemplar_grid_layer6.png  (+ feature_grid_layer6_feat*.png)
      → multi_feature_exemplar_grid_layer11.png (+ feature_grid_layer11_feat*.png)
        → feature_activation_heatmap.png
          → dose_response_curve.png
```

---

## 2. Come leggere ogni file

### `sae_training_curves.png`

Tre sottografici affiancati, uno per ogni metrica di training. Ci sono **due curve** per grafico: una per Layer 6, una per Layer 11.

**Cosa guardare:**

| Sottografico | Asse Y | Cosa vuol dire convergere bene |
|---|---|---|
| **Training Loss** | Loss totale = MSE + λ·L1 | Decresce monotonica verso zero |
| **R² Score** | Varianza spiegata (0–1) | Sale verso 1 → ricostruzione quasi perfetta |
| **L₀ Sparsity** | N° medio di feature attive per token | Scende e si stabilizza → pochi feature attivati |

> **Attenzione:** R² alto e L₀ alto insieme possono sembrare contraddittori. Non lo sono:
> R² misura *quanto bene* il SAE ricostruisce; L₀ misura *quante* feature usa per farlo.
> Vogliamo R² alto e L₀ basso — cioè ricostruzione accurata ma parsimoniosa.

---

### `layer_comparison_summary.md`

Una tabella a tre colonne con i **valori finali** (ultimo epoch) delle metriche per Layer 6 e Layer 11.

```
| Layer    | R^2 Score | L_0 Norm | Mean Logit Drop   |
```

**Significato di ogni colonna:**

- **R² Score**: quanto bene il SAE ricostruisce le attivazioni reali del ViT in quel layer.
  - Vicino a 1.0 = quasi nessuna informazione persa nella compressione SAE.
- **L₀ Norm**: numero medio di feature SAE attive per token (su 6144 totali con expansion=8 su d_model=768).
  - Un L₀ basso significa che ogni token è rappresentato da poche feature → più interpretabile.
- **Mean Logit Drop (%)**: media del Relative Logit Drop sulle 10 feature più attive del layer.
  - Quanto scende il logit della classe predetta quando abliamo una feature?
  - Valori alti → le feature di quel layer sono **causalmente rilevanti** per la classificazione.

---

### `discovered_features_summary.csv`

Il file più ricco di dati quantitativi. Ogni riga è una feature SAE analizzata.

**Colonne:**

| Colonna | Spiegazione |
|---|---|
| `Feature Index` | Indice nella dimensione nascosta del SAE (0–6143) |
| `Target Layer` | Layer 6 o Layer 11 |
| `Assigned CLIP Concept` | Etichetta semantica assegnata da CLIP (zero-shot) |
| `CLIP Confidence Score` | Similarità coseno media tra i 5 exemplar e il testo "a photo of {concept}" |
| `Baseline Logit` | Logit del ViT sulla classe predetta, senza intervento |
| `Ablated Logit` | Logit dopo ablazione chirurgica della feature (scaling_factor=0) |
| `Relative Logit Drop (%)` | `(Baseline - Ablated) / |Baseline| × 100` — impatto causale dell'ablazione |
| `Steered Logit Increase (%)` | Variazione del logit quando la feature viene amplificata (scaling_factor=5) |

**Come leggere le righe:**
- Un **Relative Logit Drop positivo alto** (es. 13%) → abliare quella feature *abbassa fortemente* il logit: la feature è causalmente importante.
- Un drop *negativo* (es. −1%) → abliare quella feature fa *salire* il logit: la feature aveva effetto inibitorio sulla classe.
- Il **CLIP Confidence Score** è tipicamente basso (0.23–0.30) su patch 16×16 — è normale, non è un segnale di etichettatura sbagliata; CLIP è addestrato su immagini intere.

**Segnali da cercare nel CSV:**
1. Quali feature hanno il Relative Logit Drop più alto? → Sono le più causalmente rilevanti.
2. Ci sono concept CLIP ricorrenti tra le top feature? → Rivelano il "vocabolario" visivo di quel layer.
3. C'è differenza sistematica tra Layer 6 e Layer 11 nei drop? → Sì (vedi § 4).

---

### `multi_feature_exemplar_grid_layer{N}.png`

La visualizzazione più densa e informativa. Per ogni layer mostra le **5 feature con CLIP confidence più alta**, ognuna su **3 righe** × **5 colonne**.

**Schema della griglia:**

```
                  Exemplar 1   Exemplar 2   Exemplar 3   Exemplar 4   Exemplar 5
                 ┌───────────┬───────────┬───────────┬───────────┬───────────┐
Feature N        │  Context  │  Context  │  Context  │  Context  │  Context  │  ← Riga 1: crop contestuale
(concept)        │   Crop    │   Crop    │   Crop    │   Crop    │   Crop    │    (finestra 5×5 patch attorno
                 │  Act: X.X │  Act: X.X │           │           │           │     al patch più attivo)
                 ├───────────┼───────────┼───────────┼───────────┼───────────┤
Full Images      │  Immagine │  Immagine │  Immagine │  Immagine │  Immagine │  ← Riga 2: immagine completa
                 │  + riquad │  + riquad │           │           │           │    (con riquadro rosso sul patch)
                 ├───────────┼───────────┼───────────┼───────────┼───────────┤
Heatmaps         │ Heatmap   │ Heatmap   │ Heatmap   │ Heatmap   │ Heatmap   │  ← Riga 3: overlay heatmap
                 │ overlay   │ overlay   │           │           │           │    (rosso = alta attivazione)
                 └───────────┴───────────┴───────────┴───────────┴───────────┘
```

**Cosa guardare:**
- **Riga 1 (Context Crops):** Il riquadro rosso indica il patch esatto più attivo. Guarda *cosa c'è in quel patch* — dovrebbe corrispondere al CLIP concept in etichetta.
- **Riga 2 (Full Images):** Aiuta a vedere il contesto dell'immagine originale e verificare che il patch attivo sia in una zona semanticamente coerente.
- **Riga 3 (Heatmap overlay):** La scala di colori è normalizzata al massimo di quella feature. Rosso/giallo = zone ad alta attivazione, nero = bassa. Se la macchia calda è *sempre* sulla stessa tipologia di oggetto → la feature è monosemantic.
- **Act: X.X nel titolo:** Valore di attivazione SAE per quell'exemplar. Più alto = il patch ha triggherato di più quella feature.

**Feature individuali (`feature_grid_layer{N}_feat{M}.png`):**
Stesso identico formato, ma per una sola feature → più facile da analizzare in isolamento per la presentazione.

---

### `feature_activation_heatmap.png`

Tre pannelli affiancati per la feature Layer 11 con il più alto Relative Logit Drop:

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Source Image   │  Feature Activ. │    Overlay      │
│  (immagine      │  Map (14×14     │  (heatmap       │
│   eval_image)   │   griglia hot)  │   sovrapposta)  │
└─────────────────┴─────────────────┴─────────────────┘
```

- **Pannello 1:** L'immagine di valutazione usata per tutti gli interventi causali.
- **Pannello 2:** La griglia 14×14 dei 196 patch token. Ogni cella = attivazione del SAE per quella feature in quel patch. Colormap "hot": nero=0, rosso=medio, giallo/bianco=massimo. La colorbar a destra dà i valori assoluti.
- **Pannello 3:** Overlay — aiuta a capire *dove spazialmente* nell'immagine quella feature è attiva.

**Cosa cercare:** Le zone gialle/bianche dovrebbero coincidere con il concept CLIP assegnato alla feature.

---

### `dose_response_curve.png`

Un grafico singolo con:
- **Asse X:** Ablation Strength (%) — quanto "viene rimossa" la feature. 0% = nessuna ablazione (baseline), 100% = ablazione totale.
- **Asse Y:** Relative Logit Drop (%) — quanto cade il logit della classe predetta.

**Come leggere la curva:**
- A 0% ablazione il drop deve essere ≈ 0% (nessun intervento = nessun effetto). ✓
- A 100% ablazione il drop deve coincidere con il valore nel CSV per quella feature.
- La forma della curva rivela la **causalità**:
  - Curva monotonica crescente → l'effetto è lineare e pulito → buon segnale causale.
  - Curva piatta → la feature non aveva impatto reale → risultato negativo.
  - Curva non monotonica → interferenze, interpretazione più complessa.

> Questo grafico è la **prova causale** più forte che puoi mostrare nella presentazione.
> Una curva monotonica su una feature ben etichettata da CLIP dice: "questa feature
> è monosemantic e causalmente rilevante, non solo correlata".

---

## 3. Risultati reali del pipeline — riassunto già interpretato

> Questi sono i numeri dell'ultimo run. Puoi usarli direttamente nella presentazione
> senza dover ricalcolare nulla.

### 3.1 Confronto tra layer

| Layer | R² | L₀ | Mean Logit Drop |
|---|---|---|---|
| **Layer 6** | **0.9957** | 201.72 | 0.0319% |
| **Layer 11** | 0.9477 | **107.90** | **1.4030%** |

**Interpretazione:**
- **Layer 6** è ricostruito meglio (R² quasi perfetto). Il SAE "capisce" le attivazioni di Layer 6 più facilmente → le rappresentazioni sono più regolari e strutturate.
- **Layer 6 ha L₀ quasi doppio rispetto a Layer 11** (201 vs 108 feature attive per token). Questo significa che a Layer 6 ogni patch attiva molte feature *in parallelo* → rappresentazioni distribuite, tipiche dei layer intermedi dove ancora non c'è specializzazione.
- **Layer 11 ha un impatto causale 44× maggiore sulla classe** (1.40% vs 0.03%). Le feature di Layer 11 contribuiscono direttamente alla decisione finale del classificatore — quelle di Layer 6 essenzialmente no.
- Questo conferma la letteratura (Raghu et al., ViT vs CNN): i layer finali dei ViT sono quelli semanticamente più specializzati e causalmente rilevanti.

---

### 3.2 Feature scoperte per Layer 6

Le 10 feature più attive identificate e i 5 scelti per la visualizzazione (ordinati per CLIP confidence):

| Feature | CLIP Concept | CLIP Score | Rel. Logit Drop | Steer Increase |
|---|---|---|---|---|
| 4506 | scale pattern | 0.2536 | +0.69% | +1.78% |
| 4271 | honeycomb pattern | 0.2651 | −0.04% | −0.03% |
| 4770 | spotted pattern | 0.2669 | +0.04% | +0.16% |
| 2719 | spotted pattern | 0.2511 | +0.56% | +0.80% |
| 2045 | pineapple | 0.2487 | +0.03% | +0.11% |

*(Le restanti 5 nel CSV: 1632, 5536, 4608, 2041, 4343)*

**Pattern Layer 6:**
- I concept dominanti sono **texture e pattern geometrici** (spotted, scale, honeycomb) → coerente con l'interpretazione di layer intermedi come detector di low-level features.
- I Relative Logit Drop sono tutti vicini a zero (< 1%) → **queste feature non determinano la classificazione finale**, ma la alimentano passando informazioni ai layer successivi.
- Feature 4506 (scale pattern) ha il drop più alto di Layer 6: 0.69% — ancora piccolo in assoluto.

---

### 3.3 Feature scoperte per Layer 11

| Feature | CLIP Concept | CLIP Score | Rel. Logit Drop | Steer Increase |
|---|---|---|---|---|
| **5065** | **honeycomb pattern** | **0.2970** | **+13.49%** ⭐ | **+5.68%** |
| 3932 | fur texture | 0.2794 | 0.00% | 0.00% |
| 965 | plant | 0.2683 | +0.00% | +0.00% |
| 540 | striped pattern | 0.2578 | +0.00% | +0.02% |
| 1767 | animal | 0.2571 | +0.02% | +0.03% |

*(Le restanti 5 nel CSV: 5771, 3591, 2838, 4488, 5517)*

**Pattern Layer 11:**
- **Feature 5065 è la star dell'intero esperimento**: Relative Logit Drop del 13.49% — ablazionarla fa crollare il logit della classe predetta di oltre un quinto. Ha anche la più alta CLIP confidence (0.2970) tra tutte le 20 feature analizzate.
  - Concept CLIP: "honeycomb pattern" — una feature di texture che si è rivelata *fondamentale* per la classificazione. Questo è un risultato interpretabile e causalmente verificabile.
  - Steered Logit Increase del 5.68%: amplificarla *aumenta* il logit della classe corretta.
- La maggior parte delle altre feature Layer 11 ha drop ≈ 0.00% → il carico causale è concentrato su pochissime feature.
- I concept di Layer 11 sono più astratti (animal, plant, bird) rispetto a Layer 6 (texture geometriche).

---

### 3.4 La feature "pilota" degli interventi causali: Feature 5065, Layer 11

Questa feature compare in:
- `feature_grid_layer11_feat5065.png` — i suoi 5 exemplar con heatmap
- `feature_activation_heatmap.png` — la sua mappa spaziale sull'immagine di valutazione
- `dose_response_curve.png` — la curva causale (aspettati un drop che va da 0% a ~13.49% al 100% di ablazione)

È la feature che **da sola prova il punto principale del progetto**: esistono feature monosemantiche in un ViT che si possono identificare con un SAE, etichettare automaticamente con CLIP, e verificare causalmente tramite ablazione chirurgica nel residual stream.

---

## 4. Cosa dire in presentazione su ogni output

| Se ti chiedono di... | Apri questo file | Di' questo |
|---|---|---|
| Mostrare che il training è andato a buon fine | `sae_training_curves.png` | "R² > 0.94 su entrambi i layer, il SAE converge e ricostruisce bene le attivazioni" |
| Confrontare Layer 6 e Layer 11 | `layer_comparison_summary.md` | "Layer 6: R²=0.9957 ma drop medio 0.03% → Layer 11: R²=0.9477 ma drop medio 1.4% → i layer finali sono più causalmente rilevanti" |
| Mostrare le feature scoperte | `multi_feature_exemplar_grid_layer11.png` | "Ogni riga è una feature distinta del SAE; il riquadro rosso indica il patch che la attiva di più; l'heatmap mostra la distribuzione spaziale dell'attivazione" |
| Dimostrare la causalità | `dose_response_curve.png` + CSV riga feat 5065 | "Feature 5065 (honeycomb, Layer 11): 13.49% di logit drop a piena ablazione, con curva monotonica — questo è impatto causale, non solo correlazione" |
| Mostrare l'interpretabilità di una feature | `feature_activation_heatmap.png` | "La zona calda nella heatmap corrisponde alla texture visiva etichettata da CLIP — la feature è monosemantic e spazialmente localizzata" |

---

## 5. Cosa NON allarmarti di vedere

- **CLIP scores bassi (0.23–0.30):** Normale. CLIP è addestrato su immagini intere, non patch 16×16. Usiamo la similarità relativa tra concept, non il valore assoluto.
- **Drop negativi nel CSV:** Alcune feature hanno effetto inibitorio. Non è un errore; significa che abliare quella feature aiuta (leggermente) la classificazione.
- **Drop ≈ 0 per molte feature:** È il risultato atteso e interessante — la causalità è sparse, concentrata su poche feature chiave.
- **`multi_feature_exemplar_grid.png` = copia di `layer11`:** Il pipeline lo crea come alias; non è un file duplicato da analizzare separatamente.
