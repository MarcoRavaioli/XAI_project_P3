# Concetto 5 — Interventi causali: dimostrare che le feature "contano davvero"

> Questo è l'ultimo anello della catena: senza questa fase, il progetto si fermerebbe a
> "abbiamo trovato delle direzioni che SEMBRANO rappresentare dei concetti" — un'
> affermazione *correlazionale*, debole. Con questa fase, possiamo dire "abbiamo
> verificato che queste direzioni hanno un *ruolo causale* nelle decisioni del modello"
> — un'affermazione molto più forte, e molto più interessante da un punto di vista
> scientifico.

## 1. Perché la correlazione non basta — il problema che questa fase risolve

Immagina di aver trovato che la feature 4049 si attiva fortemente su superfici rosse, e
CLIP la etichetta "red color" con score alto e consistente. Sembra una bella scoperta —
ma è davvero una prova che il modello *usa* quella direzione per le sue decisioni?
Potrebbe darsi che quella direzione sia semplicemente "lì", una rappresentazione
collaterale che il modello ha sviluppato ma che **non influenza in alcun modo** le sue
predizioni finali — un epifenomeno, non un meccanismo. La sola correlazione
("si attiva quando vedo X") **non distingue** questi due scenari.

L'unico modo per sciogliere questo dubbio è **intervenire attivamente**: modificare
*solo* quella direzione, lasciando tutto il resto invariato, e osservare se la
predizione del modello cambia in modo coerente. Se sì — abbiamo prova *causale*, non
solo correlazionale, che quella direzione conta. Questa logica ("manipola una variabile
alla volta, osserva l'effetto") è il fondamento di ogni esperimento scientifico
controllato — qui applicato non a un sistema fisico, ma a una rete neurale.

> 🧩 **Analogia**: è la differenza tra notare che "ogni volta che il semaforo è verde,
> le auto passano" (correlazione — ma magari le auto passerebbero comunque, magari il
> semaforo è scollegato) e *spegnere fisicamente il semaforo* per vedere se le auto
> smettono davvero di passare in modo coordinato (intervento causale — ora sai se il
> semaforo *governa* il traffico o è solo un osservatore passivo).

## 2. I due interventi: ablation e steering — due facce della stessa medaglia

Il file [`causal_eval.py`](../../src/causal_eval.py) implementa due tipi di
manipolazione **opposti e complementari**, entrambi realizzati nella stessa funzione
`perform_causal_intervention` tramite un unico hook con callback:

### 2.1 Ablation — "spegnere" una feature

```python
ablation_strength = 1.0 - scaling_factor
patch_tokens_modified = patch_tokens - ablation_strength * f_j.unsqueeze(-1) * W_dec_j.view(1, 1, -1)
```

In notazione matematica: `x_ablated = x_patches − α · f_j(x) · W_dec[:, j]`,
con `α = ablation_strength ∈ [0, 1]`

Cosa stiamo facendo, letteralmente: ricordi dal SAE
([`03_sparse_autoencoder.md`](03_sparse_autoencoder.md) §2.3) che la ricostruzione è
`x̂ = Σⱼ f_j · W_dec[:, j] + b_dec`, cioè una somma di "contributi", uno per feature
attiva? Questa operazione **sottrae esattamente il contributo della feature `j`** da
quella somma — proiettato di nuovo nello spazio originale a 768 dimensioni, e applicato
direttamente al residual stream (non alla ricostruzione del SAE, al *vero* flusso del
modello). È un intervento **chirurgico**: non tocchiamo nient'altro — non le altre
feature, non il `[CLS]`, non gli altri layer. Rimuoviamo *solo e soltanto* quella
specifica quantità di quel specifico concetto.

`α = 1.0` (cioè `scaling_factor = 0.0`) è l'**ablazione totale**: rimuoviamo tutta
l'attivazione misurata. `α = 0.5` lascia metà del contributo intatto. `α = 0.0`
(`scaling_factor = 1.0`) non modifica nulla — è la condizione di **baseline**, usata
come riferimento per misurare l'effetto.

### 2.2 Steering — amplificare una feature

```python
patch_tokens_modified = patch_tokens + (scaling_factor - 1.0) * f_j.unsqueeze(-1) * W_dec_j.view(1, 1, -1)
```

In notazione matematica: `x_steered = x_patches + (S − 1) · f_j(x) · W_dec[:, j]`,
con `S = scaling_factor`

Qui facciamo l'opposto: invece di rimuovere il contributo della feature, lo
**potenziamo artificialmente**. Con `S = 2.0`, il termine aggiunto è esattamente
`+1 · f_j · W_dec[:, j]` — cioè raddoppiamo l'intensità con cui quella feature compare
nel residual stream (il contributo passa da `f_j · W_dec_j` a `2·f_j · W_dec_j`). Con
`S = 1.0`, il termine aggiunto è zero — di nuovo la condizione di baseline.

> 💡 **Perché valutare ENTRAMBE le direzioni di manipolazione, e non solo l'ablazione?**
> Perché insieme formano un test molto più stringente. Se la feature 4049 rappresenta
> davvero "rosso", *spegnerla* dovrebbe abbassare il logit della classe "fragola" (se
> è presente del rosso nell'immagine), e *amplificarla* dovrebbe alzarlo — entrambi gli
> effetti, nella direzione attesa, sono molto più convincenti di uno solo. Un effetto
> in una sola direzione potrebbe essere un artefatto numerico (es. saturazione); un
> effetto bidirezionale e coerente è una firma molto più forte di causalità reale.

### 2.3 Il `[CLS]` resta intoccato — perché, di nuovo

```python
cls_token = x[:, 0:1, :]
patch_tokens = x[:, 1:, :]
# ... manipolazione solo su patch_tokens ...
x_modified = torch.cat([cls_token, patch_tokens_modified], dim=1)
```

Vale la pena ribadirlo qui (l'avevi già visto in
[`01_vision_transformer.md`](01_vision_transformer.md) §2.3): l'intervento agisce
**esclusivamente sui token spaziali**. Il `[CLS]` viene "tagliato fuori", lasciato
identico, e ricucito alla fine. Questo isola la domanda sperimentale a "cosa succede se
*questa specifica informazione spaziale* viene alterata, prima che si propaghi (negli
strati successivi) fino al `[CLS]` e quindi alla predizione finale?" — una domanda ben
definita e interpretabile. Toccare anche il `[CLS]` mescolerebbe l'effetto della
manipolazione con un'alterazione diretta dell'aggregatore globale, rendendo
l'interpretazione del risultato molto più ambigua.

## 3. Come misuriamo l'effetto: la Relative Logit Drop

```python
relative_drop = (baseline_logit - ablated_logit) / (abs(baseline_logit) + 1e-8)
```

In notazione matematica: `RLD = (L_baseline − L_ablated) / |L_baseline|`

`evaluate_relative_logit_drop` confronta il **logit della classe target** (es. la classe
"cane" se la feature in esame sembra rappresentare "occhio di cane") *prima*
dell'intervento (`L_baseline`, ottenuto con un forward pass normale, senza hook) e
*dopo* l'ablazione totale (`L_ablated`, ottenuto re-iniettando lo stesso input ma con
`perform_causal_intervention(..., scaling_factor=0.0)`). Il rapporto **normalizza** la
differenza assoluta rispetto alla scala del logit di partenza — rendendo la metrica
confrontabile tra classi/immagini/feature diverse, che potrebbero avere logit baseline
di magnitudine molto differente. Un `RLD` alto e positivo significa: "rimuovere questa
feature fa crollare la fiducia del modello in quella classe — la feature è
*causalmente importante* per quella predizione".

## 4. La dose-response curve — la prova più convincente di tutte

`plot_dose_response` è, a mio parere, **l'esperimento più elegante** di tutta la
pipeline, perché trasforma un singolo punto di misura ("ablazione totale → drop X%") in
una **curva continua**, con un potere esplicativo molto più forte:

```python
ablation_strengths = np.linspace(0.0, 1.0, 6)   # 0%, 20%, 40%, 60%, 80%, 100% di ablazione
for strength in ablation_strengths:
    logits_intervened = perform_causal_intervention(..., scaling_factor=strength)
    # ... calcola il logit drop relativo per ciascun livello ...
```

L'idea, mutuata direttamente dalla farmacologia/tossicologia (da cui il nome
"dose-response": "più alta è la dose di un farmaco, più forte è l'effetto, *fino a un
certo punto*"): variamo con continuità l'**intensità** dell'intervento — da "non tocco
nulla" (`strength=1.0` → 0% ablazione) a "rimuovo tutto" (`strength=0.0` → 100%
ablazione) — e osserviamo come cambia l'effetto sul logit. Cosa ci dice la forma della
curva risultante:

- **Curva monotona e graduale** (il logit drop cresce proporzionalmente all'intensità
  dell'ablazione): è la firma di un **meccanismo causale reale e continuo** — la
  feature contribuisce "a gradi" alla decisione, esattamente come ci si aspetterebbe da
  una vera direzione semantica nel residual stream. È l'evidenza più forte possibile a
  sostegno dell'interpretazione data alla feature.
- **Curva piatta o rumorosa** (il logit drop non cambia in modo sistematico
  all'aumentare dell'intensità): segnale che l'effetto osservato con l'ablazione totale
  potrebbe essere **un artefatto** — magari dovuto a un effetto di soglia non lineare
  nel modello, o alla rottura di un equilibrio interno del residual stream, più che a
  una rimozione "pulita" di un concetto specifico. Anche questo è un risultato da saper
  discutere onestamente, non da nascondere.

> 📌 **Suggerimento per la presentazione**: se hai una curva ben monotona da mostrare,
> è probabilmente **il singolo grafico più convincente** di tutta la presentazione —
> più ancora della griglia di esemplari CLIP — perché dimostra il livello più alto di
> evidenza che il progetto può fornire: non solo "interpretabile", ma "causalmente
> rilevante e in modo graduale e prevedibile". Vale la pena dedicarle tempo extra in
> fase di analisi.

## 5. Come tutto questo si incastra in un singolo meccanismo elegante: l'hook con callback

Vale la pena notare — è un bel punto da citare se ti chiedono "come è strutturato il
codice?" — che **tutta** questa logica di intervento (ablation, steering, baseline) è
realizzata riusando esattamente la stessa infrastruttura `ActivationHook` già vista
per la cattura delle attivazioni ([`model_loader.py:34-80`](../../src/model_loader.py#L34)),
semplicemente passandole una funzione di callback:

```python
hook = ActivationHook(submodule, callback=intervention_callback)
hook.register()
with torch.no_grad():
    outputs = model_wrapper.model(images)   # il forward pass "normale" del ViT...
    logits = outputs.logits                  # ...ma internamente, ad ogni passaggio per
hook.remove()                                 # il sotto-blocco MLP del layer scelto, il
                                              # callback intercetta e modifica l'output
```

Guarda bene `ActivationHook.hook_fn`
([`model_loader.py:53-67`](../../src/model_loader.py#L53)): cattura SEMPRE
l'attivazione (`self.activation = actual_output.detach()`), e SE è stato fornito un
callback, **sostituisce** l'output del modulo con il risultato del callback —
altrimenti lo lascia passare inalterato. Questa singola astrazione serve quindi **due
scopi diversi** a seconda di come viene istanziata: "sola lettura" (cattura per
addestrare il SAE o cercare esemplari, `callback=None`) e "lettura + scrittura"
(intervento causale, `callback=intervention_callback`) — proprio l'idea di "leggere e
scrivere sul residual stream" presa da Elhage et al. e citata esplicitamente nel
docstring di `ActivationHook`.

## 6. Glossario rapido di questa sezione

- **Intervento causale / chirurgico**: modificare attivamente una singola componente
  interna del modello (qui: una feature SAE) per osservarne l'effetto sull'output,
  invece di limitarsi a osservarne le correlazioni.
- **Ablation**: "spegnimento" di una feature — sottrazione del suo contributo dal
  residual stream, parametrizzata da un'intensità `α ∈ [0,1]`.
- **Steering**: amplificazione artificiale dell'attivazione di una feature, tramite
  aggiunta del suo contributo scalato da un fattore `S`.
- **Relative Logit Drop (RLD)**: metrica normalizzata che misura quanto crolla (in
  proporzione) il logit della classe target dopo l'ablazione totale di una feature.
- **Dose-response curve**: grafico dell'effetto causale (RLD) in funzione dell'intensità
  dell'intervento — una curva monotona e graduale è la prova più forte di un meccanismo
  causale reale.
- **Hook con callback**: pattern implementativo che intercetta E modifica
  un'attivazione a runtime, durante il forward pass — il "bisturi" che rende possibile
  l'intervento chirurgico.
