# Piano di Progetto XAI - Traccia 3: Mechanistic Interpretability nei ViT

**Team di progetto:** Marco, Fabio, Vito
**Obiettivo principale:** Addestramento e analisi di Sparse Autoencoders (SAEs) su modelli Vision Transformer.
**Deliverable:** Report conciso (2-3 pagine) e presentazione di 15 minuti.

---

## Fase 1: Stesura Documentale e Ricerca (Marco e Fabio)
*Pianificazione per la sessione di lavoro odierna su Overleaf. L'obiettivo è completare la prima metà del recap document, definendo in modo inequivocabile il contesto e le motivazioni delle scelte architetturali.*

### 1.1 Impostazione del documento e Introduzione (circa 0.5 pagine)
Questa sezione deve agganciare subito il lettore e definire il perimetro del problema.
- [ ] Creazione della struttura del file `.tex` su Overleaf (titolo, abstract se necessario, e sezioni principali).
- [ ] **Definizione del dominio:** Scrivere un paragrafo introduttivo sui Vision Transformers (ViT) e su come sono diventati uno standard nell'elaborazione visiva, citando il paper fondativo di Dosovitskiy et al.
- [ ] **Il problema dell'interpretabilità:** Introdurre il concetto di "polysemanticity". Spiegare in modo chiaro che i neuroni nei layer MLP dei transformer codificano concetti multipli e sovrapposti, rendendo il modello una black box.
- [ ] **La soluzione proposta:** Definire brevemente cosa sono gli Sparse Autoencoders e come vengono usati nella Mechanistic Interpretability per districare queste rappresentazioni in feature latenti singole e interpretabili (monosemantiche).

### 1.2 Analisi Sistematica della Letteratura (circa 0.75 pagine)
In questa fase è utile dividersi i paper in allegato per estrarre i concetti chiave.
- [ ] **Task per Marco (Focus SAE e LLM):** 
    - Sintetizzare i risultati di Bricken et al. (Towards Monosemanticity). Spiegare come il dictionary learning abbia avuto successo nell'estrarre feature interpretabili dai modelli linguistici.
    - Integrare i concetti di Conmy et al. sull'Automated Circuit Discovery per mostrare come queste feature possano spiegare comportamenti complessi.
- [ ] **Task per Fabio (Focus Vision e ViT):** 
    - Sintetizzare il paper di Raghu et al. (Do Vision Transformers See Like Convolutional Neural Networks?). 
    - Estrarre i concetti chiave su come i ViT aggregano le informazioni spaziali (patch tokens, attenzione globale vs locale) rispetto alle reti convoluzionali tradizionali.
- [ ] **Sintesi congiunta:** Fondere le due parti in un discorso fluido, evidenziando che l'incrocio tra queste due aree (SAE applicati a modelli puramente visivi) è terreno di ricerca molto recente.

### 1.3 Identificazione e Formulazione dei Research Gaps (circa 0.5 pagine)
Questa è la sezione fondamentale per giustificare il lavoro di implementazione che farà Vito. I gap devono essere descritti in modo analitico.
- [ ] **Stesura Gap 1: Disallineamento strutturale e bias verso il testo.** 
    - Argomentare che l'architettura SAE standard è pensata per sequenze di token testuali (word tokens). 
    - Spiegare che i ViT utilizzano patch spaziali 2D con embedding posizionali. 
    - Concludere che la trasposizione dei SAE su questo tipo di struttura spaziale crea sfide non esplorate, specialmente nel modo in cui i concetti visivi si compongono.
- [ ] **Stesura Gap 2: Assenza di metriche e metodologie sistematiche.**
    - Argomentare che, mentre nel testo è facile leggere l'output di un SAE per capire a quale parola reagisce, nelle immagini la questione è più complessa.
    - Sottolineare che la letteratura attuale manca di un'applicazione sistematica dei SAE sui modelli visivi che includa una valutazione cross-modale chiara (es. mappare un'attivazione MLP a uno specifico concetto visivo interpretabile da un umano).

### 1.4 Revisione e Allineamento
- [ ] Rilettura incrociata dei paragrafi per garantire coerenza stilistica e fluidità nel testo.
- [ ] Verifica del limite di lunghezza: tagliare eventuali ridondanze tecniche per rimanere nel limite imposto dalle specifiche del progetto.

---

## Fase 2: Passaggio di Consegne a Vito (Methodology)
*Definizione dei parametri operativi per l'implementazione del codice.*

- [ ] Definire il modello di partenza: ViT-B/16 standard o un modello DINO.
- [ ] Specificare il punto di estrazione delle attivazioni: l'output del layer MLP di un blocco intermedio (ad esempio il layer 6 o 8).
- [ ] Definire l'architettura del SAE (Linear Encoder, ReLU, Linear Decoder) e il fattore di espansione desiderato.
- [ ] Specificare la funzione di loss da implementare: MSE per la ricostruzione più la regolarizzazione L1 sulle attivazioni per garantire la sparsità.

---

## Fase 3: Implementazione (Vito)
*Fase di competenza di Vito.*

- [ ] Setup degli hook in PyTorch per estrarre le attivazioni dal ViT.
- [ ] Implementazione e training loop del SAE sul dataset designato.
- [ ] Esportazione dei pesi del modello addestrato.

---

## Fase 4: Evaluation e Analisi (Marco)
*Fase di validazione sul PC locale una volta ricevuti i pesi da Vito.*

- [ ] Calcolo delle metriche architetturali del SAE (Sparsity e Reconstruction Fidelity).
- [ ] Generazione delle salience map o selezione delle "Top-K" patch per dimostrare la monosemanticity delle singole feature estratte.
- [ ] Produzione dei grafici per il documento e per la presentazione.
