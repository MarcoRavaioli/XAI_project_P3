# Concetto 2 — Mechanistic Interpretability: aprire la scatola nera fino in fondo

> Questo è il "cappello filosofico/metodologico" sotto cui sta tutto il progetto.
> Capirlo bene ti permette di rispondere con sicurezza a "ma cos'è esattamente
> l'interpretabilità meccanicistica, e in cosa differisce dalle altre tecniche XAI che
> abbiamo visto a lezione (saliency map, LIME, SHAP, ...)?"

## 1. Le due "scuole" dell'XAI — e perché la nostra è quella più ambiziosa

Semplificando (ma non troppo), l'interpretabilità dei modelli neurali si può dividere in
due approcci con obiettivi molto diversi:

1. **Spiegazioni post-hoc / locali** (saliency maps, Grad-CAM, LIME, SHAP, ...):
   rispondono alla domanda *"quali parti dell'input hanno contato per QUESTA
   predizione?"*. Sono utili e leggere da calcolare, ma **non spiegano il meccanismo**:
   ti dicono "il modello ha guardato qui", non "perché guardando qui arriva a quella
   conclusione" o "che calcolo interno sta eseguendo".
2. **Mechanistic Interpretability**: punta molto più in alto — vuole **reverse-
   engineerizzare** la rete neurale, cioè ricostruire, componente per componente, gli
   *algoritmi* che il modello ha imparato, fino al punto di poterli descrivere come si
   descriverebbe un programma scritto da un umano. L'analogia che usa Elhage et al. nel
   loro paper fondativo (e che è entrata nel lessico comune del campo) è perfetta:

   > "Transformers... possono essere capiti studiando i pesi matriciali del modello in
   > modo simile a come si farebbe reverse engineering di un binario compilato."

   Cioè: la rete neurale è il "binario" — un blob di numeri incomprensibile a prima
   vista — e il mechanistic interpretability researcher è chi prova a "decompilarlo" in
   qualcosa di leggibile, un circuito, un algoritmo, una struttura.

> **Perché questa distinzione conta per la tua presentazione**: il nostro progetto NON
> produce una spiegazione locale ("perché il modello ha classificato QUESTA immagine
> come 'cane'?"). Produce una mappa **globale e riusabile** dei concetti che il modello
> rappresenta internamente ("il modello possiede una direzione interna che codifica
> 'pelo'; un'altra che codifica 'occhio'; ..."), valida per *tutte* le immagini che
> attivano quelle direzioni — un livello di ambizione molto più vicino al reverse
> engineering che alla spiegazione di singole predizioni.

## 2. Il "Mathematical Framework" di Elhage et al. — i tre concetti-cardine

Il paper di Elhage et al. 2021 ("A Mathematical Framework for Transformer Circuits") è
quello che fornisce il **vocabolario e l'impalcatura concettuale** su cui poggia tutto
il resto (incluso il nostro codice — `model_loader.ActivationHook` lo cita
esplicitamente come riferimento). I tre concetti da portare a casa:

### 2.1 Il Residual Stream — "il flusso condiviso su cui tutti scrivono"

Immagina il residual stream come una **lavagna condivisa che passa di mano in mano**
lungo i 12 layer. Ogni componente del modello (ogni testa di attenzione, ogni blocco
MLP):
1. **Legge** quello che è scritto sulla lavagna in quel momento (un'operazione di
   *proiezione lineare* — leggere è semplicemente "guardare in una certa direzione"
   dello spazio a 768 dimensioni);
2. **Calcola** qualcosa sulla base di ciò che ha letto;
3. **Scrive** la sua "proposta" sommandola — non sovrascrivendola! — a ciò che già c'era
   (`x = x + contributo`, la connessione residuale che hai visto in
   [`01_vision_transformer.md`](01_vision_transformer.md) §2).

Questa metafora ha una conseguenza tecnica fortissima, citata esplicitamente nel paper:

> "We think the residual stream as having a kind of 'privileged basis'... it's the
> central object through which all parts of the model communicate."

In pratica: il residual stream **non viene mai cancellato**, solo arricchito. Ogni
informazione scritta da un layer iniziale resta (almeno in parte) accessibile a tutti i
layer successivi — è la base teorica diretta per cui ha senso, ad esempio, **ablare**
chirurgicamente *solo* il contributo di una specifica feature al layer 6: stiamo
letteralmente "cancellando una riga specifica scritta sulla lavagna", lasciando intatto
tutto il resto.

### 2.2 I Circuiti — "sotto-reti che implementano un algoritmo specifico"

Un **circuito** è un sottoinsieme di componenti del modello (head di attenzione, neuroni
MLP, connessioni tra layer) che, lavorando insieme, implementano un comportamento
identificabile e descrivibile (es. il famoso "induction head" che impara la regola
"se hai visto 'A B' prima, e ora vedi di nuovo 'A', predici 'B'"). Elhage et al.
mostrano che — almeno per i blocchi di sola attenzione — si possono "decompilare" i pesi
in due matrici composte interpretabili: la matrice **QK** ("dove guardo" — quale
relazione di posizione/contenuto innesca l'attenzione) e la matrice **OV**
("cosa scrivo se vengo selezionato" — l'effetto che ha sul residual stream).

> **Perché lo citiamo ma non lo implementiamo**: scoprire circuiti completi richiede di
> tracciare interazioni *tra più componenti* attraverso *più layer* — un problema
> enormemente più complesso del nostro. Il nostro progetto si ferma a un livello più
> elementare ma più gestibile: identificare singole **feature monosemantiche** (i
> "mattoncini" su cui un circuito sarebbe costruito) e verificarne il ruolo causale
> isolatamente. È esplicitamente indicato come "direzione futura" nella nostra
> conclusione: "applicare circuit analysis tramite activation patching" sarebbe il passo
> successivo naturale, una volta che hai un dizionario di feature affidabile.

### 2.3 Polysemanticity & Superposition — il problema che ci ha spinto verso i SAE

Qui arriviamo al cuore del problema che il progetto risolve. Elhage et al. (e poi
Bricken et al. in modo più approfondito) osservano che i singoli neuroni di un
transformer sono spesso **polisemantici**: rispondono a un insieme di concetti che, a
un osservatore umano, sembrano completamente scollegati.

> 🔍 **Esempio concreto** (dal paper di Elhage, ripreso anche da Bricken): un singolo
> neurone di un piccolo modello linguistico si attiva fortemente in presenza di:
> *citazioni accademiche*, *dialoghi in inglese*, *richieste HTTP* e *testo in coreano*
> — quattro concetti che non hanno alcuna relazione semantica tra loro.

Perché succede questo? L'ipotesi esplicativa si chiama **superposition**: la rete deve
rappresentare **molti più concetti di quanti neuroni possieda** (pensa a un modello con
solo qualche migliaio di neuroni che deve "conoscere" decine di migliaia di concetti del
mondo). La sua soluzione è "comprimere": invece di assegnare un neurone per concetto
(impossibile, non ce ne sono abbastanza), codifica i concetti come **combinazioni
sovrapposte di direzioni** nello spazio delle attivazioni — un po' come comprimere un
file: si perde un po' di "pulizia" (le direzioni si sovrappongono e si confondono un
po'), ma si guadagna capacità.

> 🧩 **Analogia utile per spiegarlo a voce**: pensa a un armadio con pochi cassetti ma
> tantissimi vestiti. Non puoi dare un cassetto a ciascun capo — devi mischiarli. Il
> "trucco" che rende questo gestibile è che raramente indossi *tutti* i vestiti
> contemporaneamente: di solito te ne servono pochi alla volta. Allo stesso modo, i
> concetti del mondo reale sono **sparsi** (raramente attivi insieme in uno stesso
> input), e questa scarsità è esattamente ciò che rende la "compressione per
> sovrapposizione" possibile senza interferenze catastrofiche.

Questa scarsità nei dati è **anche** il fondamento del perché i SAE funzionano — non è
un caso che i due concetti siano collegati: vedi
[`03_sparse_autoencoder.md`](03_sparse_autoencoder.md) §1 per il filo diretto da
"i concetti sono sparsi nei dati" a "quindi possiamo costringere un modello ausiliario a
*decomprimerli* in uno spazio più ampio dove ognuno ha la sua direzione pulita".

## 3. Perché questo framework "scricchiola" sulle MLP — e perché serviva un'idea nuova

Punto sottile ma importante per la tua narrazione: lo stesso Elhage et al. **ammette
esplicitamente** che il loro framework di "decompilazione in circuiti QK/OV" funziona
bene per i blocchi di attenzione, ma **fatica con le MLP** — proprio a causa della
polisemanticità: i neuroni MLP mescolano troppi concetti per essere letti uno per uno
come "componenti di un circuito". È un problema aperto che il paper lascia esplicitamente
sul tavolo.

La soluzione che è emersa dopo (Cunningham 2024, Bricken 2023 — vedi
[`03_paper/01_bricken_monosemanticity.md`](../03_paper/01_bricken_monosemanticity.md) e
[`03_paper/02_cunningham_sae.md`](../03_paper/02_cunningham_sae.md)) è proprio quella che
adottiamo noi: **non provare a leggere i neuroni grezzi** (sono polisemantici e
illeggibili), ma **riscriverli** — tramite un Sparse Autoencoder — in uno spazio più
ampio e più pulito, dove ogni direzione corrisponde (con buona approssimazione) a un
solo concetto. È il "salto di livello" che rende il nostro progetto possibile: stiamo
applicando, nel dominio visivo, esattamente la soluzione che ha sbloccato
l'interpretabilità delle MLP nei modelli linguistici.

## 4. Una mappa mentale per ricordare come si incastrano i pezzi

```
Elhage 2021         →  vocabolario/teoria: residual stream, circuiti, polisemanticità
                       (ma "si ferma" davanti alle MLP polisemantiche)
        │
        ▼
Bricken / Cunningham → SOLUZIONE: SAE → feature monosemantiche dalle MLP (su LLM)
        │
        ▼
IL NOSTRO PROGETTO   → applichiamo la stessa soluzione, ma su un Vision Transformer:
                       1) i token non sono parole ma patch d'immagine (gap spaziale)
                       2) il modello non "parla" → serve CLIP come traduttore esterno
                          (gap di valutazione cross-modale)
        │
        ▼
Conmy / ACDC         → "next step" naturale (non fatto qui): una volta che hai feature
                       affidabili, puoi provare a tracciare i CIRCUITI che le collegano
```

## 5. Glossario rapido di questa sezione

- **Mechanistic Interpretability**: ricostruzione degli algoritmi interni di una rete
  neurale a un livello di dettaglio "leggibile come codice sorgente" (vs. spiegazioni
  locali post-hoc).
- **Residual stream**: il canale condiviso a cui ogni componente del modello legge e
  scrive, sommando (mai sovrascrivendo) il proprio contributo.
- **Circuito**: sottoinsieme di componenti che, insieme, implementano un comportamento
  identificabile.
- **Polisemanticità**: un singolo neurone che risponde a più concetti scollegati.
- **Superposition**: la strategia con cui una rete "comprime" più concetti di quanti
  neuroni possieda, sfruttando il fatto che i concetti sono sparsi (raramente co-attivi).
- **Monosemanticità**: la proprietà — desiderabile, e che i SAE permettono di ottenere —
  per cui una direzione/feature corrisponde a un solo concetto coerente.
