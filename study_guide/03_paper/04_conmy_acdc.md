# Paper 4 — Conmy et al. 2023, "Towards Automated Circuit Discovery for Mechanistic Interpretability" (ACDC)

**Citazione**: Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., Garriga-Alonso,
A. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability*.
NeurIPS 2023. (`Conmy2023` in [`references.bib`](../../paper/Your_Paper_Title_Here/references.bib))

> ⚠️ **Promemoria importante**: questo paper è citato nel nostro `related_work.tex`
> come riferimento concettuale e metodologico — **NON è ciò che il nostro progetto
> implementa**. È fondamentale saper distinguere chiaramente i due livelli (vedi §4).
> Confonderli è l'errore più probabile in sede di domande.

## 1. Il problema che affronta

Trovare un **circuito** (vedi [`02_concetti/02_interpretabilita_meccanicistica.md`](../02_concetti/02_interpretabilita_meccanicistica.md)
§2.2 per la definizione) — un sottoinsieme di componenti del modello che implementa un
comportamento specifico — è, nei lavori pionieristici di mechanistic interpretability,
un processo fatto **a mano**: i ricercatori ipotizzano dove potrebbe trovarsi il
circuito, lo testano, lo raffinano, in un processo lento e che richiede moltissima
competenza specialistica. Conmy et al. si chiedono: **si può automatizzare questo
processo di scoperta?**

## 2. L'idea centrale — potare il grafo computazionale come si pota un albero

ACDC (Automated Circuit DisCovery) tratta l'intero modello come un **grafo
computazionale**: nodi = componenti (head di attenzione, MLP...), archi = connessioni
attraverso cui l'informazione fluisce da un componente all'altro. L'algoritmo parte dal
grafo completo e lo **pota iterativamente**: per ogni connessione, verifica — tramite
*activation patching* (vedi sotto) — quanto "conta" davvero per il comportamento che si
sta studiando; se l'effetto della sua rimozione è trascurabile, la connessione viene
eliminata dal grafo. Il risultato finale, dopo molte iterazioni, è un **sottografo
minimale** — il circuito — che preserva il comportamento originale.

> 🧩 **Analogia**: è come potare un albero per rivelarne la struttura portante —
> tagli via, ramo per ramo, tutto ciò che non sostiene il "peso" del comportamento che
> ti interessa, finché non resta solo lo scheletro essenziale che lo spiega.

## 3. Activation Patching — la tecnica-chiave che rende possibile la potatura

Per decidere "questa connessione conta o no?", ACDC usa l'**activation patching**
(detto anche *causal tracing* o *interchange intervention*): si esegue il modello su
un input "pulito" e su un input "corrotto" (una variante controllata che differisce in
modo specifico), si "trapianta" l'attivazione di una componente da una corsa all'altra,
e si osserva quanto questo trapianto cambia l'output finale. Se il trapianto produce un
grande cambiamento, quella componente è causalmente rilevante per la differenza tra i
due input; se non cambia quasi nulla, probabilmente non lo è.

> 🔗 **Il collegamento concettuale con il nostro lavoro**: anche i nostri interventi
> causali (`causal_eval.py`) si basano sulla stessa logica di fondo — *modifica
> attivamente una componente interna e osserva l'effetto sull'output* — ma operano a un
> livello molto più elementare e mirato: non confrontiamo "input pulito vs corrotto"
> attraverso l'intero grafo computazionale, sostituiamo direttamente il contributo di
> *una singola feature SAE già identificata* con una versione modificata (azzerata o
> amplificata), e misuriamo l'effetto su un singolo logit di output. È un intervento
> "chirurgico su un punto", non una "mappatura sistematica dell'intero circuito".

## 4. Perché questo paper conta — ma NON è ciò che facciamo (distinzione cruciale!)

Questo è il punto su cui vale la pena essere cristallini, perché è la domanda-trabocchetto
più probabile ("ma allora fate circuit discovery come ACDC?"):

| | **ACDC (Conmy et al.)** | **Il nostro progetto** |
|---|---|---|
| **Unità di analisi** | Intero grafo computazionale del modello (migliaia di connessioni) | Singole feature SAE già scoperte e già etichettate |
| **Obiettivo** | Scoprire automaticamente *quali componenti, insieme,* implementano un comportamento | Verificare se *una* feature, isolata, ha un ruolo causale coerente con la sua etichetta |
| **Metodo** | Potatura iterativa del grafo via activation patching | Sottrazione/amplificazione mirata di un singolo contributo nel residual stream |
| **Output** | Un sottografo (circuito) | Una curva (dose-response) e una metrica (Relative Logit Drop) per feature |
| **Scala dell'analisi** | Sistemica (l'intero modello) | Locale (un punto, un layer, una feature alla volta) |

In breve: **ACDC scopre circuiti; noi verifichiamo feature**. Sono due livelli diversi
e complementari della stessa "piramide" di analisi — e il nostro lavoro, completando il
livello più elementare (un dizionario affidabile di feature monosemantiche e
causalmente verificate), **prepara il terreno** per un'eventuale applicazione futura di
tecniche come ACDC: come dice esplicitamente la nostra `conclusion.tex`, "applicare
circuit analysis tramite activation patching" è la naturale **direzione futura**, non
parte del contributo attuale.

## 5. Perché lo citiamo comunque nella related work

Tre motivi legittimi e ben argomentabili:
1. Fornisce il **vocabolario condiviso** (circuito, activation patching, grafo
   computazionale) che situa il nostro lavoro nel campo della mechanistic
   interpretability "sistemica" — anche se operiamo a un livello più elementare.
2. Dimostra che la comunità **sta lavorando attivamente** verso l'automazione di analisi
   sempre più ambiziose — il nostro lavoro fornisce un "mattoncino" (un dizionario di
   feature affidabile) che simili pipeline automatiche potrebbero, in futuro, sfruttare
   sui ViT.
3. Permette di posizionare con precisione il nostro contributo: *"non scopriamo
   circuiti — ma costruiamo, validiamo e rendiamo interpretabile l'unità di base
   (la feature monosemantica) su cui un'eventuale scoperta automatica di circuiti
   visivi dovrebbe poggiare."*

## 6. Tre frasi/idee da avere pronte per la discussione

1. *"ACDC automatizza la scoperta di circuiti potando un grafo computazionale tramite
   activation patching — un processo sistemico su scala dell'intero modello."*
2. *"Il nostro lavoro NON fa circuit discovery: facciamo interventi causali mirati
   (ablation/steering) su singole feature SAE già identificate ed etichettate — un
   livello di analisi più elementare, ma un prerequisito concettuale per
   un'eventuale applicazione futura di tecniche come ACDC al dominio visivo."*
3. *"Citiamo ACDC nella related work per il vocabolario condiviso e per situare il
   nostro contributo nel più ampio panorama della mechanistic interpretability — non
   perché lo implementiamo."*
