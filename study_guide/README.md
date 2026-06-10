# Study Guide — materiale di studio personale per il progetto P3

> ⚠️ **Questa cartella è privata**: è inclusa in `.gitignore` e non verrà mai pushata
> nel repository del gruppo. È pensata esclusivamente per te, per arrivare alla
> presentazione/discussione conoscendo il progetto "in maniera impeccabile" — anche
> nelle parti su cui partivi da zero.

## Come è organizzata, e in che ordine leggerla

La cartella è costruita per essere letta **in sequenza**, dal generale al particolare:
prima il quadro d'insieme del progetto (cosa abbiamo fatto e perché), poi i concetti
tecnici uno per uno (con tanto di equazioni, codice ed esempi), poi i paper della
letteratura (cosa dicono e perché li citiamo), poi il codice sorgente spiegato riga per
riga, infine un bigliettino riassuntivo da consultare al volo.

```
study_guide/
├── README.md                              ← sei qui: indice e percorso di lettura
│
├── 01_il_progetto_spiegato.md             ← 🎯 INIZIA DA QUI
│      Il progetto raccontato seguendo ESATTAMENTE la struttura richiesta da
│      XAI_00b_project_presentation (Introduction → Related Work → Research Gap →
│      Methodology → Results → Conclusion). Include anche una sezione finale di
│      "domande-trabocchetto plausibili e come rispondere".
│
├── 02_concetti/                           ← 📚 poi approfondisci i concetti, uno per uno
│   ├── 01_vision_transformer.md               (come funziona un ViT, dentro)
│   ├── 02_interpretabilita_meccanicistica.md  (residual stream, circuiti, polisemanticità)
│   ├── 03_sparse_autoencoder.md               (l'architettura SAE, equazione per equazione)
│   ├── 04_clip_e_valutazione_crossmodale.md   (come e perché usiamo CLIP per etichettare)
│   └── 05_interventi_causali.md               (ablation, steering, dose-response)
│
├── 03_paper/                              ← 📄 poi i riassunti dei paper della letteratura
│   ├── 01_elhage_mathematical_framework.md    (il "vocabolario": residual stream, circuiti)
│   ├── 02_bricken_monosemanticity.md          (l'idea originale dei SAE)
│   ├── 03_cunningham_sae.md                   (la validazione rigorosa dei SAE su LLM)
│   ├── 04_conmy_acdc.md                       (ACDC — citato, NON implementato: occhio!)
│   ├── 05_dosovitskiy_vit.md                  (l'architettura ViT che usiamo)
│   ├── 06_raghu_vit_vs_cnn.md                 (perché confrontiamo layer 6 e 11)
│   ├── 07_gandelsman_splice.md                (SPLICE — concept discovery su CLIP)
│   └── 08_haque_medconcept.md                 (MedConcept — concept discovery su VLM medici)
│
├── 04_codice_spiegato.md                  ← 💻 poi il codice sorgente, riga per riga
│      Giro guidato attraverso tutti e 6 i file di src/, con frammenti di codice
│      affiancati alla spiegazione di COSA fanno e PERCHÉ sono scritti così.
│
├── 05_glossario_e_cheatsheet.md           ← 📌 e infine il "bigliettino" da rileggere
│      Glossario alfabetico unificato, tutte le equazioni in un posto solo, la
│      "scheda anagrafica" del modello, una mappa "chi cita chi" tra i paper, e le
│      5 domande che — se sai rispondere bene — coprono il 90% di una discussione.
│
└── 07_guida_output.md                    ← 📊 guida completa a tutti i file in out/
       Mappa di ogni file prodotto dal pipeline, come leggerlo, cosa cercarci,
       e un riassunto già interpretato dei risultati reali (numeri, pattern, la
       feature "pilota" 5065, confronto Layer 6 vs 11). Da leggere PRIMA di
       aprire i file in out/ per non perdersi.
```

## Percorso di lettura consigliato

1. **Prima lettura — il quadro d'insieme** (~45-60 min): leggi
   [`01_il_progetto_spiegato.md`](01_il_progetto_spiegato.md) per intero. Non
   preoccuparti se alcuni concetti tecnici (SAE, CLIP, interventi causali) restano
   ancora un po' vaghi al primo passaggio — è normale, è proprio per quello che esiste
   il resto della cartella. L'obiettivo di questa prima lettura è avere **la mappa**:
   sapere come si incastrano i pezzi, prima di entrare nei dettagli di ciascuno.

2. **Seconda lettura — i concetti, uno alla volta** (~2-3 ore, anche spalmate su più
   giorni): vai in [`02_concetti/`](02_concetti/) e leggi i 5 file **nell'ordine
   numerico** — sono scritti per costruire l'uno sull'altro (es. capire il SAE richiede
   di aver capito prima la polisemanticità; capire CLIP richiede di aver capito prima
   il SAE...). Ogni file ha analogie, esempi concreti, equazioni spiegate riga per
   riga, e link diretti al codice corrispondente. **Prenditi il tempo che serve qui** —
   è la parte più densa, ma anche quella che ti darà più sicurezza.

3. **Terza lettura — i paper** (~1-2 ore): vai in [`03_paper/`](03_paper/). A questo
   punto i concetti tecnici ti saranno già familiari, quindi questi riassunti ti
   serviranno soprattutto per due cose: (a) collocare ogni paper nel "flusso logico"
   della letteratura — chi risponde a chi, chi apre quale problema — e (b) avere
   pronte 2-3 frasi precise da dire su ciascuno, nel caso ti venga chiesto
   esplicitamente "parlami del paper X". Presta particolare attenzione al file su ACDC
   ([`04_conmy_acdc.md`](03_paper/04_conmy_acdc.md)): contiene una distinzione
   importante da non confondere mai (cosa fa ACDC vs. cosa facciamo noi).

   > ⚠️ I riassunti di [SPLICE](03_paper/07_gandelsman_splice.md) e
   > [MedConcept](03_paper/08_haque_medconcept.md) sono stati scritti **senza accesso
   > diretto ai PDF originali** (non presenti in `references/research_papers/`) — si
   > basano sulla descrizione che ne dà `related_work.tex` più conoscenza generale.
   > Se hai tempo, recupera i PDF originali (sono pubblici) e usa quei riassunti come
   > base da arricchire/correggere — specialmente per MedConcept (Haque 2026), un
   > lavoro troppo recente perché io ne avessi conoscenza diretta.

4. **Quarta lettura — il codice** (~1-2 ore, **con il codice vero aperto a fianco**):
   apri [`04_codice_spiegato.md`](04_codice_spiegato.md) e, mano a mano che lo leggi,
   apri anche il file sorgente corrispondente in `src/` — leggi il codice vero, poi la
   spiegazione, poi di nuovo il codice. Questo "andirivieni" è quello che ti permetterà,
   in sede di domande, di **indicare una riga di codice e spiegarla a memoria** — un
   livello di padronanza molto più convincente del semplice "saper riassumere cosa fa
   il file".

5. **Prima di guardare i grafici** (~20 min): leggi
   [`07_guida_output.md`](07_guida_output.md) *prima* di aprire qualsiasi file in
   `out/`. Contiene la mappa di tutti i file prodotti dal pipeline, le istruzioni per
   leggerli, e un riassunto già interpretato dei risultati reali — così non devi
   dedurre i numeri da solo mentre hai davanti decine di immagini.

6. **Ripasso finale, il giorno prima** (~30 min): rileggi
   [`05_glossario_e_cheatsheet.md`](05_glossario_e_cheatsheet.md) per intero — è
   pensato apposta per essere il "ripasso lampo" che consolida tutto. Se trovi una
   voce del glossario che non sapresti spiegare a voce con sicurezza, torna al file di
   approfondimento collegato e rileggi quella sezione specifica.

## Una nota sul "perché" di questa organizzazione

Hai chiesto di costruire materiale che ti permetta di conoscere il progetto "in
maniera impeccabile" e di "eviscerare" i concetti su cui sei ancora alle basi. Per
questo, ogni file di [`02_concetti/`](02_concetti/) e [`03_paper/`](03_paper/) non si
limita a *definire* i termini — costruisce **analogie concrete** (l'armadio coi
cassetti, il riassunto col budget di parole, l'interprete bilingue, il semaforo...),
mostra **equazioni accanto al codice reale** che le implementa, e spiega sempre **il
"perché"** di ogni scelta, non solo il "cosa". L'obiettivo non è che tu memorizzi
definizioni, ma che tu costruisca un **modello mentale solido** — quello che ti
permette di rispondere con sicurezza anche a domande che non avevi previsto, perché
capisci *come si incastrano* i pezzi, non solo *come si chiamano*.

In bocca al lupo per la presentazione! 🎓
