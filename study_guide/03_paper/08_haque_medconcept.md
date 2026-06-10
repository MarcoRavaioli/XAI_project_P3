# Paper 8 — Haque et al. 2026, "MedConcept" (concept discovery non supervisionato in VLM medici)

**Citazione**: Haque, et al. (2026). *MedConcept* (titolo e dettagli completi da
verificare — riferimento arXiv:2604.11868 nel nostro
[`references.bib`](../../paper/Your_Paper_Title_Here/references.bib), voce `Haque2026`).

> ⚠️ **Nota di trasparenza — più importante che per SPLICE**: anche il PDF di questo
> paper non è presente localmente in `references/research_papers/`, ed essendo un
> lavoro molto recente (2026, oltre il mio cutoff di conoscenza), questo riassunto si
> basa **quasi interamente** sulla descrizione che ne dà il nostro
> [`related_work.tex`](../../paper/Your_Paper_Title_Here/Chapters/related_work.tex) —
> non posso arricchirlo con conoscenza diretta del paper. **Prima della consegna è
> fortemente consigliato recuperare il PDF originale** (cercalo su arXiv con l'ID
> `2604.11868`) e verificare/correggere quanto segue, specialmente titolo esatto, sede
> di pubblicazione, e dettagli tecnici della metodologia — qui ti do solo
> l'inquadramento concettuale, che è comunque coerente con il ruolo che il paper gioca
> nella nostra related work.

## 1. Il problema che affronta (per come è descritto nella nostra related work)

Anche questo paper si occupa di **concept discovery non supervisionato** — scoprire,
senza etichette predefinite, quali concetti un modello visivo (o vision-language) ha
imparato a rappresentare internamente — ma in un dominio applicativo specifico e
delicato: i **modelli vision-language in ambito medico** (es. modelli che analizzano
immagini radiologiche o istologiche insieme a referti testuali). In questo dominio, la
posta in gioco dell'interpretabilità è particolarmente alta: un concetto "scoperto" che
sembra ragionevole ma è in realtà fuorviante potrebbe avere conseguenze cliniche serie.

## 2. L'idea centrale — un LLM come "giudice esterno" della qualità semantica

Il contributo distintivo che il nostro `related_work.tex` evidenzia è l'uso di un
**LLM come giudice esterno**: dopo aver scoperto dei concetti candidati nelle
rappresentazioni interne del VLM medico, il paper non si ferma a una valutazione
qualitativa/aneddotica, ma usa un modello linguistico di grandi dimensioni per **valutare
quantitativamente, e su larga scala, quanto i concetti scoperti siano semanticamente
coerenti e ben allineati** con le categorie cliniche di interesse.

> 🔗 **Il parallelo con il nostro lavoro**: questa idea — "usa un modello esterno e
> indipendente per valutare quantitativamente l'interpretabilità di ciò che hai
> scoperto" — è strutturalmente molto simile a ciò che facciamo noi con CLIP: anche
> noi deleghiamo a un modello esterno (CLIP, non un LLM) il compito di "giudicare" e
> dare un nome ai concetti scoperti dal nostro SAE, proprio perché il modello che
> stiamo analizzando non possiede gli strumenti per farlo da sé. Haque et al. ci offre
> quindi un secondo precedente — in un dominio applicativo completamente diverso (
> medicina vs. visione generica) — della stessa strategia generale: **"se il modello
> da analizzare non sa giudicare/nominare se stesso, prendi in prestito un modello che
> sa farlo, e usalo come arbitro esterno e indipendente."**

## 3. Cosa condivide con Gandelsman et al., e dove se ne differenzia (per la nostra related work)

Il nostro `related_work.tex` accomuna questo lavoro a quello di Gandelsman et al. sotto
la stessa osservazione critica — il "filo conduttore" che porta dritto al nostro
research gap:

> Anche Haque et al., come Gandelsman et al., lavora su **modelli vision-language** —
> modelli cioè che, per costruzione, possiedono già una qualche forma di allineamento
> tra rappresentazioni visive e linguaggio (sono allenati su coppie immagine-testo, in
> questo caso immagini mediche e referti). Questo allineamento incorporato è ciò che
> rende possibile, in primo luogo, "interrogare" il modello — o un giudice esterno — in
> linguaggio naturale sui concetti che rappresenta.

La differenza rispetto a Gandelsman et al. sta nel **dove si colloca il "ponte
linguistico"**: in SPLICE il ponte è interno al modello stesso (CLIP è già
vision-language); in Haque et al. il ponte è — almeno in parte — esterno, delegato a un
LLM giudice. Quest'ultima impostazione è, in un certo senso, **più vicina nello spirito
alla nostra**: anche noi deleghiamo il giudizio/l'etichettatura a un modello esterno
indipendente. La differenza residua è che il *modello analizzato* in Haque et al. è
comunque un VLM (con un certo grado di allineamento testo-immagine già presente),
mentre il nostro è un ViT puramente visivo, **senza alcun allineamento testuale
incorporato in nessun punto del processo** — il caso "più difficile" della famiglia.

## 4. Perché questo paper conta — direttamente — per il nostro progetto

Serve, nella nostra related work, a **completare il quadro** prima di presentare il
gap: mostra che la strategia "usa un modello/giudice esterno per valutare
l'interpretabilità" è già stata esplorata con successo — ma sempre, fino ad ora, in
contesti dove il modello analizzato possedeva già una qualche forma di allineamento
linguistico (per costruzione, o tramite il dominio applicativo). Insieme a Gandelsman
et al., permette al nostro `related_work.tex` di concludere con la frase-cardine che
introduce il gap: *"la domanda di come identificare ed etichettare feature
interpretabili in un modello puramente visivo, allenato senza alcuna supervisione
testuale, resta aperta."*

## 5. Tre frasi/idee da avere pronte per la discussione (con cautela — verifica i dettagli prima!)

1. *"Haque et al. applica concept discovery non supervisionato a modelli
   vision-language in ambito medico, usando un LLM come giudice esterno per valutare
   quantitativamente la coerenza semantica dei concetti scoperti — un precedente
   diretto della strategia 'delega il giudizio a un modello esterno indipendente' che
   adottiamo anche noi con CLIP."*
2. *"Sia Gandelsman et al. sia Haque et al. lavorano su modelli che possiedono già,
   per costruzione o per dominio applicativo, una qualche forma di allineamento
   testo-immagine — è proprio l'assenza di questo allineamento, nel nostro ViT puro,
   a definire il gap che affrontiamo."*
3. ⚠️ *"Non ho potuto verificare i dettagli tecnici precisi di questo lavoro sul PDF
   originale (non disponibile localmente, e pubblicato dopo il mio cutoff di
   conoscenza) — se il docente fa una domanda molto specifica su questo paper, è
   meglio rispondere con onestà ('non ho approfondito i dettagli tecnici fini di
   questo lavoro specifico, ma il suo ruolo nella nostra related work è...') piuttosto
   che inventare dettagli che potrebbero risultare imprecisi."*
