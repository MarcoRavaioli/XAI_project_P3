# Paper 6 — Raghu et al. 2021, "Do Vision Transformers See Like Convolutional Neural Networks?"

**Citazione**: Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., Dosovitskiy, A.
(2021). *Do Vision Transformers See Like Convolutional Neural Networks?*. NeurIPS 2021.
(`Raghu2021` in [`references.bib`](../../paper/Your_Paper_Title_Here/references.bib))

> Ruolo nel nostro progetto: è il paper che fornisce **l'evidenza empirica diretta**
> per la nostra scelta più importante a livello sperimentale — confrontare layer 6
> (mid-network) e layer 11 (late-network) — e la base teorica del perché ha senso
> mappare anche i token profondi su ritagli precisi dell'immagine.

## 1. Il problema che affronta

Dosovitskiy et al. avevano dimostrato *che* i ViT funzionano. Ma una domanda restava
aperta: **funzionano "allo stesso modo" delle CNN — solo con un'architettura diversa
che converge alle stesse soluzioni — oppure sviluppano un modo *qualitativamente
diverso* di "vedere" le immagini?** Raghu et al. affrontano questa domanda con un
confronto sistematico e quantitativo tra le rappresentazioni interne di ViT e ResNet,
usando come strumento principale la **CKA (Centered Kernel Alignment)** — una metrica
che misura quanto due rappresentazioni interne (anche di reti con architetture
completamente diverse) si "somigliano".

## 2. L'idea centrale — "stesso compito, percorsi interni diversi"

La risposta — sorprendentemente netta — è: **no, i ViT non vedono come le CNN**.
Risolvono lo stesso compito (classificazione di immagini) arrivando a prestazioni
comparabili, ma il *percorso interno* attraverso cui ci arrivano è qualitativamente
diverso. È un po' come scoprire che due persone arrivano alla stessa destinazione, ma
una ha seguito l'autostrada (un percorso lineare, con tappe nette e ben distinte) e
l'altra ha tagliato per stradine secondarie che si assomigliano molto tra loro
(un percorso più "uniforme", senza fasi nettamente separate).

## 3. Le scoperte chiave — punto per punto, con la rilevanza per noi

1. **I ViT integrano informazione globale fin dal primo layer**: già nei primissimi
   blocchi, alcune teste di attenzione "guardano" porzioni lontane dell'immagine
   (misurato tramite la metrica della "distanza media di attenzione"). Le CNN, al
   contrario, costruiscono il loro campo recettivo *gradualmente*, layer dopo layer —
   nei primi strati vedono solo piccole zone locali.
2. **Le rappresentazioni dei ViT sono molto più "uniformi" lungo la profondità**: la
   CKA tra layer consecutivi (e anche tra layer non adiacenti) è sistematicamente più
   alta nei ViT che nelle ResNet. Le CNN attraversano fasi nettamente distinte
   (early = bordi/texture/colore, late = forme/oggetti/parti); nei ViT questa
   transizione è molto più graduale e i blocchi si "rassomigliano" tra loro molto di
   più — un effetto attribuito in parte alle connessioni residuali, che permettono
   all'informazione di "saltare" i blocchi e restare sostanzialmente intatta.
3. **L'informazione di localizzazione spaziale è preservata fino agli ultimi layer**
   nei ViT — un punto **cruciale per la fattibilità del nostro intero progetto**: nelle
   CNN, il pooling distrugge progressivamente l'informazione "di dove si trova cosa";
   nei ViT, grazie all'assenza di pooling e alla presenza dei positional embedding,
   anche un token al layer 11 mantiene un legame riconoscibile con la sua posizione
   originale nella griglia. **Senza questa proprietà, la nostra `extract_patch_crop` —
   applicata a feature di layer profondi — non avrebbe alcun significato.**
4. **Il ruolo del pre-training su larga scala**: la quantità di dati di pre-training
   influenza sensibilmente *quanto* le rappresentazioni del ViT assomigliano (o non
   assomigliano) a quelle delle CNN — un'ulteriore conferma che il ViT non eredita
   "soluzioni" pronte dall'architettura, ma le costruisce (in modo diverso a seconda
   delle condizioni) dai dati che vede.

## 4. Perché questo paper conta — direttamente — per il nostro progetto

Questo è probabilmente, fra i sei paper "tecnici", quello con la **connessione più
diretta e concreta** a una scelta sperimentale specifica del nostro lavoro:

> 🔗 **La connessione**: se le rappresentazioni cambiano qualitativamente lungo la
> profondità — passando (presumibilmente) da pattern più locali/di basso livello a
> concetti più globali/semantici — allora ha senso, dal punto di vista
> dell'interpretabilità, **non limitarsi a un solo layer**, ma confrontare almeno due
> punti rappresentativi di questa traiettoria. Da qui la nostra scelta di analizzare
> separatamente il **layer 6** ("mid-network" — dove ti aspetti feature ancora
> abbastanza locali, legate a texture/colori/pattern) e il **layer 11** ("late-network"
> — dove ti aspetti feature più astratte, legate a forme/parti/oggetti).
>
> Se i nostri risultati confermano questo gradiente — feature del layer 6 etichettate
> come "fur", "scale pattern", "red color"; feature del layer 11 etichettate come
> "eye", "snout", "background foliage" — abbiamo ottenuto una **conferma indipendente,
> con uno strumento completamente diverso (SAE + etichettatura CLIP)**, di un fenomeno
> che Raghu et al. avevano osservato con tecniche diverse (CKA, attention distance). È
> esattamente il tipo di triangolazione che rende un risultato scientificamente solido
> — e vale la pena sottolinearlo esplicitamente in fase di analisi dei risultati.

E se invece *non* lo confermano? Anche quello è interessante da discutere — magari il
nostro modello (ViT-B/16 supervisionato su ImageNet-1k, una scala di dati molto più
piccola di JFT-300M) sviluppa un gradiente meno marcato o diverso da quanto osservato
con i modelli a scala maggiore studiati da Raghu et al. (vedi punto 4 sopra: il
pre-training su scala diversa produce rappresentazioni qualitativamente diverse).

## 5. Tre frasi/idee da avere pronte per la discussione

1. *"Raghu et al. dimostrano, con la metrica CKA, che i ViT 'vedono' in modo
   qualitativamente diverso dalle CNN: integrano informazione globale fin dai primi
   layer, e le loro rappresentazioni sono molto più uniformi lungo la profondità —
   senza le fasi nettamente distinte (locale → globale) tipiche delle CNN."*
2. *"Un punto cruciale per noi: l'informazione spaziale resta accessibile anche nei
   layer profondi del ViT — è ciò che rende possibile, e sensato, mappare le feature
   scoperte dal SAE su ritagli precisi dell'immagine, anche al layer 11."*
3. *"La nostra scelta di confrontare layer 6 e layer 11 è motivata direttamente da
   questo paper: vogliamo verificare — con uno strumento diverso, SAE + CLIP — se il
   gradiente di astrazione (locale → semantico) osservato da Raghu et al. con la CKA
   si manifesta anche a livello di singole feature monosemantiche scoperte nello
   spazio delle attivazioni MLP."*
