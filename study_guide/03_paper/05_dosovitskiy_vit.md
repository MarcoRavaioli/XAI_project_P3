# Paper 5 — Dosovitskiy et al. 2021, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

**Citazione**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). *An Image is
Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
(`Dosovitskiy2021` in [`references.bib`](../../paper/Your_Paper_Title_Here/references.bib))

> Ruolo nel nostro progetto: definisce **l'architettura esatta del modello che
> studiamo** — il ViT-B/16. Ogni numero che usiamo (768, 196, 14×14, 12 layer...) viene
> da qui. È il paper "anagrafico": meno concettualmente denso degli altri, ma
> indispensabile per parlare con precisione del nostro oggetto di studio.

## 1. Il problema che affronta

Fino al 2020, la visione artificiale era dominio quasi esclusivo delle CNN — reti
costruite attorno a un'idea molto specifica: la **convoluzione**, un'operazione che
incorpora *a priori* due assunzioni sul mondo visivo (**bias induttivi**): la
*località* (i pixel vicini sono più correlati di quelli lontani) e l'*equivarianza
traslazionale* (un pattern riconosciuto in un angolo dell'immagine deve essere
riconoscibile anche altrove). I Transformer, nel frattempo, stavano rivoluzionando
l'NLP senza alcuna assunzione strutturale di questo tipo — solo attenzione e
connessioni dense. Il paper si chiede: **un'architettura "generica" come il
Transformer, applicata direttamente alle immagini (senza incorporare i bias delle
CNN), può competere con — o superare — le CNN?**

## 2. L'idea centrale — "un'immagine vale 16x16 parole"

La risposta del paper è disarmante nella sua semplicità (e geniale proprio per questo):
**tratta i pezzi di un'immagine esattamente come un Transformer tratta le parole di una
frase**. Taglia l'immagine in patch quadrate non sovrapposte (16×16 pixel), proietta
ciascuna linearmente in un vettore, e tratta la sequenza di questi vettori esattamente
come una sequenza di token testuali — stesso identico Transformer encoder usato in NLP,
nessuna modifica architetturale specifica per le immagini. Il titolo del paper è
letteralmente questo: ogni patch 16×16 diventa una "parola" nel "vocabolario" visivo
del modello.

> 🧩 **Analogia**: è come se, invece di insegnare a qualcuno a leggere un'immagine
> guardando prima i contorni, poi le forme, poi gli oggetti (l'approccio "graduale,
> guidato" delle CNN), gli dessi semplicemente in mano un mazzo di tessere — ciascuna
> un pezzettino dell'immagine — e gli dicessi "trova da solo le relazioni tra queste
> tessere; non ti do alcun suggerimento su come farlo". È un approccio "senza reti di
> sicurezza" — meno strutturato, ma potenzialmente più potente se i dati e la scala
> di calcolo sono sufficienti per imparare quelle relazioni da zero.

## 3. I risultati chiave — e perché contano per noi

- **Il ViT compete (e supera) le CNN — ma solo a grande scala**: con dataset di
  pre-training enormi (JFT-300M), il ViT raggiunge prestazioni state-of-the-art,
  superando le ResNet a parità di costo computazionale. Su dataset più piccoli, invece,
  le CNN restano competitive — proprio perché i loro bias induttivi "aiutano" quando i
  dati da cui imparare sono pochi. **Conseguenza diretta per noi**: tutto ciò che il
  nostro ViT-B/16 sa sulla struttura spaziale delle immagini, *non gli è stato dato in
  dote dall'architettura* — lo ha dovuto apprendere interamente dai dati. Questo rende
  ancora più interessante (e meno scontata) la domanda "che tipo di rappresentazioni
  spaziali ha sviluppato, internamente, per risolvere questo compito?" — esattamente la
  domanda a cui il nostro progetto prova a rispondere.
- **L'architettura, in dettaglio, è quella che useremo per tutto il progetto**:
  immagine 224×224 → patch 16×16 → 196 patch in griglia 14×14 → embedding 768-dim
  (`d_model`) → token `[CLS]` aggiunto in testa → 12 blocchi Transformer identici
  (attention + MLP, con connessioni residuali e pre-norm) → solo il `[CLS]` finale va
  alla testa di classificazione. **Tutta** la "scheda anagrafica" che trovi in
  [`02_concetti/01_vision_transformer.md`](../02_concetti/01_vision_transformer.md) §3
  viene da qui.

## 4. Perché questo paper conta — direttamente — per il nostro progetto

Non è esagerato dire che **senza questo paper il nostro progetto non avrebbe un
soggetto da studiare**: definisce l'oggetto stesso della nostra analisi. Ogni volta che
scriviamo `model_wrapper.patch_size`, `grid_size`, `d_model`, o calcoliamo
`row = (spatial_idx // grid_size) * patch_size` in `extract_patch_crop`, stiamo
manipolando direttamente le quantità che questo paper ha definito. È anche il paper che
rende **possibile** — concettualmente — la domanda di ricerca del nostro
`introduction.tex`: *"se il ViT non eredita alcun bias induttivo sulla struttura
spaziale delle immagini, e impara tutto dai dati, allora cosa ha effettivamente
imparato, internamente, per risolvere il compito di classificazione?"*

## 5. Tre frasi/idee da avere pronte per la discussione

1. *"Il ViT applica un Transformer 'puro' — identico nello spirito a quello usato per
   il testo — a sequenze di patch d'immagine linearizzate, senza alcun bias induttivo
   specifico per le immagini (niente convoluzioni, niente assunzioni di località).
   È esattamente questa 'genericità' che rende interessante chiedersi cosa abbia
   imparato internamente, e che il nostro progetto prova a investigare."*
2. *"Il nostro modello, ViT-B/16, processa immagini 224×224 in 196 patch da 16×16
   pixel, organizzate in una griglia 14×14, ciascuna proiettata in un vettore a 768
   dimensioni — più un token [CLS] aggregatore — attraverso 12 blocchi Transformer
   identici."*
3. *"Le scelte architetturali di questo paper — in particolare il fatto che
   l'informazione spaziale resti 'leggibile' anche nei layer profondi (perché niente la
   distrugge per pooling, a differenza delle CNN) — sono ciò che rende possibile, anche
   ai layer 6 e 11, mappare ogni token attivo del SAE su un ritaglio preciso
   dell'immagine originale."*
