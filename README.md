# 📈 Markowitz Portfolio Optimizer — Streamlit App

## Avvio locale (sul tuo PC)

### 1. Prerequisiti
- Python 3.9 o superiore installato
- Git installato

### 2. Installazione

Apri il terminale nella cartella dove hai salvato i file e lancia:

```bash
pip install -r requirements.txt
```

### 3. Avvio

```bash
streamlit run app.py
```

Il browser si apre automaticamente su `http://localhost:8501`

---

## Pubblicazione online GRATIS (Streamlit Cloud)

### 1. Crea un repository GitHub
- Vai su https://github.com e crea un nuovo repository (es. `markowitz-app`)
- Carica i file `app.py` e `requirements.txt` nel repository

### 2. Deploy su Streamlit Cloud
- Vai su https://share.streamlit.io
- Clicca **"New app"**
- Collega il tuo repository GitHub
- Seleziona `app.py` come file principale
- Clicca **Deploy** → l'app sarà online con un URL pubblico in pochi minuti

---

## Come usare l'app

1. **Sidebar sinistra**: inserisci le credenziali TradingView (opzionali), il benchmark, e i ticker con i relativi pesi nel formato `TICKER, peso` (es. `AAPL, 0.10`)
2. **Premi "AVVIA ANALISI"**: l'app scarica i dati e calcola tutto automaticamente
3. **Naviga tra i tab**:
   - 🌐 Frontiera Efficiente — grafico interattivo Plotly
   - 📊 Performance & Drawdown — rendimenti e drawdown annuali
   - 📈 Rendimenti Cumulativi — confronto cumulativo in scala log + drawdown nel tempo
   - 🔄 Rolling Analytics — rolling returns e distribuzioni (orizzonte selezionabile)
   - ⚖️ Pesi & Recovery — pie chart pesi + analisi tempi di recovery

---

## Note importanti

- TradingView in modalità anonima (senza username/password) può avere limitazioni con molti ticker
- Se un ticker non viene trovato, viene escluso automaticamente e compare un avviso
- I calcoli utilizzano 252 giorni di trading per anno (standard per dati giornalieri)
- Lo Sharpe Ratio è calcolato senza risk-free rate (rf = 0)
