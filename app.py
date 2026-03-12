from flask import Flask, render_template_string
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg') # 伺服器端繪圖必須加上這行，避免產生視窗錯誤
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import concurrent.futures
from io import StringIO
import warnings
import os
import datetime
import json
import logging

app = Flask(__name__)

# ==========================================
# 參數設定區 (對應原本的 Colab 參數)
# ==========================================
LOOKBACK_YEARS = 1
MAX_TICKERS = 3 # ⚠️ 測試階段先設為 30，實際上線可改為 200
FTD_VALID_DAYS = 20
ATR_STOP_LOSS_MULT = 2.5
ATR_TAKE_PROFIT_MULT = 5
TIME_STOP_DAYS = 15
MAX_VOLATILITY_PCT = 0.05
MAX_ACCOUNT_RISK_PCT = 0.01

BLACKLIST_SECTORS = ['Utilities', 'Healthcare', 'Consumer Cyclical', 'Consumer Defensive', 'Real Estate']

# 設定 Flask 專用的靜態檔案路徑
CHARTS_DIR = os.path.join("static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

@app.route('/')
def home():
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8-dark')
    plt.ioff()

    print("⏳ 收到網頁請求，正在執行量化運算 (這可能需要幾十秒)...")

    # ==========================================
    # 模組 1 & 2: 載入清單與基本面
    # ==========================================
    BASE_TICKER_DESC = {'SPY': '標普500', 'QQQ': '納斯達克', 'TLT': '20年美債', 'GLD': '黃金', 'ITA': '軍工', 'XLE': '能源', 'XLK': '科技', '^VIX': '恐慌指數'}
    dynamic_tickers = {}
    try:
        res = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(res.text, 'html.parser')
        sp500_table = str(soup.find('table', {'id': 'constituents'}))
        sp500_df = pd.read_html(StringIO(sp500_table))[0]
        for _, row in sp500_df.head(MAX_TICKERS).iterrows():
            tk = row['Symbol'].replace('.', '-')
            if tk not in BASE_TICKER_DESC: dynamic_tickers[tk] = str(row['Security'])
    except:
        for tk in ['AAPL','MSFT','GOOGL','AMZN','NVDA']: dynamic_tickers[tk] = "S&P500"

    for tk in ['MSTR', 'PLTR', 'ARM', 'LMT', 'NOC', 'TSM', 'ANET', 'APH', 'AXP', 'BAC']: dynamic_tickers[tk] = "Watchlist"
    ALL_TICKERS = list({**BASE_TICKER_DESC, **dynamic_tickers}.keys())

    fund_data = {}
    def fetch_fundamentals(tk):
        try:
            info = yf.Ticker(tk).info
            return tk, {'sector': info.get('sector', 'ETF/Index')}
        except: return tk, {'sector': 'Unknown'}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(fetch_fundamentals, ALL_TICKERS):
            fund_data[res[0]] = res[1]

    # ==========================================
    # 模組 3: 下載數據與準備大盤指標
    # ==========================================
    data = yf.download(ALL_TICKERS, period=f"{LOOKBACK_YEARS}y", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        closes, vols, lows, highs, opens = data['Close'].ffill(), data['Volume'].ffill(), data['Low'].ffill(), data['High'].ffill(), data['Open'].ffill()
    else:
        closes, vols, lows, highs, opens = data[['Close']].ffill(), data[['Volume']].ffill(), data[['Low']].ffill(), data[['High']].ffill(), data[['Open']].ffill()

    spy_c = closes['SPY'] if 'SPY' in closes.columns else closes.iloc[:, 0]
    spy_l, spy_v = lows['SPY'] if 'SPY' in lows.columns else lows.iloc[:, 0], vols['SPY'] if 'SPY' in vols.columns else vols.iloc[:, 0]
    vix_c = closes['^VIX'] if '^VIX' in closes.columns else pd.Series(20, index=spy_c.index)
    spy_20, spy_50, spy_200 = spy_c.rolling(20).mean(), spy_c.rolling(50).mean(), spy_c.rolling(200).mean()

    # ==========================================
    # 模組 4: 大盤市寬、派發日與 FTD 偵測
    # ==========================================
    sma50_all = closes.rolling(50).mean()
    market_breadth = (closes > sma50_all).sum(axis=1) / closes.shape[1] * 100
    curr_breadth = round(float(market_breadth.iloc[-1]), 1)

    spy_ret = spy_c.pct_change()
    dist_mask = (spy_ret < -0.002) & (spy_v > spy_v.shift(1))
    dist_days_series = dist_mask.rolling(25).sum()
    curr_dist_days = int(dist_days_series.iloc[-1])
    dist_dates = spy_c.index[dist_mask]

    r126 = closes / closes.shift(126) - 1
    r252 = closes / closes.shift(252) - 1
    rs_rank = ((0.6 * r126) + (0.4 * r252)).rank(axis=1, pct=True) * 99 + 1
    rs_momentum = rs_rank - rs_rank.shift(20)

    ftd_history = np.zeros(len(spy_c))
    ftd_dates = []
    rally_day, rally_low, last_ftd_idx = 0, float('inf'), -999

    for i in range(1, len(spy_c)):
        c, pc = spy_c.iloc[i], spy_c.iloc[i-1]
        l, v, pv = spy_l.iloc[i], spy_v.iloc[i], spy_v.iloc[i-1]
        if l < rally_low:
            rally_low, rally_day = l, 1 if c > pc else 0
        else:
            if c > pc: rally_day = max(1, rally_day + 1)
            elif rally_day > 0: rally_day += 1
        if rally_day >= 4 and c > pc * 1.012 and v > pv:
            last_ftd_idx, rally_low, rally_day = i, c, 0
            ftd_dates.append(spy_c.index[i])
        ftd_history[i] = (i - last_ftd_idx) if last_ftd_idx > 0 else 999

    current_ftd_days = int(ftd_history[-1])
    is_bull_market = spy_c.iloc[-1] > spy_200.iloc[-1]
    is_vix_panic = vix_c.iloc[-1] > 25

    # 繪製並儲存 SPY 圖表到 static/charts
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    ax.plot(spy_c.index[-200:], spy_c.iloc[-200:], color='#cbd5e1', label='SPX', linewidth=1.5)
    ax.plot(spy_20.index[-200:], spy_20.iloc[-200:], color='#3b82f6', label='20MA', linewidth=1, alpha=0.8)
    ax.plot(spy_50.index[-200:], spy_50.iloc[-200:], color='#f59e0b', label='50MA', linewidth=1, alpha=0.8)
    ax.plot(spy_200.index[-200:], spy_200.iloc[-200:], color='#dc2626', label='200MA', linestyle='-.', linewidth=1.5)

    recent_ftds = [d for d in ftd_dates if d >= spy_c.index[-200]]
    recent_dists = [d for d in dist_dates if d >= spy_c.index[-200]]
    if recent_ftds:
        ax.scatter(recent_ftds, spy_c.loc[recent_ftds] * 0.97, marker='^', color='#10b981', s=100, label='FTD', zorder=5)
    if recent_dists:
        ax.scatter(recent_dists, spy_c.loc[recent_dists] * 1.02, marker='v', color='#ef4444', s=40, label='Dist Day', zorder=5)

    fig.patch.set_facecolor('#0f172a'); ax.set_facecolor('#0f172a')
    ax.tick_params(colors='white', labelsize=8)
    ax.legend(facecolor='#1e293b', labelcolor='white', loc='upper left', ncol=3, fontsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#334155')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "SPY_Trend.png"), transparent=True)
    plt.close(fig)

    if is_vix_panic:
        ftd_status = "🚨 VIX 恐慌警戒 (暫停買進)"
        ftd_color = "text-red-500 bg-red-500/20 border-red-500/50"
    elif is_bull_market:
        ftd_status = "🟢 牛市格局 (SPX > 200MA)"
        ftd_color = "text-emerald-500 bg-emerald-500/10 border-emerald-500/20"
    elif current_ftd_days <= FTD_VALID_DAYS:
        ftd_status = f"✅ 底部確認 ({current_ftd_days}天前 FTD)"
        ftd_color = "text-blue-400 bg-blue-500/10 border-blue-500/20"
    else:
        ftd_status = "❌ 熊市空頭 (等待 FTD)"
        ftd_color = "text-red-500 bg-red-500/10 border-red-500/20"

    # ==========================================
    # 模組 5: 高階回測模擬
    # ==========================================
    etf_js_data = []

    for ticker in ALL_TICKERS:
        if ticker == '^VIX': continue
        try:
            if ticker not in closes.columns: continue
            c, h, l, o, v = closes[ticker], highs[ticker], lows[ticker], opens[ticker], vols[ticker]
            if c.isna().sum() > 50: continue

            sector = fund_data.get(ticker, {}).get('sector', '')
            is_blacklisted = sector in BLACKLIST_SECTORS

            sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
            atr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).ewm(alpha=1/14, adjust=False).mean()

            volatility_pct = atr / c
            is_too_volatile = volatility_pct > MAX_VOLATILITY_PCT

            rs = rs_rank[ticker]
            rs_mom = rs_momentum[ticker]

            is_uptrend = (sma20 > sma50) & (sma50 > sma200)
            sig_pullback = is_uptrend & (rs >= 70) & (l <= sma20 * 1.01) & (c > sma20) & (~is_blacklisted) & (~is_too_volatile)

            trades_log = []
            in_trade = False
            entry_px, sl, tp, entry_date, days_held, initial_atr = 0, 0, 0, None, 0, 0

            for i in range(200, len(c)):
                market_safe = ((spy_c.iloc[i] > spy_200.iloc[i]) or (0 < ftd_history[i] <= FTD_VALID_DAYS)) and (vix_c.iloc[i] < 25)

                if not in_trade and sig_pullback.iloc[i] and market_safe:
                    in_trade = True
                    entry_px = c.iloc[i]
                    entry_date = c.index[i]
                    initial_atr = atr.iloc[i]
                    sl = entry_px - (ATR_STOP_LOSS_MULT * initial_atr)
                    tp = entry_px + (ATR_TAKE_PROFIT_MULT * initial_atr)
                    days_held = 0

                elif in_trade:
                    days_held += 1
                    exit_date = c.index[i]
                    if h.iloc[i] >= (entry_px + (2.0 * initial_atr)): sl = max(sl, entry_px)
                    time_stop_triggered = (days_held >= TIME_STOP_DAYS) and (c.iloc[i] < (entry_px + initial_atr))

                    if l.iloc[i] <= sl:
                        ret_pct = (sl/entry_px - 1)
                        trades_log.append({'entry': entry_date, 'exit': exit_date, 'ret': ret_pct, 'type': 'LOSS', 'px': sl})
                        in_trade = False
                    elif h.iloc[i] >= tp:
                        ret_pct = (tp/entry_px - 1)
                        trades_log.append({'entry': entry_date, 'exit': exit_date, 'ret': ret_pct, 'type': 'WIN', 'px': tp})
                        in_trade = False
                    elif time_stop_triggered:
                        ret_pct = (c.iloc[i]/entry_px - 1)
                        trades_log.append({'entry': entry_date, 'exit': exit_date, 'ret': ret_pct, 'type': 'TIME_STOP', 'px': c.iloc[i]})
                        in_trade = False

            returns = [t['ret'] for t in trades_log]
            win_rate = (len([r for r in returns if r > 0]) / len(returns) * 100) if returns else 0
            avg_ret = (np.mean(returns) * 100) if returns else 0

            is_active = sig_pullback.iloc[-1]
            curr_market_safe = (is_bull_market or (current_ftd_days <= FTD_VALID_DAYS)) and not is_vix_panic

            curr_price = float(c.iloc[-1])
            curr_atr = float(atr.iloc[-1])
            calc_sl = curr_price - (ATR_STOP_LOSS_MULT * curr_atr)
            calc_tp = curr_price + (ATR_TAKE_PROFIT_MULT * curr_atr)
            risk_per_share = curr_price - calc_sl if curr_price > calc_sl else 0.001

            risk_dist_pct = risk_per_share / curr_price
            suggested_size_pct = min((MAX_ACCOUNT_RISK_PCT / risk_dist_pct) * 100, 100) if risk_dist_pct > 0 else 0

            if is_blacklisted: status_text = "🚫 Weak Sector"
            elif is_too_volatile.iloc[-1]: status_text = "🚫 High Volatility"
            elif is_active and curr_market_safe: status_text = "🔥 Active"
            elif is_active: status_text = "⚠️ Macro Unsafe"
            else: status_text = "Idle"

            has_chart = False
            if is_active or len(trades_log) > 0:
                plot_df = pd.DataFrame({'Close': c, 'SMA20': sma20}).last('252D')
                recent_trades = [t for t in trades_log if t['entry'] >= plot_df.index[0]]

                fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=100)
                ax2.plot(plot_df.index, plot_df.Close, color='#cbd5e1', linewidth=1.5)
                ax2.plot(plot_df.index, plot_df.SMA20, color='#f59e0b', linewidth=2)

                for t in recent_trades:
                    ax2.scatter(t['entry'], plot_df.loc[t['entry'], 'Close']*0.95, marker='^', color='#3b82f6', s=120, zorder=5)
                    if t['type'] == 'WIN': ax2.scatter(t['exit'], t['px']*1.05, marker='v', color='#10b981', s=120, zorder=5)
                    elif t['type'] == 'LOSS': ax2.scatter(t['exit'], t['px']*0.95, marker='X', color='#ef4444', s=100, zorder=5)
                    elif t['type'] == 'TIME_STOP': ax2.scatter(t['exit'], t['px']*1.05, marker='s', color='#f59e0b', s=80, zorder=5)

                ax2.set_facecolor('#1e293b'); fig2.patch.set_facecolor('#1e293b')
                ax2.tick_params(colors='white', labelsize=8); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=30); plt.tight_layout()
                plt.savefig(os.path.join(CHARTS_DIR, f"{ticker}_bt.png"), transparent=True)
                plt.close(fig2)
                has_chart = True

            if is_active or len(trades_log) > 0:
                curr_rs_mom = round(float(rs_mom.iloc[-1]), 1) if not pd.isna(rs_mom.iloc[-1]) else 0
                etf_js_data.append({
                    "ticker": ticker, "sector": sector,
                    "rs": round(float(rs.iloc[-1]), 0) if not pd.isna(rs.iloc[-1]) else 0,
                    "rs_mom": curr_rs_mom,
                    "status": status_text,
                    "win_rate": round(win_rate, 1), "avg_ret": round(avg_ret, 1),
                    "trades_cnt": len(trades_log), "has_chart": has_chart,
                    "pos_size": f"{round(suggested_size_pct, 1)}%",
                    "curr_price": curr_price, "sl_price": calc_sl, "tp_price": calc_tp, "risk_per_share": risk_per_share
                })

        except Exception as e: pass

    # ==========================================
    # 模組 6: 生成 HTML (已更新靜態路徑)
    # ==========================================
    breadth_color = "text-emerald-400" if curr_breadth > 40 else "text-red-400"
    dist_color = "text-red-400" if curr_dist_days >= 5 else "text-emerald-400"

    html_template = f"""<!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.tailwindcss.com"></script>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <title>Quant Master V3.1 Live</title>
    </head>
    <body class="bg-[#0f172a] text-slate-200 h-screen overflow-hidden flex flex-col font-sans">

        <header class="bg-slate-900 border-b border-slate-800 p-3 flex justify-between items-center shrink-0">
            <div>
                <h1 class="text-2xl font-black text-white">QUANT <span class="text-blue-500">MASTER V3.1</span></h1>
                <p class="text-[10px] text-slate-400">Live Web App Execution</p>
            </div>

            <div class="flex gap-3">
                <div class="px-3 py-1 rounded-lg border border-slate-700 bg-slate-800/50 flex flex-col justify-center">
                    <span class="text-[9px] uppercase font-bold text-slate-400">Market Breadth (>50MA)</span>
                    <span class="font-black text-lg {breadth_color}">{curr_breadth}%</span>
                </div>
                <div class="px-3 py-1 rounded-lg border border-slate-700 bg-slate-800/50 flex flex-col justify-center">
                    <span class="text-[9px] uppercase font-bold text-slate-400">Distribution Days</span>
                    <span class="font-black text-lg {dist_color}">{curr_dist_days} Days</span>
                </div>
                <div class="px-4 py-1 rounded-lg border {ftd_color} flex flex-col justify-center">
                    <span class="text-[9px] uppercase font-bold opacity-70">Macro Regime Status</span>
                    <span class="font-black">{ftd_status}</span>
                </div>
            </div>
        </header>

        <main class="flex-1 flex overflow-hidden p-3 gap-3">
            <div class="w-1/3 flex flex-col gap-3 overflow-hidden h-full">
                <div class="bg-slate-900 rounded-xl border border-slate-800 flex flex-col flex-1 h-[60%]">
                    <div class="p-3 border-b border-slate-800 font-bold text-blue-400 text-sm flex justify-between items-center shrink-0">
                        <span>🎯 Signals & RS Momentum</span>
                    </div>

                    <div class="p-2 border-b border-slate-800 flex gap-2 text-[10px] shrink-0">
                        <button class="bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded" onclick="applyFilter('all')">All</button>
                        <button class="bg-blue-900/50 text-blue-300 hover:bg-blue-800/50 px-2 py-1 rounded" onclick="applyFilter('active')">🔥 Active</button>
                        <button class="bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded" onclick="applyFilter('rs80')">RS > 80</button>
                    </div>

                    <div class="overflow-y-auto flex-1 relative">
                        <table class="w-full text-left text-xs">
                            <thead class="bg-slate-800 sticky top-0 z-10 cursor-pointer select-none border-b border-slate-700">
                                <tr>
                                    <th class="p-2 hover:text-white" onclick="sortBy('ticker')">Ticker ↕</th>
                                    <th class="p-2 hover:text-white" onclick="sortBy('rs')">RS (Mom) ↕</th>
                                    <th class="p-2 hover:text-white" onclick="sortBy('win_rate')">Win% ↕</th>
                                    <th class="p-2 hover:text-white" onclick="sortBy('avg_ret')">AvgRet ↕</th>
                                </tr>
                            </thead>
                            <tbody id="signal-table"></tbody>
                        </table>
                    </div>
                </div>

                <div class="bg-slate-800/50 rounded-xl border border-slate-700 flex flex-col h-[40%] p-2 relative shrink-0">
                    <div class="text-xs font-bold text-slate-400 mb-1 absolute top-2 left-2 z-10 bg-slate-900/80 px-2 py-1 rounded" id="bt_title">Backtest</div>
                    <div class="flex-1 flex items-center justify-center overflow-hidden mt-6">
                        <p id="bt_placeholder" class="text-slate-500 text-[10px] text-center leading-relaxed">Select a ticker</p>
                        <img id="bt_img" src="" class="hidden max-h-full max-w-full object-contain rounded-lg">
                    </div>
                </div>
            </div>

            <div class="w-2/3 flex flex-col gap-3">
                <div class="bg-slate-900 p-2 rounded-xl border border-slate-800 h-[180px] shrink-0 flex items-center justify-center relative">
                    <div class="absolute top-2 left-3 z-10 flex gap-2 items-center">
                        <span class="text-xs font-bold text-slate-400">SPX Anatomy:</span>
                        <span class="text-[9px] bg-blue-500/20 text-blue-400 px-1 rounded border border-blue-500/30">20MA</span>
                        <span class="text-[9px] bg-amber-500/20 text-amber-400 px-1 rounded border border-amber-500/30">50MA</span>
                        <span class="text-[9px] bg-red-500/20 text-red-400 px-1 rounded border border-red-500/30">200MA</span>
                        <span class="text-[9px] text-emerald-400 ml-2">▲ FTD</span>
                        <span class="text-[9px] text-red-400">▼ Dist Day</span>
                    </div>
                    <img src="/static/charts/SPY_Trend.png" class="max-h-full max-w-full object-contain">
                </div>

                <div class="bg-slate-900 rounded-xl border border-slate-700 p-3 shrink-0">
                    <div class="flex justify-between items-center mb-2">
                        <div class="flex items-center gap-2">
                            <h3 class="text-sm font-bold text-amber-500">🧮 Trade Execution Plan</h3>
                            <span id="calc_ticker_name" class="text-xs font-bold text-white bg-slate-700 px-2 rounded">-</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <label class="text-[10px] text-slate-400">Account Size ($):</label>
                            <input type="number" id="acc_size" value="100000" class="bg-slate-800 border border-slate-600 text-white text-xs px-2 py-1 rounded w-24 text-right focus:outline-none focus:border-amber-500" onchange="updateCalculator()" onkeyup="updateCalculator()">
                        </div>
                    </div>
                    <div class="grid grid-cols-5 gap-2 text-center">
                        <div class="bg-slate-800/50 p-2 rounded border border-slate-700">
                            <div class="text-[9px] text-slate-400 uppercase">Entry Price</div>
                            <div class="font-bold text-white text-sm" id="calc_entry">-</div>
                        </div>
                        <div class="bg-red-900/10 p-2 rounded border border-red-900/50">
                            <div class="text-[9px] text-red-400 uppercase font-bold">Stop Loss (-2 ATR)</div>
                            <div class="font-bold text-red-400 text-sm" id="calc_sl">-</div>
                        </div>
                        <div class="bg-emerald-900/10 p-2 rounded border border-emerald-900/50">
                            <div class="text-[9px] text-emerald-400 uppercase font-bold">Target (+4 ATR)</div>
                            <div class="font-bold text-emerald-400 text-sm" id="calc_tp">-</div>
                        </div>
                        <div class="bg-slate-800/50 p-2 rounded border border-slate-700">
                            <div class="text-[9px] text-amber-500 uppercase font-bold" title="To strictly risk only 1% of account">Shares to Buy</div>
                            <div class="font-bold text-amber-400 text-sm" id="calc_shares">-</div>
                        </div>
                        <div class="bg-slate-800/50 p-2 rounded border border-slate-700">
                            <div class="text-[9px] text-slate-400 uppercase">Total Position Cost</div>
                            <div class="font-bold text-blue-300 text-sm" id="calc_cost">-</div>
                        </div>
                    </div>
                </div>

                <div class="bg-slate-900 p-1 rounded-xl border border-slate-800 flex-1 relative" id="tv_chart_container"></div>
            </div>
        </main>

        <script>
            let rawData = {json.dumps(etf_js_data)};
            let currentData = [...rawData];
            let sortCol = 'rs';
            let sortAsc = false;
            let currentSelectedTicker = null;

            let tvWidget = null;
            function loadContent(ticker) {{
                currentSelectedTicker = ticker;
                if (tvWidget) {{ tvWidget.remove(); }}
                tvWidget = new TradingView.widget({{
                    "autosize": true, "symbol": ticker, "interval": "D", "timezone": "Etc/UTC",
                    "theme": "dark", "style": "1", "locale": "en", "container_id": "tv_chart_container"
                }});

                const data = rawData.find(d => d.ticker === ticker);
                const imgEl = document.getElementById('bt_img');
                const placeholder = document.getElementById('bt_placeholder');
                const title = document.getElementById('bt_title');

                if(data && data.has_chart) {{
                    // 💡 注意這裡改成了 /static/charts/
                    imgEl.src = '/static/charts/' + ticker + '_bt.png';
                    imgEl.classList.remove('hidden');
                    placeholder.classList.add('hidden');
                    title.innerText = ticker + " 1-Yr Backtest";
                }} else {{
                    imgEl.classList.add('hidden');
                    placeholder.classList.remove('hidden');
                    title.innerText = ticker + " (No Recent Trades)";
                }}

                updateCalculator();
            }}

            function updateCalculator() {{
                if (!currentSelectedTicker) return;
                const data = rawData.find(d => d.ticker === currentSelectedTicker);
                if (!data) return;

                document.getElementById('calc_ticker_name').innerText = data.ticker;

                const accountSize = parseFloat(document.getElementById('acc_size').value) || 100000;
                const riskAmount = accountSize * {MAX_ACCOUNT_RISK_PCT};
                let shares = Math.floor(riskAmount / data.risk_per_share);
                if (shares <= 0) shares = 0;

                const totalCost = shares * data.curr_price;
                const actualPosPct = (accountSize > 0) ? (totalCost / accountSize * 100).toFixed(1) : 0;

                document.getElementById('calc_entry').innerText = "$" + data.curr_price.toFixed(2);
                document.getElementById('calc_sl').innerText = "$" + data.sl_price.toFixed(2);
                document.getElementById('calc_tp').innerText = "$" + data.tp_price.toFixed(2);
                document.getElementById('calc_shares').innerText = shares;
                document.getElementById('calc_cost').innerText = "$" + totalCost.toLocaleString(undefined, {{maximumFractionDigits: 0}}) + " (" + actualPosPct + "%)";
            }}

            function renderTable() {{
                let html = "";
                currentData.forEach(d => {{
                    const isFire = d.status.includes('🔥');
                    const isBlocked = d.status.includes('🚫');
                    let statColor = isFire ? 'text-emerald-400 font-bold' : (isBlocked ? 'text-red-500/50' : 'text-slate-500');
                    let rowBg = isFire ? 'bg-blue-900/20 border-blue-500/30' : 'hover:bg-slate-800 border-slate-800/50';
                    let momColor = d.rs_mom > 0 ? 'text-emerald-400' : (d.rs_mom < 0 ? 'text-red-400' : 'text-slate-500');
                    let rsText = d.rs_mom > 0 ? '+'+d.rs_mom : d.rs_mom;

                    html += `<tr class="border-b cursor-pointer ${{rowBg}}" onclick="loadContent('${{d.ticker}}')">
                        <td class="p-2 font-black text-blue-400">${{d.ticker}}<br><span class="${{statColor}} text-[9px] block">${{d.status}}</span></td>
                        <td class="p-2 font-bold text-[11px]">${{d.rs}} <span class="text-[9px] ${{momColor}} ml-1" title="RS Momentum (20D)">${{rsText}}</span></td>
                        <td class="p-2">${{d.win_rate}}% <span class="text-[9px] text-slate-500 block">(${{d.trades_cnt}} trd)</span></td>
                        <td class="p-2 ${{d.avg_ret > 0 ? 'text-emerald-400' : 'text-red-400'}}">${{d.avg_ret}}%</td>
                    </tr>`;
                }});
                document.getElementById('signal-table').innerHTML = html || '<tr><td colspan="4" class="p-4 text-center text-slate-500">No results found</td></tr>';
            }}

            function sortBy(col) {{
                if(sortCol === col) {{ sortAsc = !sortAsc; }}
                else {{ sortCol = col; sortAsc = false; }}

                currentData.sort((a, b) => {{
                    let valA = a[col]; let valB = b[col];
                    if(col === 'pos_size') {{ valA = parseFloat(valA); valB = parseFloat(valB); }}
                    if (typeof valA === 'string') {{ valA = valA.toLowerCase(); valB = valB.toLowerCase(); }}
                    if (valA < valB) return sortAsc ? -1 : 1;
                    if (valA > valB) return sortAsc ? 1 : -1;
                    return 0;
                }});
                renderTable();
            }}

            function applyFilter(type) {{
                if(type === 'all') currentData = [...rawData];
                else if(type === 'active') currentData = rawData.filter(d => d.status.includes('🔥'));
                else if(type === 'rs80') currentData = rawData.filter(d => d.rs >= 80);
                sortBy(sortCol);
            }}

            window.onload = () => {{ sortBy('rs'); loadContent('SPY'); }};
        </script>
    </body>
    </html>"""

    print("✅ 運算完成，將網頁回傳給瀏覽器！")
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


