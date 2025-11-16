# Finance Analysis CLI

互動式美股財報分析工具，適用於 WSL（Ubuntu）+ VS Code。  
使用 [Financial Modeling Prep](https://financialmodelingprep.com/) API 取得財報與股價資料，並可選用 OpenAI Chat API 摘要財報逐字稿。

## 功能概述

- 以公司名稱或代號搜尋美股標的並選擇
- 選擇「最新財報」或「指定季度（Year + Quarter）」進行分析
- 顯示：
  - 收益表 / 資產負債表 / 現金流量表 YoY & QoQ 變化
  - EPS / 營收驚喜與公告時點（BMO/AMC）
  - 財報後 T+1 / T+3 / T+7 價格事件研究
  - 財報逐字稿要點摘要（OpenAI）或關鍵詞統計（fallback）
- 產出 Markdown 報告：`reports/{SYMBOL}_{FY}Q{Q}.md`

## WSL 初始化與執行（CLI）

在 WSL（Ubuntu）終端機中：

```bash
cd /path/to/finance-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 必要：FMP API Key
export FMP_API_KEY="YOUR_FMP_KEY"

# 可選：用於逐字稿 LLM 摘要
export OPENAI_API_KEY="YOUR_OPENAI_KEY"

# 互動式 CLI 模式
python3 app.py

# 或使用參數直接指定季度
python3 app.py --symbol AAPL --latest
python3 app.py --symbol MSFT --year 2024 --quarter 4

## 啟動 Web 版（FastAPI）

安裝好依賴並設定好 `FMP_API_KEY` 後，可啟動 Web 伺服器：

```bash
uvicorn web_app:app --reload --host 0.0.0.0 --port 8000
```

瀏覽器開啟 `http://localhost:8000`：

- 在首頁輸入公司名稱或代號關鍵字並搜尋
- 在搜尋結果中選擇標的
- 在季度頁面選擇欲分析的季度並按「執行分析」
- 分析結果（財報重點、EPS 驚喜、價格反應、逐字稿重點）會直接呈現在網頁上  
  同時會在伺服器端輸出對應的 Markdown 報告到 `reports/{SYMBOL}_{FY}Q{Q}.md`
```

## CLI 主要參數

- `--symbol SYMBOL`：指定股票代號（如：`AAPL`）
- `--year YEAR` / `--quarter Q`：指定財報年與季（如：`2024`、`4`）
- `--latest / --no-latest`：是否以最新財報為主（預設：`--latest`）
- `--save / --no-save`：是否輸出 Markdown 報告（預設：`--save`）
- `--plot / --no-plot`：是否產生事件窗折線圖（預設：`--plot`；若無圖形環境或未安裝 matplotlib 則自動略過）

## 驗收建議流程

1. **AAPL 最新財報**
   - 執行：`python3 app.py --symbol AAPL --latest`
   - 檢查：
     - EPS / Revenue estimate vs actual 與驚喜率
     - 公告日期與 BMO / AMC 標記
     - T+1 / T+3 / T+7 累積報酬

2. **MSFT 指定季度（例如 2024 Q4）**
   - 執行：`python3 app.py --symbol MSFT --year 2024 --quarter 4`
   - 檢查：
     - 三大報表是否對齊該季
     - YoY / QoQ 變化是否合理
     - FCF（自由現金流）計算是否存在且數值合理

3. **逐字稿缺失情境**
   - 選擇一檔逐字稿可能缺失或較久以前的標的
   - 確認程式會：
     - 清楚提示逐字稿不可用
     - 仍能顯示財報與價格分析
     - 允許改選其他季度（互動式模式）

## 推上 GitHub 並佈署到 Zeabur

1. 初始化 git 並推到 GitHub（範例）：

   ```bash
   git init
   git add .
   git commit -m "Add finance analysis CLI & web"
   git branch -M main
   git remote add origin git@github.com:YOUR_NAME/finance-analysis.git
   git push -u origin main
   ```

2. 在 Zeabur 建立服務：

   - 新增一個 Web Service，來源選擇剛推上的 GitHub 專案
   - Build Command：`pip install -r requirements.txt`
   - Start Command：`uvicorn web_app:app --host 0.0.0.0 --port $PORT`
   - Runtime 選擇 Python 3.10+（或對應版本）

3. 在 Zeabur 設定環境變數：

   - `FMP_API_KEY=YOUR_FMP_KEY`
   - （可選）`OPENAI_API_KEY=YOUR_OPENAI_KEY`

4. 部署完成後，打開 Zeabur 提供的 URL，即可在瀏覽器上使用同樣的搜尋 / 選季度 / 分析流程。
