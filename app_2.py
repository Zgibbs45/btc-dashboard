import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from PIL import Image
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from datetime import timezone
import os

st.set_page_config(layout="wide")

NEWS_API_KEY = "ea839db4d45f4bea8e06e5b38762d5f1"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAKZq2AEAAAAA7Gx62rJuSBtFw61J5xfNQSsLpsA%3DFdrBHcVYJGPwKUUO5PnQ6nKtYA2s2hVb5T3auWwLbnphdhRpIr"

SEC_FACTS = {
    "Bitcoin Held": [
        "us-gaap:CryptoAssetNumberOfUnits"
    ],
    "Revenue": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax"
    ],
    "General and Administrative Expense": [
        "us-gaap:GeneralAndAdministrativeExpense"
    ],
    "Cost of Revenue": [
        "us-gaap:CostOfRevenue",
        "us-gaap:CostsAndExpenses"
    ],
    "Cash & Equivalents": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue"
    ],
    "Earnings Per Share (Basic)": [
        "us-gaap:EarningsPerShareBasic",
        "us-gaap:IncomeLossFromContinuingOperationsPerBasicShare",
        "us-gaap:BusinessAcquisitionProFormaEarningsPerShareBasic"
    ],
    "Earnings Per Share (Diluted)": [
        "us-gaap:EarningsPerShareDiluted",
        "us-gaap:IncomeLossFromContinuingOperationsPerDilutedShare",
        "us-gaap:BusinessAcquisitionProFormaEarningsPerShareDiluted"
    ]
}

range_options = {
    "1 Day": "1d", "1 Week": "5d", "1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y"
}

competitor_tickers = ["MARA", "RIOT", "CIFR", "HUT", "BITF"]

# --- Helper Functions ---

def format_timestamp(iso_string):
    dt = date_parser.parse(iso_string).astimezone(timezone.utc)
    now = datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
    delta = now - dt

    # Format like "June 18, 12:34 PM"
    nice_time = dt.strftime("%B %d, %I:%M %p").lstrip("0").replace(" 0", " ")

    # Format relative time
    seconds = int(delta.total_seconds())
    if seconds < 60:
        relative = f"{seconds} sec ago"
    elif seconds < 3600:
        minutes = seconds // 60
        relative = f"{minutes} min ago"
    elif seconds < 86400:
        hours = seconds // 3600
        relative = f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = seconds // 86400
        relative = f"{days} day{'s' if days != 1 else ''} ago"

    return f"{nice_time} ({relative})"

@st.cache_data(ttl=300) # 5 minutes
def get_coingecko_btc_data():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true"
    }
    try:
        resp = requests.get(url, params=params)
        data = resp.json()["bitcoin"]
        return {
            "price": data["usd"],
            "market_cap": data["usd_market_cap"],
            "volume": data["usd_24h_vol"],
            "change": data["usd_24h_change"]
        }
    except:
        st.warning("Failed to fetch CoinGecko data.")
        return {}

def get_history(_ticker, period):
    interval = "5m" if period == "1d" else "1d"
    return _ticker.history(period=period, interval=interval)

@st.cache_data(ttl=1800) # 30 minutes
def get_news(query, exclude=None, sort_by="popularity", page_size=10, from_days=30, page=1, domains=None):
    term = f"{query} -{exclude}" if exclude else query
    from_date = (datetime.now() - timedelta(days=from_days)).strftime("%Y-%m-%d")
    
    url = (
        f"https://newsapi.org/v2/everything?q={term}&from={from_date}&sortBy={sort_by}"
        f"&language=en&pageSize={page_size}&page={page}&apiKey={NEWS_API_KEY}"
    )
    
    if domains:
        domain_str = ",".join(domains)
        url += f"&domains={domain_str}"

    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            st.warning(f"NewsAPI Error {resp.status_code}: {resp.json().get('message', 'Unknown error')}")
            return []
        return resp.json().get("articles", [])
    except Exception as e:
        st.error(f"NewsAPI request failed: {e}")
        return []


def regulatory_article_filter(article):
    t = (article.get("title") or "").strip().lower()
    d = (article.get("description") or "").strip().lower()
    content = f"{t} {d}"
    crypto_terms = [
        "crypto", "cryptocurrency", "bitcoin", "ethereum", "stablecoin",
        "tokenized", "defi", "blockchain", "digital asset", "web3"
    ]
    if not any(term in content for term in crypto_terms):
        return False
    regulatory_terms = [
        "sec", "cftc", "irs", "treasury", "white house", "congress",
        "senate", "legislation", "bill", "law", "regulator", "oversight",
        "framework", "compliance", "approval", "vote", "policy", "rules"
    ]
    return any(term in content for term in regulatory_terms)

def load_articles(key, query, exclude=None, from_days=30, sort_by="popularity", filter_func=None):
    PAGE_SIZE = 10
    batch = get_news(query, exclude, sort_by, PAGE_SIZE, from_days, page=1)
    if filter_func:
        batch = [a for a in batch if filter_func(a)]
    st.session_state[key] = batch[:PAGE_SIZE]
    for art in st.session_state[key]:
        st.markdown(f"**[{art['title']}]({art['url']})**")
        st.caption(f"Source: {art['source']['name']} | {art['publishedAt'][:10]}")

@st.cache_data(ttl=600)
def get_cleanspark_tweets(query_scope="CleanSpark", max_age_days=1, sort_by="likes", max_results=6):
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

    if query_scope == "CleanSpark":
        query = '("CleanSpark" OR $CLSK OR #CLSK) -is:retweet has:links'
    else:
        query = '(bitcoin OR BTC OR mining OR crypto) -is:retweet has:links'

    sort_map = {"likes": "like_count", "retweets": "retweet_count"}

    from_date = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat("T") + "Z"

    url = f"https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": 20,
        "tweet.fields": "public_metrics,created_at,author_id,entities",
        "start_time": from_date,
        "expansions": "attachments.media_keys,author_id",
        "media.fields": "url,preview_image_url,type",
        "user.fields": "username,name,profile_image_url"
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"Twitter API Error: {response.status_code} — {response.text}")
        return []

    data = response.json()
    tweets = data.get("data", [])
    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
    media_map = {m["media_key"]: m for m in data.get("includes", {}).get("media", [])}
    # ✅ Sort tweets locally
    if sort_by == "published":
        sorted_tweets = sorted(tweets, key=lambda t: t["created_at"], reverse=True)
    else:
        sorted_tweets = sorted(
            tweets,
            key=lambda t: t["public_metrics"].get(sort_map.get(sort_by, "like_count"), 0),
            reverse=True
        )[:max_results]

    results = []
    for tweet in sorted_tweets:
        user = users.get(tweet["author_id"])
        media_urls = [
            media_map[m].get("url") or media_map[m].get("preview_image_url")
            for m in tweet.get("attachments", {}).get("media_keys", []) if m in media_map
        ]
        results.append({
            "text": tweet["text"],
            "username": user["username"] if user else "",
            "name": user["name"] if user else "",
            "profile_img": user.get("profile_image_url", "") if user else "",
            "created_at": tweet["created_at"][:10],
            "likes": tweet["public_metrics"]["like_count"],
            "retweets": tweet["public_metrics"]["retweet_count"],
            "tweet_id": tweet["id"],
            "media": media_urls
        })

    return results

@st.cache_data(ttl=300)
def get_competitor_prices(symbols):
    try:
        data = yf.download(
            tickers=" ".join(symbols),
            period="2d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True
        )
    except Exception as e:
        st.error(f"Failed to fetch batch stock data: {e}")
        return []

    results = []

    for sym in symbols:
        try:
            df = data[sym] if len(symbols) > 1 else data
            if df.empty:
                continue

            if len(df) >= 2:
                prev_close = df["Close"].iloc[-2]
                latest_close = df["Close"].iloc[-1]
            else:
                prev_close = df["Open"].iloc[0]
                latest_close = df["Close"].iloc[0]

            change = ((latest_close - prev_close) / prev_close) * 100

            results.append({
                "symbol": sym,
                "price": latest_close,
                "change": change
            })
        except Exception as e:
            st.warning(f"Error processing {sym}: {e}")
            continue

    return results

@st.cache_data(ttl=900)
def get_latest_sec_fact_with_fallback(cik, tags, year_cutoff=2024, expected_duration=90, tolerance=10):
    cik_clean = str(int(cik)).zfill(10)
    headers = {"User-Agent": "CleanSpark Dashboard <zgibbs@cleanspark.com>"}

    for tag in tags:
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik_clean}/{tag.replace(':', '/')}.json"
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                continue

            data = resp.json()
            units = data.get("units", {})

            for unit, values in units.items():
                filtered = []
                for v in values:
                    val = v.get("val")
                    if val is None:
                        continue

                    end = v.get("end") or v.get("date")
                    try:
                        end_year = int(end[:4])
                    except:
                        continue
                    if end_year < year_cutoff:
                        continue

                    start = v.get("start")
                    if start:
                        try:
                            d_start = date_parser.parse(start)
                            d_end = date_parser.parse(end)
                            duration = (d_end - d_start).days
                            if abs(duration - expected_duration) > tolerance:
                                continue
                        except:
                            continue

                    scale = (v.get("scale") or "").lower()
                    scale_factors = {
                        "thousands": 1_000,
                        "millions": 1_000_000,
                        "billions": 1_000_000_000
                    }
                    val *= scale_factors.get(scale, 1)
                    accn = v.get("accn")
                    filtered.append((val, end, accn))

                if filtered:
                    return sorted(filtered, key=lambda x: x[1], reverse=True)[0]

        except Exception as e:
            print(f"Error fetching {tag}: {e}")

    return None, None, None
    
@st.cache_data(ttl=86400)  # cache for 1 day
def load_cik_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "CleanSpark Dashboard <zgibbs@cleanspark.com>"}
    try:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        # Build a simple lookup: {"MARA": "0001507605", ...}
        return {item["ticker"]: str(item["cik_str"]).zfill(10) for item in data.values()}
    except Exception as e:
        st.error(f"Failed to load SEC ticker data: {e}")
        return {}

cik_map = load_cik_map()

@st.cache_data(ttl=86400)
def get_latest_edgar_inline_url(cik):
    cik_clean = str(int(cik)).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_clean}.json"
    headers = {"User-Agent": "CleanSpark Dashboard <zgibbs@cleanspark.com>"}
    
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return None

        data = resp.json()
        recent = data["filings"]["recent"]
        accession_numbers = recent["accessionNumber"]
        primary_docs = recent["primaryDocument"]
        forms = recent["form"]

        for accn, doc, form in zip(accession_numbers, primary_docs, forms):
            if form in ("10-K", "10-Q") and doc.endswith(".htm"):
                accn_nodash = accn.replace("-", "")
                return f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{int(cik)}/{accn_nodash}/{doc}"
    except Exception as e:
        print("EDGAR inline link fetch failed:", e)

    return None

@st.cache_data(ttl=3600)
def get_latest_press_release_metrics(company_name, ticker_symbol):
    import re

    ticker = ticker_symbol.upper()
    query = f'"{company_name}" earnings OR update OR "quarterly report"'
    articles = get_news(
        query,
        page_size=10,
        sort_by="publishedAt",
        from_days=29,
        domains=["globenewswire.com", "nasdaq.com", "seekingalpha.com", "markets.businessinsider.com"]
    )

    if not articles:
        print("❌ No articles returned by NewsAPI.")
    else:
        print(f"✅ Found {len(articles)} articles:")
        for i, art in enumerate(articles):
            print(f"{i+1}. {art.get('title')} — {art.get('source', {}).get('name')}")

    # (Optional) Temporarily skip the filter to inspect full list
    article = articles[0]  # Force display of top result for testing

    keywords = ["update", "quarter"]
    article = next(
        (
            a for a in articles
            if (
                ticker in (a.get("title", "") + a.get("description", "")).upper()
                or company_name.upper() in (a.get("title", "") + a.get("description", "")).upper()
            )
        ),
        None
    )

    if not article:
        return None

    title = article.get("title", "")
    description = article.get("description", "")
    content = f"{title} {description}".lower()
    published_date = article.get("publishedAt", "")[:10]
    url = article.get("url", "#")

    metrics = {}

    patterns = {
        "Bitcoin Held": r"(?:held(?: a total of)?|holdings(?: total)?)[^\d]{0,20}([\d,]+)\s*(?:btc|bitcoin)|(?:btc|bitcoin)[^\d]{0,20}([\d,]+)",
        "Revenue": r"revenue(?: (?:of|was))?[\s:]*\$([\d,\.]+)",
        "Cash & Equivalents": r"(?:cash and cash equivalents|cash)(?: (?:totaled|of))?[\s:]*\$([\d,\.]+)",
        "Earnings Per Share (Basic)": r"(?:earnings per share|eps)(?: \(basic\))?[\s:]*\$([\d\.]+)",
        "Earnings Per Share (Diluted)": r"(?:earnings per share|eps)(?: \(diluted\))?[\s:]*\$([\d\.]+)"
    }

    for label, pattern in patterns.items():
        matches = re.findall(pattern, content)

        # Flatten all match groups and keep only digits/decimals
        numbers = [g for pair in matches for g in pair if g.strip().replace(',', '').replace('.', '').isdigit()]

        if numbers:
            try:
                if "Bitcoin" in label:
                    val = max(int(n.replace(",", "")) for n in numbers)
                    metrics[label] = f"{val:,} BTC"
                else:
                    val = max(float(n.replace(",", "")) for n in numbers)
                    metrics[label] = f"${val:,.2f}"
            except ValueError:
                continue

    return {
        "title": title,
        "url": url,
        "date": published_date,
        "metrics": metrics
    }

# --- Tabs and Layout Config ---
st.title("Legal & Market Dashboard")
edgar.set_identity("CleanSpark Dashboard <zgibbs@cleanspark.com>")


with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)  # spacing
    logo = Image.open("cleanspark_logo.png")
    st.image(logo, width=300)  # adjust width as needed
    st.markdown("<hr>", unsafe_allow_html=True)
tab = st.sidebar.selectbox("Select a page", ["News", "📈 Market & Competition"])
if tab == "News":
    
    day_options = {
        "1 Day": 1,
        "3 Days": 3,
        "1 Week": 7,
        "1 Month": 30
    }

    sort_by_map = {
        "Popularity": "popularity",
        "Published": "publishedAt",
    }
    btc_metrics = get_coingecko_btc_data()
    btc = yf.Ticker("BTC-USD")

    st.subheader("📈 Bitcoin Market Stats")
    sel = st.pills("Bitcoin price range:", options=list(range_options.keys()), default="1 Day", key="btc_range")
    selected_range = range_options.get(sel, "1mo")
    data = get_history(btc, selected_range)

    if not data.empty:
        m1, m2, m3 = st.columns([.80, 1.20, 1])
        with m1:
            st.metric(
                "Bitcoin Price (USD)",
                f"${btc_metrics.get('price', 0):,.2f}",
                f"{btc_metrics.get('change', 0):.2f}%"
            )
        with m2:
            st.metric(
                "Market Cap",
                f"${btc_metrics.get('market_cap', 0):,.0f}"
            )
        with m3:
            st.metric(
                "24h Volume",
                f"${btc_metrics.get('volume', 0):,.0f}"
            )
        st.line_chart(data["Close"].round(2).rename("Bitcoin Price"))

    col1, col2 = st.columns([1.8,2.2])

    with col1:
        st.subheader("🐦 Twitter Feed")

        # Row 1: Scope pill (full width)
        tw_scope = st.pills("Tweet Scope:", ["All Bitcoin", "CleanSpark Only"], default="CleanSpark Only", key="tw_scope")

        # Row 2: Time + Sort filters side-by-side
        m1, m2 = st.columns([1, 1])

        with m1:
            tw_days = st.pills("Tweets from the past...", ["1 Day", "3 Days", "1 Week"], default="1 Day", key="tw_days")

        with m2:
            tw_sort = st.pills("Sort tweets by:", ["Likes", "Retweets", "Published"], default="Likes", key="tw_sort")
            
        tw_scope_val = "CleanSpark" if tw_scope == "CleanSpark Only" else "General"
        tw_days_map = {"1 Day": 1, "3 Days": 3, "1 Week": 7}
        tw_max_days = tw_days_map[tw_days]

        tweets = get_cleanspark_tweets(
            query_scope=tw_scope_val,
            max_age_days=tw_max_days,
            sort_by=tw_sort,
            max_results=6
        )

        # Render tweets
        for tweet in tweets:
            with st.container():
                st.markdown(f"**[{tweet['name']}](https://twitter.com/{tweet['username']})** • @{tweet['username']} • *{format_timestamp(tweet['created_at'])}*")
                st.markdown(tweet["text"])
                st.markdown(f"🔁 {tweet['retweets']} &nbsp;&nbsp;&nbsp; ❤️ {tweet['likes']}")
                st.markdown(f"[View on Twitter](https://twitter.com/{tweet['username']}/status/{tweet['tweet_id']})")
                for url in tweet["media"]:
                    if url.lower().endswith((".jpg", ".png", ".jpeg")):
                        st.image(url, use_column_width=True)
                    elif url.lower().endswith((".mp4", ".mov", ".webm")):
                        st.video(url)
                    else:
                        st.markdown(f"[View Media]({url})")

    # General News
        with col2:
            st.subheader("📰 Bitcoin News")

            # Pills for filtering
            scope_options = ["All Bitcoin", "CleanSpark Only", "Regulatory Only"]
            gen_scope = st.pills("Article Scope:", scope_options, default="All Bitcoin", key="gen_scope")

            gen_col1, gen_col2 = st.columns([1, 1])
            with gen_col1:
                gen_days = st.pills("Articles from the past...", list(day_options.keys()), default="1 Day", key="gen_days")
            with gen_col2:
                gen_sort = st.pills("Sort by:", list(sort_by_map.keys()), default="Popularity", key="gen_sort")

            # Handle query + filtering logic
            if gen_scope == "CleanSpark Only":
                query_term = "CleanSpark"
                exclude_term = None
                filter_func = None
            elif gen_scope == "Regulatory Only":
                query_term = "GENIUS Act OR cryptocurrency legislation OR crypto bill OR digital asset policy"
                exclude_term = None
                filter_func = regulatory_article_filter
            else:  # All Bitcoin
                query_term = "bitcoin mining"
                exclude_term = "CleanSpark"
                filter_func = None

            # Pass in correct params
            gen_from_days = day_options[gen_days]
            gen_sort_by = sort_by_map[gen_sort]

            load_articles(
                key="gen_articles",
                query=query_term,
                exclude=exclude_term,
                from_days=gen_from_days,
                sort_by=gen_sort_by,
                filter_func=filter_func
            )
                    
# --- HOME TAB ---
if tab == "📈 Market & Competition":
    btc_metrics = get_coingecko_btc_data()
    btc         = yf.Ticker("BTC-USD")
    col1, col2  = st.columns([2,2])
    sym = st.session_state.get("stock_lookup_ticker", "MARA").strip().upper()
    ticker_obj = yf.Ticker(sym)
    try:
        info = ticker_obj.get_info()
        company_name = info.get("longName", sym)
    except Exception:
        company_name = sym
    # Stock
    with col1:
        st.subheader(f"📊 Stock Market Lookup: {company_name}")        
        m1, m2 = st.columns([1.5, 2.5])
        with m1:
            sym = st.text_input("Stock ticker:", "MARA", key="stock_lookup_ticker").strip().upper()
            cik = cik_map.get(sym.upper())
            entered_ticker = sym.upper()
        
        with m2:
            lookup_range = st.pills("Timeframe:", options=list(range_options.keys()), default="1 Day", key="lookup_range")

        selected_range = range_options.get(lookup_range, "1mo")

        # Fetch data
        ticker_obj = yf.Ticker(sym)

        try:
            info = ticker_obj.get_info()
        except Exception:
            info = {
                "regularMarketPrice": None,
                "open": None,
                "dayHigh": None,
                "dayLow": None,
                "trailingPE": None,
                "fiftyTwoWeekHigh": None,
                "fiftyTwoWeekLow": None,
                "volume": None,
                "averageVolume": None,
                "marketCap": None
            }
            st.warning(f"⚠️ Could not fetch company info for ticker: `{sym}`. Displaying fallback values.")
        try:
            df = ticker_obj.history(period=selected_range, interval="1d")
        except Exception:
            df = None
            st.warning(f"⚠️ No historical data found for `{sym}`.")

        # ROW 2: Show current price on its own line
        price = info.get("regularMarketPrice")
        change_amount = info.get("regularMarketChange")
        change_percent = info.get("regularMarketChangePercent")
        clsk_price = clsk_open = clsk_high = clsk_low = None
        
        if sym != "CLSK":
            clsk = yf.Ticker("CLSK")
            df = ticker_obj.history(period=selected_range, interval="1d")
            try:
                clsk_info = clsk.get_info()
            except Exception:
                clsk_info = {
                    "regularMarketPrice": None,
                    "open": None,
                    "dayHigh": None,
                    "dayLow": None
                }
                st.warning("⚠️ Could not fetch CLSK data. Displaying fallback values.")

            clsk_price = clsk_info.get("regularMarketPrice")
            clsk_open  = clsk_info.get("open")
            clsk_high  = clsk_info.get("dayHigh")
            clsk_low   = clsk_info.get("dayLow")

        # Layout: start with m1 already occupied by st.metric("Current Price")
        m1, m2, m3, m4, m5 = st.columns([1.4, 1.2, 1.2, 1.2, 1.2])

        def render_metric_block(label, value, delta=None, reference=None, show_arrow=False):
            if value is None:
                return f"<div style='font-size:16px; font-weight:600; margin-bottom:4px;'>{label}</div><div>-</div>"

            color = "green" if (delta or 0) >= 0 else "red"
            delta_html = ""

            if delta is not None and reference:
                pct = abs(delta / reference * 100) if reference != 0 else 0
                arrow = "🔺" if delta >= 0 else "🔻"
                change_text = f"{arrow} ${abs(delta):.2f} ({pct:.1f}%)" if show_arrow else f"${abs(delta):.2f} ({pct:.1f}%)"
                delta_html = f"<div style='color:{color}; font-size:0.85em'>{change_text}</div>"

            return f"""
                <div style='font-size:16px; font-weight:600; margin-bottom:4px;'>{label}</div>
                <div style='font-size:24px; font-weight:bold;'>${value:.2f}</div>
                {delta_html}
            """
        with m1:
            st.markdown(render_metric_block("Current Price", price, delta=change_amount, reference=price, show_arrow=True), unsafe_allow_html=True)

        with m2:
            st.markdown(render_metric_block("CLSK Price", clsk_price), unsafe_allow_html=True)

        with m3:
            st.markdown(render_metric_block("CLSK Open", clsk_open), unsafe_allow_html=True)

        with m4:
            st.markdown(render_metric_block("CLSK High", clsk_high), unsafe_allow_html=True)

        with m5:
            st.markdown(render_metric_block("CLSK Low", clsk_low), unsafe_allow_html=True)

        if "regularMarketPrice" not in info or df.empty:
            st.warning(f"No data available for ticker `{sym}`.")
        else:
            # Format large numbers
            def fmt_float(val): return f"{val:.2f}" if val is not None else "N/A"
            def fmt_millions(val): return f"{val/1_000_000:.2f} M" if val else "N/A"
            def fmt_billions(val): return f"{val/1_000_000_000:.2f} B" if val else "N/A"

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"**Open**: {fmt_float(info.get('open'))}")
                st.markdown(f"**High**: {fmt_float(info.get('dayHigh'))}")
                st.markdown(f"**Low**: {fmt_float(info.get('dayLow'))}")
            with m2:
                st.markdown(f"**P/E**: {fmt_float(info.get('trailingPE'))}")
                st.markdown(f"**52wk High**: {fmt_float(info.get('fiftyTwoWeekHigh'))}")
                st.markdown(f"**52wk Low**: {fmt_float(info.get('fiftyTwoWeekLow'))}")
            with m3:
                st.markdown(f"**Vol**: {fmt_millions(info.get('volume'))}")
                st.markdown(f"**Avg Vol**: {fmt_millions(info.get('averageVolume'))}")
                st.markdown(f"**Mkt Cap**: {fmt_billions(info.get('marketCap'))}")
        try:
            interval = "5m" if selected_range == "1d" else "1d"
            df = ticker_obj.history(period=selected_range, interval=interval)
            df.index = pd.to_datetime(df.index)

            if not df.empty and "Close" in df.columns:
                st.line_chart(df["Close"].round(2).rename(f"{sym} Price"))
            else:
                st.warning("No price data available for this range.")
        except Exception as e:
            st.error(f"Error loading chart data: {e}")

        with col2:
            st.subheader("Live Competiton View")
            competitors = get_competitor_prices(competitor_tickers)

            m1, m2, m3, m4, m5 = st.columns(5)
            col_map = [m1, m2, m3, m4, m5]

            for i, comp in enumerate(competitors):
                if i >= len(col_map):
                    break  # safety check if there are fewer than 5 columns defined
                with col_map[i]:
                    arrow = "🔺" if comp["change"] >= 0 else "🔻"
                    color = "green" if comp["change"] >= 0 else "red"
                    st.markdown(
                        f"""
                        <div style='font-size:16px; text-align: center; padding: 8px 0;'>
                            <strong>{comp['symbol']}</strong><br>
                            ${comp['price']:.2f}<br>
                            <span style='color:{color};'>{arrow} {comp['change']:.2f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col2:
                st.subheader("📄 Latest Press Filing & SEC Filing Metrics")
                ticker_upper = sym.upper()
                cik = cik_map.get(ticker_upper)
                if cik:
                    inline_url = get_latest_edgar_inline_url(cik)
                    if inline_url:
                        st.markdown("---")
                        st.markdown(f"🔗 [**View the most recent SEC filing in EDGAR**]({inline_url})")
                    else:
                        st.info("No recent EDGAR filing link found.")
                    from collections import Counter

                    facts = {}
                    dates = []
                    latest_accn = None  # to store one accession number for the filing link

                    for label, tags in SEC_FACTS.items():
                        val, end_date, accn = get_latest_sec_fact_with_fallback(cik, tags)
                        if end_date:
                            dates.append(end_date)
                        if not latest_accn and accn:
                            latest_accn = accn  # store the first one
                        facts[label] = (val, end_date)

                    # Most common reporting date across all facts
                    if dates:
                        common_date = Counter(dates).most_common(1)[0][0]
                    else:
                        common_date = None

                    for label, (val, date) in facts.items():
                        if date == common_date:
                            if val is not None:
                                if label == "Bitcoin Held":
                                    formatted_val = f"{val:,.0f} BTC"
                                else:
                                    formatted_val = f"(${abs(val):,.2f})" if val < 0 else f"${val:,.2f}"
                                st.markdown(f"- **{label}**: {formatted_val} _(Period ending {date})_")
                            else:
                                st.markdown(f"- **{label}**: *Not available*")
                        else:
                            st.markdown(f"- **{label}**: *Not available*")
                else:
                    st.warning("CIK not found for this ticker.")
                st.markdown("---")
            st.subheader("📢 Press Release Metrics")

            company_name = info.get("longName") or ticker_upper
            press_data = get_latest_press_release_metrics(company_name, ticker_upper)

            if press_data:
                st.markdown(f"📰 [**{press_data['title']}**]({press_data['url']}) — *{press_data['date']}*")
                if press_data["metrics"]:
                    for label, val in press_data["metrics"].items():
                        st.markdown(f"- **{label}**: {val}")
                else:
                    st.info("No clean financial metrics found in the press release.")
            else:
                st.info("No recent press release found.")
