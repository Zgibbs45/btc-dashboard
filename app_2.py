import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import base64
import os
import pytz
import time
import re
import html
import altair as alt
import math
from zoneinfo import ZoneInfo
from pycoingecko import CoinGeckoAPI
from PIL import Image
import csv, os
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from dateutil.parser import parse as parse_date
from urllib.parse import quote

#STYLES
st.set_page_config(layout="wide")

st.markdown("""
<style>
.article-title { margin: 0; font-weight: 600; font-size: 16px; line-height: 1.35; }
.article-title a { text-decoration-thickness: 1px; }
.article-meta  { color:#6b7280; font-size:13px; margin:2px 0 0; } /* no extra gap */

.article-actions{
  display:flex; justify-content:flex-end; align-items:center; height:100%;
  margin:0; padding:0;           /* keeps pill on the same baseline as title */
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Lock the sidebar width */
    [data-testid="stSidebar"] {
        background-color: #D5EDF8 !important;
    }

    /* Prevent resizer from showing on hover */
    [data-testid="stSidebar"] + div [data-testid="stResizer"] {
        display: none !important;
    }

    /* Sidebar pills/buttons */
    .st-emotion-cache-1avcm0n, .st-emotion-cache-16txtl3 {
        background-color: #00C2D6 !important;
        color: white !important;
        font-weight: bold;
    }

    /* Header */
    h1 {
        color: #102A43;
    }

    /* Global font */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        font-size: 15px;
        line-height: 1.6;
    }

    /* Sidebar "Select a page" label */
    [data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: bold;
    }

    /* Sidebar selectbox/dropdown styling */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #102A43 !important;
        color: #EEEFF3 !important;
        border-radius: 5px;
    }

    /* Input text (search box inside dropdown) */
    [data-testid="stSidebar"] input {
        color: #EEEFF3 !important;
    }

    /* Dropdown list background */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] {
        background-color: #102A43 !important;
        color: #EEEFF3 !important;
    }

    /* GLOBAL FLAT CORNER STYLE */
    button, input, select, textarea,
    div[data-baseweb="select"],
    .stButton > button,
    .stTextInput > div,
    .stSelectbox > div,
    .stMultiSelect > div,
    .stDateInput > div,
    .stRadio > div,
    .stSlider > div {
        border-radius: 4px !important;  /* Use 0px for square, 4px for slight curve */
        box-shadow: none !important;
    
    }
    
    /* Hide anchor link icons from all headers */
    h1 a, h2 a, h3 a, h4 a {
        display: none !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

#Tweet Image Format
st.markdown("""
<style>
/* Cap each tweet's content width (similar to Twitter’s ~680px) */
.tweet-block { max-width: 680px; }

<style>
/* Prevent long tokens (URLs, @mentions chains) from spilling into the next column */
.tweet-block, .tweet-block * {
  overflow-wrap: anywhere;   /* modern */
  word-break: break-word;    /* fallback */
  hyphens: auto;
}
</style>

/* Responsive media grid */
.media-grid { display: grid; gap: 6px; margin-top: 8px; width: 100%; }
.media-grid.cols-1 { grid-template-columns: 1fr; }
.media-grid.cols-2 { grid-template-columns: 1fr 1fr; }
.media-grid.cols-3 { grid-template-columns: 2fr 1fr; grid-auto-rows: 1fr; }
.media-grid.cols-4 { grid-template-columns: 1fr 1fr; }

/* Cropped tiles with smooth scaling on zoom */
.media { position: relative; overflow: hidden; border-radius: 12px; background: #f1f5f9; }
.media img { width: 100%; height: 100%; object-fit: cover; display: block; }

/* Aspect ratios that mimic Twitter’s crops */
.media.ratio-16x9 { aspect-ratio: 16 / 9; }
.media.ratio-4x5 { aspect-ratio: 4 / 5; }
.media.ratio-1x1 { aspect-ratio: 1 / 1; }

@media (max-width: 720px) { .tweet-block { max-width: 100%; } }
</style>
""", unsafe_allow_html=True)

#Tweet Report Pill
st.markdown("""
<style>
/* Small pill for popover button */
div[data-testid="stPopover"] > button {
  padding: 2px 8px;
  min-height: 26px; min-width: 26px;
  border-radius: 9999px;
  font-size: 16px; line-height: 1;
  background: #f8fafc; border: 1px solid #e2e8f0;
}

/* Fallback: compact look for expander summary if popover isn't available */
details > summary {
  list-style: none; cursor: pointer;
  border: 1px solid #e2e8f0; border-radius: 9999px;
  padding: 2px 8px; font-size: 16px; line-height: 1; background: #f8fafc;
  display: inline-block;
}
details[open] > summary { background: #eef2f7; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* POPover: match the real button (not necessarily a direct child) */
div[data-testid="stPopover"] button {
  padding: 4px 12px !important;
  min-width: 44px !important;    /* visibly wider */
  min-height: 28px !important;
  border-radius: 9999px !important;
  display: inline-flex !important;
  align-items: center; justify-content: center;
}

/* Expander fallback: hide default triangle + match pill sizing */
details > summary::-webkit-details-marker { display: none; }
details > summary::marker { content: ""; }
details > summary {
  padding: 4px 12px !important;
  min-width: 44px !important;
  min-height: 28px !important;
  border-radius: 9999px !important;
  display: inline-flex !important;
  align-items: center; justify-content: center;
}
</style>
""", unsafe_allow_html=True)


ET = pytz.timezone("America/New_York")

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
TWITTER_BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
COINGECKO_API_KEY = st.secrets["COINGECKO_API_KEY"]

SEC_FACTS = {
    "Adjusted EBITDA": [],
    "Adjusted EBITDA Margin": [],
    "Bitcoin Held": [
        "us-gaap:CryptoAssetNumberOfUnits"
    ],
    "Cash & Equivalents": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue"
    ],
    "Cost of Goods Sold": [],
    "Cost of Revenue": [
        "us-gaap:CostOfRevenue",
        "us-gaap:CostsAndExpenses"
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
    ],
    "EBIT Margin": [],
    "EH/s": [],
    "General and Administrative Expense": [
        "us-gaap:GeneralAndAdministrativeExpense"
    ],
    "Gross Profit Margin": [],
    "Long Term Debt": [],
    "Net Income": [],
    "Net Income Margin": [],
    "Operating Profit": [],
    "Revenue": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax"
    ],
    "Total Shareholder Equity": []
}

range_options = {
    "1 Day": "1d",
    "1 Week": "7d",
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
}

competitor_tickers = ["CLSK", "BITF", "BTDR", "CANG", "CIFR", "CORZ", "HIVE", "HUT", "IREN", "MARA", "MTPLF", "RIOT", "WULF"]

# --- Helper Functions ---

def github_create_issue(title: str, body: str, labels=None, assignees=None):
    token = st.secrets.get("ISSUE_TOKEN")
    repo  = st.secrets.get("GITHUB_REPO")
    if not token or not repo:
        return False, "missing-secrets"

    url = f"https://api.github.com/repos/{repo}/issues"
    payload = {"title": title, "body": body}
    if labels: payload["labels"] = labels
    if assignees: payload["assignees"] = assignees

    try:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=payload,
            timeout=10,
        )
        if r.status_code >= 300:
            return False, f"status={r.status_code} body={r.text[:200]}"
        return True, None
    except Exception as e:
        return False, str(e)

def _short(s: str, n: int = 72) -> str:
    s = re.sub(r"\s+", " ", (s or "").replace("\n", " ")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def _fmt_tweet_option(t: dict) -> str:
    who = t.get("name") or t.get("username") or "Unknown"
    text = re.sub(r"https://t\.co/\S+$", "", t.get("text", "")).strip()
    return f'{who} — "{_short(text)}"'

def _fmt_article_option(a: dict) -> str:
    src   = (a.get("source") or {}).get("name", "Source")
    title = a.get("title", "")
    return f"{src} — {_short(title)}"

def render_section_report_pill(*, section_key: str, items: list[dict], kind: str, page: str):
    """
    Renders a single 'Report' pill that opens a form.
    kind: 'tweet' or 'article'
    items: the items currently shown in the section (what the user sees)
    """
    pop = getattr(st, "popover", None)
    ctx = pop("Report") if callable(pop) else st.expander("Report", expanded=False)

    with ctx:
        label = f"What {kind} do you want to report?"
        format_func = _fmt_tweet_option if kind == "tweet" else _fmt_article_option

        with st.form(f"report-{section_key}", clear_on_submit=True):
            selected = st.selectbox(
                label,
                options=items,
                format_func=format_func,
                key=f"select-{section_key}",
                placeholder=f"Select a {kind}…"
            )

            issue = st.selectbox(
                "Type",
                ["Spam", "Broken link", "Not relevant", "Low-quality source", "Misleading headline", "Other"],
                key=f"type-{section_key}",
            )
            note = st.text_area("Details (optional)", key=f"note-{section_key}", height=88)

            if st.form_submit_button("Send"):
                if kind == "tweet":
                    tid  = selected.get("tweet_id", "")
                    turl = f"https://x.com/i/web/status/{tid}" if tid else ""
                    log_feedback({
                        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
                        "page": page,
                        "tweet_id": tid,
                        "username": selected.get("username", ""),
                        "tweet_url": turl,
                        "issue": issue,
                        "note": (note or "").strip(),
                    })
                else:  # article
                    src = (selected.get("source") or {}).get("name", "")
                    url = selected.get("url", "")
                    title = selected.get("title", "")
                    log_feedback({
                        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
                        "page": page,
                        "tweet_id": "",              
                        "username": src,             
                        "tweet_url": url,             
                        "issue": issue + (f" — {title[:80]}" if title else ""),
                        "note": (note or "").strip(),
                    })
                st.success("Thanks! Logged.")

def format_timestamp(iso_string):
    dt = date_parser.parse(iso_string).astimezone(ZoneInfo("UTC"))
    now = datetime.now(ZoneInfo("UTC"))
    delta = now - dt

    nice_time = dt.strftime("%B %d, %I:%M %p").lstrip("0").replace(" 0", " ")

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

@st.cache_data(ttl=300)
def get_coingecko_btc_data():
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true"
    }
    headers = {
        "x-cg-pro-api-key": st.secrets["COINGECKO_API_KEY"]
    }

    try:
        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()["bitcoin"]
        return {
            "price": data["usd"],
            "market_cap": data["usd_market_cap"],
            "volume": data["usd_24h_vol"],
            "change": data["usd_24h_change"]
        }
    except Exception as e:
        st.warning("Failed to fetch CoinGecko Pro data.")
        st.text(f"Error: {e}")
        return {}

@st.cache_data(ttl=90, show_spinner=False)
def history_cached(symbol: str, *, period=None, interval="1d", start=None, end=None):
    """Centralized, memoized wrapper around yfinance.history."""
    try:
        return yf.Ticker(symbol).history(period=period, interval=interval, start=start, end=end)
    except Exception:
        return pd.DataFrame()

def get_history(_ticker, period):
    if period == "1d":
        now = datetime.now(ZoneInfo("UTC"))
        start = now - timedelta(hours=24)
        df = _ticker.history(start=start, end=now, interval="5m")
        if df is None or df.empty:
            df = _ticker.history(period="1d", interval="5m")
        return df
    else:
        interval = "5m" if period == "1d" else "1d" 
        return history_cached(_ticker.ticker, period=period, interval=interval)

@st.cache_data(ttl=1800) # 30 minutes
def get_news(query, exclude=None, sort_by="popularity", page_size=10, from_days=30, page=1, domains=None):
    term = f"{query} -{exclude}" if exclude else query
    from_date = (datetime.now(ZoneInfo("UTC")) - timedelta(days=from_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    url = (
        f"https://newsapi.org/v2/everything?q={term}&from={from_date}&sortBy={sort_by}"
        f"&language=en&pageSize={page_size}&page={page}&apiKey={st.secrets['NEWS_API_KEY']}"
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

def load_articles(key, query, exclude=None, from_days=30, sort_by="popularity", filter_func=None, pill_at="title"):
    PAGE_SIZE = 10
    batch = get_news(query, exclude, sort_by, PAGE_SIZE, from_days, page=1)
    if filter_func:
        batch = [a for a in batch if filter_func(a)]
    st.session_state[key] = batch[:PAGE_SIZE]
    for art in st.session_state[key]:
        st.markdown(
            f'<div class="article-title"><a href="{art["url"]}" target="_blank">{html.escape(art["title"])}</a></div>',
            unsafe_allow_html=True
        )
        src = (art.get("source") or {}).get("name", "")
        pub = art.get("publishedAt") or ""
        try:
            pub = pd.to_datetime(pub, utc=True).tz_convert(ET).strftime("%Y-%m-%d")
        except Exception:
            pub = pub[:10]

        st.markdown(f'<div class="article-meta">{src} · {pub}</div>', unsafe_allow_html=True)

@st.cache_data(ttl=1800)
def get_cleanspark_tweets(query_scope="CleanSpark", max_age_days=1, sort_by="likes", max_results=6):
    headers = {"Authorization": f"Bearer {st.secrets['TWITTER_BEARER_TOKEN']}"}

    if query_scope == "CleanSpark":
        query = '("CleanSpark" OR #CLSK OR CLSK) -is:retweet has:links'
    else:
        query = '(bitcoin OR BTC OR mining OR crypto) -is:retweet has:links'

    from_date = (datetime.now(ZoneInfo("UTC")) - timedelta(days=max_age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": 50,
        "tweet.fields": "public_metrics,created_at,author_id,entities",
        "start_time": from_date,
        "expansions": "attachments.media_keys,author_id",
        "media.fields": "url,preview_image_url,type",
        "user.fields": "username,name,profile_image_url"
    }

    # Retry logic
    for attempt in range(3):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            st.warning("Twitter rate limit hit. Retrying in 60 seconds...")
            time.sleep(60)
        else:
            break

    if response.status_code != 200:
        st.error(f"Twitter API Error: {response.status_code} — {response.text}")
        return []

    data = response.json()
    tweets = data.get("data", [])

    cutoff = datetime.now(ZoneInfo("UTC")) - timedelta(days=max_age_days)
    today = datetime.now(ZoneInfo("UTC")).date()
    
    if max_age_days > 1:
    # keep anything within the rolling N-day window
        filtered = []
        for t in tweets:
            if "created_at" not in t:
                continue
            dt = parse_date(t["created_at"])
            if dt >= cutoff:
                filtered.append(t)
        tweets = filtered
    else:
        tweets = [
            t for t in tweets
            if "created_at" in t and parse_date(t["created_at"]) >= cutoff
        ]

    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
    media_map = {m["media_key"]: m for m in data.get("includes", {}).get("media", [])}
    
    # Build full tweet results
    results = []
    for tweet in tweets:
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
            "created_at": tweet["created_at"],
            "likes": tweet["public_metrics"]["like_count"],
            "retweets": tweet["public_metrics"]["retweet_count"],
            "tweet_id": tweet["id"],
            "media": media_urls
        })

    # Sort locally
    if sort_by == "likes":
        results.sort(key=lambda x: x["likes"], reverse=True)
    elif sort_by == "retweets":
        results.sort(key=lambda x: x["retweets"], reverse=True)
    else:  # published
        results.sort(key=lambda x: x["created_at"], reverse=True)

    return results[:max_results]
    
def translate_text(text, api_key, target="en"):
    if not text or not text.strip():
        return text
    try:
        r = requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            params={"key": api_key},  # <-- key in query string
            json={"q": text, "target": target, "format": "text"}  # safer than form-encoded
        )
        r.raise_for_status()
        data = r.json()["data"]["translations"][0]
        # Google returns HTML-escaped text; unescape so your HTML-escaping later is clean
        return html.unescape(data["translatedText"])
    except Exception as e:
        # optional: flip this flag in session_state if you want to see errors
        if st.session_state.get("debug_translate", False):
            st.error(f"Translate failed: {e}")
        return text  # graceful fallback
    
def twitter_img_variant(url: str, size: str = "medium") -> str:
    # pbs.twimg.com returns multiple sizes via the "name=" query param
    if "pbs.twimg.com/media/" in url:
        if "name=" in url:
            url = re.sub(r"name=\w+", f"name={size}", url)
        else:
            url += ("&" if "?" in url else "?") + f"name={size}"
    return url

def render_tweet_media(urls: list[str]):
    # Separate images from videos
    imgs = [u for u in urls if u.lower().endswith((".jpg", ".jpeg", ".png")) or "pbs.twimg.com/media/" in u]
    vids = [u for u in urls if u.lower().endswith((".mp4", ".mov", ".webm"))]

    # Request a medium-sized variant for Twitter-hosted images
    imgs = [twitter_img_variant(u, "medium") for u in imgs]

    n = len(imgs)
    if n:
        cols_class = f"cols-{min(n,4)}"
        html = [f'<div class="media-grid {cols_class}">']
        if n == 1:
            html.append(f'<div class="media ratio-16x9"><img src="{imgs[0]}" loading="lazy"></div>')
        elif n == 2:
            for u in imgs:
                html.append(f'<div class="media ratio-4x5"><img src="{u}" loading="lazy"></div>')
        elif n == 3:
            # Big tile on the left, two squares on the right
            html.append(f'<div class="media ratio-4x5" style="grid-row: span 2;"><img src="{imgs[0]}" loading="lazy"></div>')
            for u in imgs[1:]:
                html.append(f'<div class="media ratio-1x1"><img src="{u}" loading="lazy"></div>')
        else:
            # First four as a 2x2 grid
            for u in imgs[:4]:
                html.append(f'<div class="media ratio-1x1"><img src="{u}" loading="lazy"></div>')
        html.append("</div>")
        st.markdown("\n".join(html), unsafe_allow_html=True)

    # Videos (keep Streamlit’s native player)
    for v in vids:
        st.video(v)


def _append_csv(row, path="data/user_feedback.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        cols = ["timestamp_utc","page","tweet_id","username","tweet_url","issue","note"]
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file: w.writeheader()
        w.writerow(row)

def log_feedback(entry: dict):
    """Always write to CSV; optionally mirror to Slack / Google Sheets if secrets exist."""
    _append_csv(entry)
    try:
        # labels help triage in GitHub
        labels = ["user-feedback", f"page:{entry.get('page','unknown')}"]
        labels.append("kind:tweet" if entry.get("tweet_id") else "kind:article")

        title = f"[Feedback] {entry.get('issue','(no type)')} — {entry.get('page','(no page)')}"

        body = (
            f"**Page**: {entry.get('page','')}\n"
            f"**Issue**: {entry.get('issue','')}\n"
            f"**Note**: {entry.get('note','(none)')}\n\n"
            f"**Context**:\n"
            f"- tweet_id/article_id: {entry.get('tweet_id','')}\n"
            f"- source/user: {entry.get('username','')}\n"
            f"- url: {entry.get('tweet_url','')}\n"
            f"- when (UTC): {entry.get('timestamp_utc','')}\n"
        )

        ok, err = github_create_issue(title, body, labels=labels)
        if not ok and st.session_state.get("debug_feedback"):
            st.info(f"Saved to CSV; GitHub mirror failed: {err}")
    except Exception as e:
        if st.session_state.get("debug_feedback"):
            st.info(f"Saved to CSV; GitHub mirror error: {e}")

def feedback_popover(tweet: dict, page: str):
    tid = tweet.get("tweet_id", "")
    turl = f"https://x.com/i/web/status/{tid}"
    pop = getattr(st, "popover", None)
    if callable(pop):
        try:
            ctx = pop("⋯")     # ≤ some Streamlit versions: no `key` arg
        except TypeError:
            ctx = st.expander("⋯", expanded=False)
    else:
        ctx = st.expander("⋯", expanded=False)
    with ctx:
        st.markdown("**Report an issue with this tweet**")
        with st.form(f"fb_form_{tid}", clear_on_submit=True):
            issue = st.selectbox("Type",
                                 ["No image","Broken link","Translation wrong","Bad source","Spam","Other"],
                                 key=f"fb_type_{tid}")
            note = st.text_area("Details (optional)", key=f"fb_note_{tid}", placeholder="Tell us what you saw…")
            if st.form_submit_button("Send"):
                log_feedback({
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
                    "page": page, "tweet_id": tid, "username": tweet.get("username",""),
                    "tweet_url": turl, "issue": issue, "note": (note or "").strip()
                })
                st.success("Thanks! Logged for review.")

@st.cache_data(ttl=300)
def get_competitor_prices(symbols):
    results = []

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.get_info()

            price = info.get("regularMarketPrice")
            prev_close = info.get("regularMarketPreviousClose")

            if price is None or prev_close is None or prev_close == 0:
                continue

            # ✅ Recalculate % change manually
            change_percent = ((price - prev_close) / prev_close) * 100

            results.append({
                "symbol": sym,
                "price": price,
                "change": change_percent
            })
        except Exception as e:
            st.warning(f"Error fetching data for {sym}: {e}")
            continue

    return results

@st.cache_data(ttl=300)
def fetch_btc_market_stats():
    # 1) Try Pro for price/market cap/volume (fast, key-protected)
    try:
        pro = get_coingecko_btc_data()
        if pro:
            circ = total = None
            max_sup = 21_000_000
            last = None

            # 1a) Fill in supply via public endpoint (only for the extra fields)
            try:
                url = ("https://api.coingecko.com/api/v3/coins/bitcoin"
                       "?localization=false&tickers=false&market_data=true"
                       "&community_data=false&developer_data=false&sparkline=false")
                r = requests.get(url, timeout=15, headers={"User-Agent": "CleanSpark Dashboard/1.0"})
                r.raise_for_status()
                data = r.json()
                md = data["market_data"]
                circ = md.get("circulating_supply")
                total = md.get("total_supply")
                max_sup = md.get("max_supply") or 21_000_000
                last = data.get("last_updated")
            except Exception:
                # If this fails, we still return Pro fields + sensible defaults
                pass

            return {
                "price_usd": pro["price"],
                "market_cap_usd": pro["market_cap"],
                "volume_24h_usd": pro["volume"],
                "circulating_supply": circ,
                "total_supply": total,
                "max_supply": max_sup,
                "last_updated": last or datetime.now(ZoneInfo("UTC")).isoformat(),
            }
    except Exception:
        pass  # fall through to public endpoint

    # 2) Fallback: public endpoint only
    url = ("https://api.coingecko.com/api/v3/coins/bitcoin"
           "?localization=false&tickers=false&market_data=true"
           "&community_data=false&developer_data=false&sparkline=false")
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "CleanSpark Dashboard/1.0"})
        r.raise_for_status()
        data = r.json()
        md = data["market_data"]
        return {
            "price_usd": md["current_price"]["usd"],
            "market_cap_usd": md["market_cap"]["usd"],
            "volume_24h_usd": md["total_volume"]["usd"],
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply") or 21_000_000,
            "last_updated": data.get("last_updated"),
        }
    except Exception:
        return {}

@st.cache_data(ttl=600)
def fetch_comp_price_series(ticker, period):
    try:
        if period == "1d":
            interval = "5m"
        else:
            interval = "1d"

        df = history_cached(ticker, period=period, interval=interval)
        df = df[["Close"]].rename(columns={"Close": ticker})
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        return None

@st.cache_data(ttl=900)
def get_latest_sec_fact_with_fallback(cik, tags, year_cutoff=2024, expected_duration=90, tolerance=10, start_date=None, end_date=None):
    if cik is None:
        return None, None, None
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

                    if start_date is not None and end_date is not None:
                        try:
                            end_dt = date_parser.parse(end).replace(tzinfo=None)
                        except Exception:
                            continue
                        if not (start_date <= end_dt <= end_date):
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

@st.cache_data(ttl=3600)
def get_available_filing_quarters(tickers, year=None):
    if year is None:
        year = datetime.now().year

    headers = {"User-Agent": "CleanSpark Dashboard <zgibbs@cleanspark.com>"}

    def infer_quarter_from_date(date_str):
        try:
            dt = parse_date(date_str)
            if dt.year != year:
                return None
            month = dt.month
            if 1 <= month <= 3:
                return "Q1"
            elif 4 <= month <= 6:
                return "Q2"
            elif 7 <= month <= 9:
                return "Q3"
            elif 10 <= month <= 12:
                return "Q4"
        except:
            return None

    found_quarters = set()

    for ticker in tickers:
        cik = cik_map.get(ticker)
        if not cik:
            continue
        cik_clean = str(int(cik)).zfill(10)

        # Use a reliable financial tag (like revenue) to extract report periods
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik_clean}/us-gaap/Revenues.json"
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                continue
            data = resp.json()
            units = data.get("units", {})
            for unit_values in units.values():
                for v in unit_values:
                    end = v.get("end")
                    if not end:
                        continue
                    q = infer_quarter_from_date(end)
                    if q:
                        found_quarters.add(q)
        except Exception as e:
            print(f"[SEC QUARTERS] Error for {ticker}: {e}")
            continue

    return sorted(list(found_quarters))

@st.cache_data(ttl=3600)
def get_quarter_date_bounds(quarter, year=None):
    """Returns the start and end datetime for a given quarter and year."""
    if year is None:
        year = datetime.now().year

    if quarter == "Q1":
        return datetime(year, 1, 1), datetime(year, 3, 31, 23, 59, 59)
    elif quarter == "Q2":
        return datetime(year, 4, 1), datetime(year, 6, 30, 23, 59, 59)
    elif quarter == "Q3":
        return datetime(year, 7, 1), datetime(year, 9, 30, 23, 59, 59)
    elif quarter == "Q4":
        return datetime(year, 10, 1), datetime(year, 12, 31, 23, 59, 59)
    else:
        raise ValueError(f"Unknown quarter: {quarter}")

@st.cache_data(ttl=86400)
def get_latest_edgar_inline_url(cik, accn=None):
    cik_clean = str(int(cik)).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_clean}.json"
    headers = {"User-Agent": "CleanSpark Dashboard <zgibbs@cleanspark.com>"}
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return None
        data = resp.json()
        recent = data["filings"]["recent"]
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs       = recent.get("primaryDocument", [])
        forms              = recent.get("form", [])

        def normalize(a): return (a or "").replace("-", "").lower()

        if accn:
            # Try to find the exact recent filing that matches this accession number
            target = normalize(accn)
            for a, doc, form in zip(accession_numbers, primary_docs, forms):
                if normalize(a) == target and form in ("10-K", "10-Q") and doc.endswith(".htm"):
                    accn_nodash = a.replace("-", "")
                    return f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{int(cik)}/{accn_nodash}/{doc}"
            # If not found in recent, fall through to latest below

        # Fallback: latest 10-K or 10-Q (previous behavior)
        for a, doc, form in zip(accession_numbers, primary_docs, forms):
            if form in ("10-K", "10-Q") and doc.endswith(".htm"):
                accn_nodash = a.replace("-", "")
                return f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{int(cik)}/{accn_nodash}/{doc}"
    except Exception as e:
        print("EDGAR inline link fetch failed:", e)
    return None

@st.cache_data(ttl=3600)
def get_latest_press_release_metrics(company_name, ticker_symbol):
    import re
    import requests
    from bs4 import BeautifulSoup

    ticker = ticker_symbol.upper()
    query = f'"{company_name}" earnings OR update OR quarter OR "quarterly report"'

    articles = get_news(
        query,
        page_size=10,
        sort_by="publishedAt",
        from_days=29,
        domains=["globenewswire.com", "nasdaq.com", "seekingalpha.com", "markets.businessinsider.com"]
    )

    if not articles:
        print("❌ No articles returned by NewsAPI.")
        return None

    article = next(
    (
        a for a in articles
        if (
            (ticker in (a.get("title", "") + a.get("description", "")).upper()
             or company_name.upper() in (a.get("title", "") + a.get("description", "")).upper())
            and any(
                kw in (a.get("title", "") + a.get("description", "")).lower()
                for kw in ["earnings", "update", "quarter", "quarterly report"]
            )
        )
    ),
    None
)

    if not article:
        print("❌ No matching article found in NewsAPI results.")
        return None

    title = article.get("title", "")
    url = article.get("url", "")
    published_date = article.get("publishedAt", "")[:10]

    # Fetch and parse full HTML page
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        tables = soup.find_all("table")

        table_text = []
        for table in tables:
            for row in table.find_all("tr"):
                cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                table_text.append(" ".join(cells))

        full_text = " ".join(p.get_text() for p in paragraphs) + " " + " ".join(table_text)

    except Exception as e:
        print(f"⚠️ Error loading article HTML: {e}")
        full_text = f"{title} {article.get('description', '')}"

    content = full_text.lower()
    metrics = {}

    # Patterns
    patterns = {
        "Bitcoin Held": r"(?:held(?: a total of)?|holdings(?: total)?)[^\d]{0,15}([\d,]+)\s*(?:btc|bitcoin)",
        "Revenue": r"revenue(?: (?:of|was))?[\s:]*\$([\d,\.]+)",
        "Cash & Equivalents": r"(?:cash and cash equivalents|cash)(?: (?:totaled|of))?[\s:]*\$([\d,\.]+)",
        "Earnings Per Share (Basic)": r"(?:earnings per share|eps)(?: \(basic\))?[\s:]*\$([\d\.]+)",
        "Earnings Per Share (Diluted)": r"(?:earnings per share|eps)(?: \(diluted\))?[\s:]*\$([\d\.]+)",
        "EH/s": r"([\d\.]+)\s*(?:EH/s|exahash|exahashes)"
    }

    # Step 1: Extract EH/s first and redact the entire EH/s string from content
    ehs_matches = re.findall(r"(([\d\.]+)\s*(EH/s|exahash|exahashes))", content)
    if ehs_matches:
        values = []
        for full, number, unit in ehs_matches:
            try:
                val = float(number.replace(",", ""))
                values.append(val)
                content = content.replace(full, "")  # redact full match like "50 EH/s"
            except:
                continue
        if values:
            metrics["EH/s"] = f"{max(values):.2f} EH/s"

    # Step 2: Extract everything else
    for label, pattern in patterns.items():
        if label == "EH/s":
            continue  # Already handled

        matches = re.findall(pattern, content)
        if not matches:
            continue

        # Flatten based on type
        if isinstance(matches[0], tuple):
            numbers = [g for pair in matches for g in pair if g and g.strip().replace(',', '').replace('.', '').isdigit()]
        else:
            numbers = [g for g in matches if g and g.strip().replace(',', '').replace('.', '').isdigit()]

        if not numbers:
            continue

        try:
            if "Bitcoin" in label:
                val = max(int(n.replace(",", "")) for n in numbers)
                metrics[label] = f"{val:,} BTC"
            else:
                val = max(float(n.replace(",", "")) for n in numbers)
                metrics[label] = f"${val:,.2f}"
        except:
            continue

    return {
        "title": title,
        "url": url,
        "date": published_date,
        "metrics": metrics
    }

# --- Tabs and Layout Config ---
st.title("Legal & Market Dashboard")
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)  # spacing

    def image_to_base64(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    img_b64 = image_to_base64("cleanspark_logo.png")

    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="https://www.cleanspark.com" target="_blank" title="Visit CleanSpark">
                <img src="data:image/png;base64,{img_b64}" style="width:100%; max-width:400px; border-radius:10px; display:block; margin:auto;" />
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
tab = st.sidebar.selectbox("Select a page", ["Bitcoin News", "Live Market"])
if tab == "Bitcoin News":
    
    day_options = {
        "1 Day": 1,
        "3 Days": 3,
        "5 Days": 5,
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
    btc_range_options = range_options  # show 1 Day / 1 Week / 1 Month / 6 Months / 1 Year
    sel = st.pills("Bitcoin price range:", options=list(btc_range_options.keys()), default="1 Day", key="btc_range")
    selected_range = btc_range_options.get(sel, "1mo")
    with st.spinner("Loading BTC price…"):
        data = get_history(btc, selected_range)

    if not data.empty:
        stats = fetch_btc_market_stats()

        if not stats:
            st.info("Showing price chart only; market stats will return automatically when available.")
        else:
            # Format helpers
            def fmt_btc(n): 
                return f"{n:,.0f} BTC" if n is not None else "—"

            def fmt_usd(n):
                return f"${n:,.0f}" if n is not None else "—"

            def fmt_time_et(iso):
                try:
                    ts = pd.to_datetime(iso, utc=True).tz_convert(ET)
                    return ts.strftime("%Y-%m-%d %I:%M %p ET")
                except Exception:
                    return "—"

            circulating = stats["circulating_supply"] or 0
            max_supply  = stats["max_supply"] or 21_000_000
            pct_mined   = (circulating / max_supply * 100) if max_supply else None

            st.subheader("Market Stats")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price (USD)", f"{stats['price_usd']:,.2f}")
            c2.metric("Circulating Supply", fmt_btc(circulating))
            c3.metric("Max Supply", fmt_btc(max_supply))
            c4.metric("Market Cap", fmt_usd(stats["market_cap_usd"]))

        btc_close = data["Close"].dropna().round(2).rename("Bitcoin Price").reset_index()
        btc_close.columns = ["Date", "Price"]
        btc_close["Price"] = btc_close["Price"].round(2)

        # Convert to PT
        btc_close["Date"] = pd.to_datetime(btc_close["Date"])
        if btc_close["Date"].dt.tz is None:
            btc_close["Date"] = btc_close["Date"].dt.tz_localize("UTC")
        
        btc_close["Date"] = btc_close["Date"].dt.tz_convert("US/Pacific")
        btc_close["Label"] = btc_close["Date"].dt.strftime("%I:%M %p")
        if selected_range != "1d":
            btc_close["Date_Day"] = btc_close["Date"].dt.normalize()
        if selected_range == "1y":
            # make sure rows are in time order
            btc_close = btc_close.sort_values("Date_Day").reset_index(drop=True)

            # keep every other row, but always include the last point
            keep_idx = list(range(0, len(btc_close), 2))
            if keep_idx and keep_idx[-1] != len(btc_close) - 1:
                keep_idx.append(len(btc_close) - 1)

            btc_close = btc_close.iloc[keep_idx].copy()
        if selected_range == "1d":
            btc_close = (
                btc_close
                .set_index("Date")
                .resample("30min", label="right", closed="right")
                .ffill()              # keep the latest known price in each 30m bucket
                .reset_index()
            )
            # Fallback: if resample produced <2 points, use raw 5m data
            if len(btc_close) < 2:
                raw = data["Close"].dropna().round(2).rename("Bitcoin Price").reset_index()
                raw.columns = ["Date", "Price"]
                raw["Date"] = pd.to_datetime(raw["Date"])
                if raw["Date"].dt.tz is None:
                    raw["Date"] = raw["Date"].dt.tz_localize("UTC")
                raw["Date"] = raw["Date"].dt.tz_convert("US/Pacific")
                btc_close = raw

        # Optional trim for 1 Week
        if sel == "1 Week":
            # Keep full 1-week range; do not arbitrarily tail rows
            pass

        # Y-axis bounds
        btc_low = btc_close["Price"].min()
        btc_high = btc_close["Price"].max()

        if selected_range == "1d":
            min_y = btc_low * .996
            max_y = btc_high * 1.003
        elif selected_range == "7d":
            min_y = btc_low * .997
            max_y = btc_high * 1.003
        elif selected_range == "1mo":
            min_y = btc_low * 0.988
            max_y = btc_high * 1.01
        else:
            min_y = btc_low * 0.9
            max_y = btc_high * 1.05

        # Label for tooltip + x-axis format
        if selected_range == "1d":
            btc_close["Label"] = btc_close["Date"].dt.strftime("%-I:%M %p")
            x_axis = alt.X(
                "Date:T",
                title="Time (PT)",
                axis=alt.Axis(labelAngle=45, format="%I:%M %p")
            )
            tooltip_title = "Time (PT)"
        else:
            x_axis = alt.X(
                "Date_Day:T",
                title="Date",
                scale=alt.Scale(nice="day"),
                axis=alt.Axis(
                    labelAngle=45,
                    format="%b %d",
                    tickCount={"interval": "day", "step": 1},  # one tick per day
                ),
            )
            tooltip_title = "Date"

        # Build chart with points
        line = alt.Chart(btc_close).mark_line().encode(
            x=x_axis,
            y=alt.Y("Price:Q", scale=alt.Scale(domain=[min_y, max_y]))
        )

        points = alt.Chart(btc_close).mark_circle(size=40).encode(
            x=alt.X("Date:T") if selected_range == "1d" else alt.X("Date_Day:T"),
            y="Price:Q",
            tooltip=[
                alt.Tooltip(
                    "Date:T" if selected_range == "1d" else "Date_Day:T",
                    title=tooltip_title,
                    format="%I:%M %p" if selected_range == "1d" else "%b %d",
                ),
                alt.Tooltip("Price:Q", format=".2f"),
            ],
        )

        chart = (line + points).properties(
            width="container",
            height=400,
            title="Bitcoin Price"
        )

        st.altair_chart(chart, use_container_width=True)

    col1, col2 = st.columns([1.8,2.2])

    with col1:
        header_slot = st.container()

        tw_scope = st.pills("Tweet Scope:", ["All Bitcoin", "CleanSpark Only"], default="CleanSpark Only", key="tw_scope")
        tw_sort = st.pills("Sort tweets by:", ["Likes", "Retweets"], default="Likes", key="tw_sort")
        tw_scope_val = "CleanSpark" if tw_scope == "CleanSpark Only" else "General"
        tw_max_days = 1  # fixed 24-hour window

        tweets = get_cleanspark_tweets(
            query_scope=tw_scope_val,
            max_age_days=tw_max_days,        
            sort_by=tw_sort.lower(),         
            max_results=15
        )

        with header_slot:
            header, pill = st.columns([1, 0.18])
            with header:
                st.subheader("🐦 Twitter Feed (last 24 hours)")
            with pill:
                render_section_report_pill(
                    section_key="tweets",
                    items=tweets,            
                    kind="tweet",
                    page="Bitcoin News"
                )

        for tweet in tweets:
            translated_text = translate_text(tweet["text"], GOOGLE_API_KEY)
            clean_text = re.sub(r'https://t\.co/\S+$', '', translated_text).strip()

            def custom_escape(text):
                text = text.replace("&", "<<<AMP>>>")  
                text = html.escape(text)
                return text.replace("<<<AMP>>>", "&")

            final_text = html.escape(clean_text).replace("\n", "<br>")

            with st.container():
                st.markdown(f"""
                <div class="tweet-block" style="display:flex; align-items:flex-start; gap:12px; margin-bottom:1rem;">
                    <img src="{tweet['profile_img']}" style="width:48px; height:48px; border-radius:50%; flex:0 0 auto;">
                    <div style="flex:1 1 auto;">
                        <div style="font-weight:600;">{tweet['name']}</div>
                        <div style="color:gray; font-size:13px;">@{tweet['username']} • {format_timestamp(tweet['created_at'])}</div>
                        <div style="margin-top:6px; font-size:15px; line-height:1.5; overflow-wrap:anywhere; word-break:break-word; hyphens:auto;">{final_text}</div>
                        <div style="color:gray; font-size:13px; margin-top:6px;">🔁 {tweet['retweets']} &nbsp;&nbsp;&nbsp; ❤️ {tweet['likes']}</div>
                        <div style="margin-top:6px;"><a href="https://x.com/i/web/status/{tweet['tweet_id']}" target="_blank" style="color:#1DA1F2; font-size:13px;">View on Twitter</a></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                render_tweet_media(tweet["media"])
                st.markdown("<hr style='margin: 1rem 0; border: 2px solid #ddd;'>", unsafe_allow_html=True)

                    
    # General News
    with col2:
        header_slot = st.container()
        scope_options = ["All Bitcoin", "CleanSpark Only", "Regulatory Only"]
        gen_scope = st.pills("Article Scope:", scope_options, default="All Bitcoin", key="news_scope_filter")

        gen_col1, gen_col2 = st.columns([1, 1])
        with gen_col1:
            gen_days = st.pills("Articles from the past...", list(day_options.keys()), default="1 Day", key="news_days_filter")
        with gen_col2:
            gen_sort = st.pills("Sort by:", list(sort_by_map.keys()), default="Popularity", key="news_sort_filter")

        if gen_scope == "CleanSpark Only":
            query_term = "CleanSpark"
            exclude_term = None
            filter_func = None
        elif gen_scope == "Regulatory Only":
            query_term = "GENIUS Act OR cryptocurrency legislation OR crypto bill OR digital asset policy"
            exclude_term = None
            filter_func = regulatory_article_filter
        else:  
            query_term = "bitcoin mining"
            exclude_term = "CleanSpark"
            filter_func = None

        gen_from_days = day_options[gen_days]
        gen_sort_by   = sort_by_map[gen_sort]

        load_articles(
            key="gen_articles",
            query=query_term,
            exclude=exclude_term,
            from_days=gen_from_days,
            sort_by=gen_sort_by,
            filter_func=filter_func,
            pill_at="title"
        )

        with header_slot:
            header, pill = st.columns([1, 0.18])
            with header:
                st.subheader("📰 Bitcoin News")
            with pill:
                render_section_report_pill(
                    section_key="news",
                    items=st.session_state.get("gen_articles", []),
                    kind="article",
                    page="Bitcoin News"
                )
                    
# --- HOME TAB ---
if tab == "Live Market":
    btc_metrics = get_coingecko_btc_data()
    btc         = yf.Ticker("BTC-USD")
    raw_sym = st.session_state.get("stock_lookup_ticker")
    sym = (raw_sym or "CLSK").strip().upper()  # <-- never empty

    try:
        ticker_obj = yf.Ticker(sym)
        info = ticker_obj.get_info()
        company_name = info.get("longName", sym)
    except Exception:
        company_name = sym
        info = {}
        ticker_obj = None
    # Stock
    with st.container():
        st.subheader(f"📊 Stock Market Lookup: {company_name}")        
        m1, m2 = st.columns([1.5, 2.5])
        with m1:
            raw = st.text_input(
                "Stock ticker:", 
                "CLSK", 
                key="stock_lookup_ticker",
                placeholder="e.g., CLSK"
            )
            sym = (raw or "").strip().upper()
            cik = cik_map.get(sym) if sym else None
            entered_ticker = sym or ""

        # Prevent crash on empty input: hint + safe fallback
        if not sym:
            st.info("Type a ticker to search (e.g., CLSK). Defaulting to CLSK.")
            sym = "CLSK"
        
        with m2:
            market_range_options = range_options  # include 1 Week
            lookup_range = st.pills("Timeframe:", options=list(market_range_options.keys()), default="1 Day", key="lookup_range")
            selected_range = market_range_options.get(lookup_range, "1mo")
            extended_range = selected_range

        # Fetch data for selected ticker
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

        # ROW 2: Show current price on its own line
        price = info.get("regularMarketPrice")
        change_amount = info.get("regularMarketChange")
        prev_close = info.get("regularMarketPreviousClose")
        change_percent = (((price - prev_close) / prev_close) * 100
            if (price is not None and prev_close not in (None, 0))
            else None
        )        
        clsk_price = clsk_open = clsk_high = clsk_low = None
        
        if sym != "CLSK":
            clsk = yf.Ticker("CLSK")
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
            st.markdown(render_metric_block("Current Price", price, delta=change_amount, reference=prev_close, show_arrow=True), unsafe_allow_html=True)

        with m2:
            st.markdown(render_metric_block("CLSK Price", clsk_price), unsafe_allow_html=True)

        with m3:
            st.markdown(render_metric_block("CLSK Open", clsk_open), unsafe_allow_html=True)

        with m4:
            st.markdown(render_metric_block("CLSK High", clsk_high), unsafe_allow_html=True)

        with m5:
            st.markdown(render_metric_block("CLSK Low", clsk_low), unsafe_allow_html=True)
        
        interval = "5m" if selected_range == "1d" else "1d"
        with st.spinner(f"Loading {sym} price…"):
            df = history_cached(sym, period=extended_range, interval=interval)
        
        if lookup_range == "1 Day":
            pass
            
        # Limit to 5 most recent valid market days (skip holidays/weekends)
        if lookup_range == "5 Days":
            df = df.dropna(subset=["Close"])
            df["DateOnly"] = df.index.date
            recent_days = sorted(df["DateOnly"].unique())[-5:]  # Get last 5 trading days
            df = df[df["DateOnly"].isin(recent_days)]
        
        if "regularMarketPrice" not in info or df.empty:
            st.warning(f"No data available for ticker `{sym}`.")
        else:
            # Format large numbers
            def fmt_float(val): return f"{val:.2f}" if val is not None else "N/A"
            def fmt_millions(val): return f"{val/1_000_000:.2f} M" if val is not None else "N/A"
            def fmt_billions(val): return f"{val/1_000_000_000:.2f} B" if val is not None else "N/A"
            st.markdown("<hr>", unsafe_allow_html=True)
            try:
                interval = "5m" if selected_range == "1d" else "1d"

                if not df.empty and "Close" in df.columns:
                    stock_close = df["Close"].round(2).rename("Price").reset_index()
                    stock_close.columns = ["Date", "Price"]
                    
                    # Timezone handling and tooltip formatting
                    stock_close["Date"] = pd.to_datetime(stock_close["Date"])

                    if selected_range == "1d":
                        # intraday: ET-aware
                        if stock_close["Date"].dt.tz is None:
                            stock_close["Date"] = stock_close["Date"].dt.tz_localize("UTC")
                        stock_close["Date"] = stock_close["Date"].dt.tz_convert("US/Eastern")
                        stock_close["Label"] = stock_close["Date"].dt.strftime("%I:%M %p")
                        order_col = "Date"
                    else:
                        # daily ranges: align the CALENDAR DAY to ET, then drop tz so Vega-Lite won’t shift
                        if stock_close["Date"].dt.tz is None:
                            stock_close["Date"] = stock_close["Date"].dt.tz_localize("UTC")
                        stock_close["Date_Day"] = (
                            stock_close["Date"]
                            .dt.tz_convert("US/Eastern")   # move to ET first
                            .dt.normalize()                 # midnight ET
                            .dt.tz_localize(None)           # strip tz for stable Altair rendering
                        )
                        order_col = "Date_Day"

                    # sort by the same column you plot
                    stock_close.sort_values(order_col, inplace=True)

                    if selected_range == "1y":
                        stock_close = stock_close.sort_values("Date" if "Date_Day" not in stock_close else "Date_Day").reset_index(drop=True)
                        keep = list(range(0, len(stock_close), 2))
                        if keep and keep[-1] != len(stock_close) - 1:
                            keep.append(len(stock_close) - 1)
                        stock_close = stock_close.iloc[keep].copy()
                    
                    
                    # Calculate bounds
                    stock_low = df["Low"].min()
                    stock_high = df["High"].max()
                    
                    if selected_range in ["1d", "7d"]:
                        min_y = stock_low * 0.99
                        max_y = stock_high * 1.01
                    elif selected_range == "1mo":
                        min_y = stock_low * 0.985
                        max_y = stock_high * 1.05
                    else:
                        min_y = stock_low * 0.88
                        max_y = stock_high * 1.05
                    
                    if selected_range == "1d":
                        x_axis = alt.X(
                            "Date:T",
                            title="Time (PT)",
                            axis=alt.Axis(labelAngle=45, format="%I:%M %p")
                        )
                    else:
                        x_axis = alt.X(
                            "Date_Day:T",
                            title="Date",
                            scale=alt.Scale(nice="day"),
                            axis=alt.Axis(
                                format="%b %d",
                                tickCount={"interval": "day", "step": 1},
                            ),
                        )
                        
                    # Build chart
                    stock_chart = alt.layer(
                        alt.Chart(stock_close).mark_line().encode(
                            x=x_axis,
                            y=alt.Y("Price:Q", scale=alt.Scale(domain=[min_y, max_y])),
                            tooltip=[
                                alt.Tooltip("Date:T" if selected_range == "1d" else "Date_Day:T",
                                            title="Time (PT)" if selected_range == "1d" else "Date",
                                            format="%I:%M %p" if selected_range == "1d" else "%b %d"),
                                alt.Tooltip("Price:Q", format=",.2f"),
                            ]
                        ),
                        alt.Chart(stock_close).mark_circle(size=40).encode(
                            x=x_axis,  # <— important
                            y="Price:Q",
                            tooltip=[
                                alt.Tooltip("Date:T" if selected_range == "1d" else "Date_Day:T",
                                            title="Time (PT)" if selected_range == "1d" else "Date",
                                            format="%I:%M %p" if selected_range == "1d" else "%b %d"),
                                alt.Tooltip("Price:Q", format=",.2f"),
                            ],
                        )
                    ).properties(width="container", height=400, title=f"{sym} Price")

                    points = alt.Chart(stock_close).mark_circle(size=40).encode(
                        x=x_axis,                      # <— use the same x spec as the line
                        y="Price:Q",
                        tooltip=[
                            alt.Tooltip("Date:T" if selected_range == "1d" else "Date_Day:T",
                                        title="Time (PT)" if selected_range == "1d" else "Date",
                                        format="%I:%M %p" if selected_range == "1d" else "%b %d"),
                            alt.Tooltip("Price:Q", format=",.2f"),
                        ],
                    )

                    st.altair_chart(stock_chart, use_container_width=True)
                    
                else:
                    st.warning("No price data available for this range.")
            except Exception as e:
                st.error(f"Error loading chart data: {e}")
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
        st.markdown("<hr>", unsafe_allow_html=True)


    st.subheader("Live Competition View")
    competitors = get_competitor_prices(competitor_tickers)

    m1, m2, m3, m4, m5,m6 ,m7, m8, m9, m10, m11, m12, m13= st.columns(13)
    col_map = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11,m12, m13]

    for i, comp in enumerate(competitors):
        if i >= len(col_map):
            break
        with col_map[i]:
            if comp["symbol"] == sym:
                comp_price = price
                comp_change = change_percent
            else:
                comp_price = comp["price"]
                comp_change = comp["change"]

            arrow = "🔺" if comp_change >= 0 else "🔻"
            color = "green" if comp_change >= 0 else "red"
            st.markdown(
                f"""
                <div style='font-size:16px; text-align: center; padding: 8px 0;'>
                    <strong>{comp['symbol']}</strong><br>
                    ${comp_price:.2f}<br>
                    <span style='color:{color};'>{arrow} {comp_change:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    # Live Competition View Chart
    # New range selector just for this chart
    comp_range = st.pills(
        "Select Time Range:",
        options=list(range_options.keys()),
        default="1 Day",
        key="comp_chart_range"
    )
    comp_selected_period = range_options.get(comp_range, "1mo")

    # Ticker selector
    comp_selected_tickers = st.multiselect(
        "Select companies to compare:",
        options=competitor_tickers,
        default=["CLSK", "MARA"],
        key="comp_chart_tickers"
    )

    combined_df = None
    for ticker in comp_selected_tickers:
        df = fetch_comp_price_series(ticker, comp_selected_period)
        if df is not None and not df.empty:
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how="outer")
            
    # Build chart (OUTSIDE the for-loop)
    if combined_df is not None and not combined_df.empty:
        chart_df = combined_df.reset_index()
        chart_df.rename(columns={chart_df.columns[0]: "Date"}, inplace=True)

        # Convert to Eastern Time (tz-aware)
        chart_df["Date"] = pd.to_datetime(chart_df["Date"])
        if chart_df["Date"].dt.tz is None:
            chart_df["Date"] = chart_df["Date"].dt.tz_localize("UTC")
        chart_df["Date"] = chart_df["Date"].dt.tz_convert("US/Eastern")

        if comp_selected_period != "1d":
            # daily ranges: keep the *calendar day in ET*, then drop tz
            chart_df["Date_Day"] = (
                chart_df["Date"]
                .dt.normalize()         # midnight ET
                .dt.tz_localize(None)   # make it tz-naive so Altair won't shift to UTC
            )

        # Sort by the same column we’ll plot on X
        order_col = "Date" if comp_selected_period == "1d" else "Date_Day"
        chart_df.sort_values(order_col, inplace=True)

        # Keep Date_Day when present so axis/points align
        id_vars = ["Date"] + (["Date_Day"] if comp_selected_period != "1d" else [])
        chart_df = chart_df.melt(id_vars=id_vars, var_name="Ticker", value_name="Price")

        # NOW it's long-form, so this is valid:
        chart_df.sort_values(["Ticker", order_col], inplace=True)

        # Thin ONLY the 1-year view (~50% fewer points), per ticker, keeping each ticker's last point
        if comp_selected_period == "1y":
            order_col = "Date_Day" if comp_selected_period != "1d" else "Date"
            chart_df = (chart_df
                .sort_values(["Ticker", order_col])
                .assign(_n=lambda d: d.groupby("Ticker").cumcount())
            )
            last_n = chart_df.groupby("Ticker")["_n"].transform("max")
            chart_df = chart_df[(chart_df["_n"] % 2 == 0) | (chart_df["_n"] == last_n)].drop(columns="_n")


        # Format time for tooltip (PT)
        if comp_selected_period == "1d":
            chart_df["TimeET"] = chart_df["Date"].dt.strftime("%I:%M %p")
        else:
            chart_df["TimeET"] = chart_df["Date"].dt.strftime("%b %d")

        chart_df["Price"] = pd.to_numeric(chart_df["Price"], errors="coerce")
        chart_df.dropna(subset=["Price"], inplace=True)
        chart_df["Price"] = chart_df["Price"].round(2)

        # Y-axis bounds (keep your math)
        min_y = chart_df["Price"].min() * 0.99
        max_y = chart_df["Price"].max() * 1.01
        y_scale = alt.Scale(domain=[min_y, max_y]) if not math.isnan(min_y) else alt.Scale()

        label_angle = 45 if comp_selected_period == "1d" else 0

        tooltip_title = "Time (PT)" if comp_selected_period == "1d" else "Date"

        points = alt.Chart(chart_df).mark_circle(size=40).encode(
            x=alt.X("Date:T") if comp_selected_period == "1d" else alt.X("Date_Day:T"),
            y="Price:Q",
            color="Ticker:N",
            tooltip=[
                alt.Tooltip(
                    "Date:T" if comp_selected_period == "1d" else "Date_Day:T",
                    title="Time (PT)" if comp_selected_period == "1d" else "Date",
                    format="%I:%M %p" if comp_selected_period == "1d" else "%b %d",
                ),
                alt.Tooltip("Price:Q", format=",.2f"),
                alt.Tooltip("Ticker:N"),
            ],
        )
        
        if comp_selected_period == "1d":
            x_axis_comp = alt.X(
                "Date:T",
                title="Time (PT)",
                axis=alt.Axis(labelAngle=45, format="%I:%M %p"),
                # scale=None  # ← delete this line
            )
        else:
            x_axis_comp = alt.X(
                "Date_Day:T",
                title="Date",
                scale=alt.Scale(nice="day"),
                axis=alt.Axis(
                    labelAngle=0,
                    format="%b %d",
                    tickCount={"interval": "day", "step": 1},
                ),
            )

        line = alt.Chart(chart_df).mark_line().encode(
            x=x_axis_comp,
            y=alt.Y("Price:Q", scale=y_scale),
            color="Ticker:N",
            tooltip=[
                alt.Tooltip(
                    "Date:T" if comp_selected_period == "1d" else "Date_Day:T",
                    title="Time (PT)" if comp_selected_period == "1d" else "Date",
                    format="%I:%M %p" if comp_selected_period == "1d" else "%b %d",
                ),
                alt.Tooltip("Price:Q", format=",.2f"),
                alt.Tooltip("Ticker:N"),
            ],
        )

        st.altair_chart((line + points).properties(
            width="container",
            height=400,
            title="Stock Price Comparison"
        ), use_container_width=True)
    else:
        st.info("No data available for selected tickers/time range.")

    # --- Combined Filing & Press Metrics ---
    st.markdown("### 📋 Competitor Financial Metrics Table")
    st.write("Metrics pulled from SEC filings or recent press releases.")
    
    # Dynamically determine available filing quarters
    available_quarters = get_available_filing_quarters(competitor_tickers)

    # Build dynamic options list
    pill_options = ["Current Metrics"] + available_quarters
    quarter = st.pills("Select View:", pill_options, default=pill_options[0], key="quarter_selector")

    with st.spinner("Loading competitor metrics…"):
        is_current_metrics = (quarter == "Current Metrics")

        if is_current_metrics:
            start_date = end_date = None  # no filtering, show latest
        else:
            start_date, end_date = get_quarter_date_bounds(quarter)

        st.caption(f"Data shown for: **{quarter} {datetime.now().year}** {'(live)' if is_current_metrics else '(locked)'}")

        df_rows = []
        for ticker in competitor_tickers:
            cik = cik_map.get(ticker)
            name = yf.Ticker(ticker).info.get("shortName", ticker)
            row = {"Ticker": ticker, "Name": name}
            sources_used = []
            sec_data = {}

            # quarter bounds (per your selection)
            if is_current_metrics:
                start_date = end_date = None
            else:
                start_date, end_date = get_quarter_date_bounds(quarter)

            # 1) PRESS (only for Current Metrics)
            if is_current_metrics:
                press = get_latest_press_release_metrics(name, ticker)
                if press and press.get("url"):
                    press_date = press.get("date")
                    press_url = press.get("url")
                    for label, val_str in press["metrics"].items():
                        try:
                            # normalize to float first; ignore EH/s strings that don’t parse
                            val_clean = float(
                                val_str.replace("$", "").replace(",", "").replace(" BTC", "").strip()
                            )
                            sec_data[label] = val_clean
                        except Exception:
                            pass
                    sources_used.append(("Press", press_date, press_url))

            # 2) SEC fallback (single most-recent filing)
            latest_accn = None
            latest_date = None
            latest_values = {}

            for label, tags in SEC_FACTS.items():
                if label in sec_data or not cik:
                    continue
                val, date, accn = get_latest_sec_fact_with_fallback(
                    cik, tags, start_date=start_date, end_date=end_date
                )
                if val is not None and accn is not None:
                    if not latest_accn or date > latest_date:
                        latest_accn = accn
                        latest_date = date
                    latest_values[label] = (val, accn)

            if latest_accn:
                sec_url = get_latest_edgar_inline_url(cik, accn=latest_accn)
                for label, (val, accn) in latest_values.items():
                    if accn == latest_accn:
                        sec_data[label] = val
                sources_used.append(("SEC", latest_date, sec_url))

            # 3) finalize row
            for label in SEC_FACTS.keys():
                row[label] = sec_data.get(label, None)

            if sources_used:
                link_list = []
                for source, date, url in sorted(sources_used, key=lambda x: x[1], reverse=True):
                    if isinstance(url, str) and url.strip():
                        safe_url = quote(url, safe=':/?=&')
                        link_list.append(f'<a href="{safe_url}" target="_blank">{date} ({source})</a>')
                row["Last Report"] = " • ".join(link_list) if link_list else "-"
            else:
                row["Last Report"] = "-"

            df_rows.append(row)

    df = pd.DataFrame(df_rows)
    df.set_index("Ticker", inplace=True)

    for label in SEC_FACTS:
        if label in df.columns:
            s = pd.to_numeric(df[label], errors="coerce")
            if "Bitcoin" in label:
                df[label] = s.apply(lambda v: f"{v:,.0f} BTC" if pd.notna(v) else "–")
            elif "EH/s" in label:
                df[label] = s.apply(lambda v: f"{v:,.2f} EH/s" if pd.notna(v) else "–")
            else:
                df[label] = s.apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "–")
                    
    # Define formatting
    def get_formatter(col):
        if "Bitcoin" in col:
            return "{:,.0f} BTC"
        elif "EH/s" in col:
            return "{:,.2f} EH/s"
        else:
            return "${:,.2f}"

    formatters = {col: get_formatter(col) for col in SEC_FACTS.keys() if df[col].dtype in ['float64', 'int64']}

    # Display table with custom formats and clickable date link
    st.dataframe(df.drop(columns=["Last Report"]), use_container_width=True)
    st.markdown("#### 🔗 Filing Report Links")
    for idx, row in df.iterrows():
        st.markdown(f"- **{idx}** ({row['Name']}): {row['Last Report']}", unsafe_allow_html=True)
