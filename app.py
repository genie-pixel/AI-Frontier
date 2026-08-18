from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp
import requests
import streamlit as st
from bs4 import BeautifulSoup
from ddgs import DDGS
from google import genai

MODEL_NAME = "gemini-3.5-flash"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 12_000
MAX_SCRAPED_CHARS = 1_200
HTTP_TIMEOUT_SECONDS = 6
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md", "csv"}
TEXT_EXTENSIONS = {"txt", "md", "csv"}
AUDIENCE_OPTIONS = ["CEO / 최고경영진", "본부장 / 임원", "팀장 / 실무진", "외부 고객"]
FEATURE_OPTIONS = [
    "결론 및 핵심 성과 우선 (실용주의)", "수치/데이터 중심의 논리 검증 선호",
    "리스크/한계점의 투명한 공유 선호", "감성적 공감 및 스토리 중심 선호",
]
PATH_HELP = {
    "purpose": ("01 목적 해석", "작성한 사항은 목적 사분면 선정, 의사결정 질문 정의, 필수 포함 관점에 반영됩니다."),
    "research": ("02 정보 탐색", "작성한 사항은 검색 범위, 근거의 우선순위, 제외할 정보 기준에 반영됩니다."),
    "flow": ("03 흐름 설계", "작성한 사항은 보고 구성 요소의 선택과 정렬, 강조 메시지와 리스크 제시 순서에 반영됩니다."),
}
QUADRANTS = {
    "Trouble Shooting": "재발 방지, 사태 수습, 대안 강구",
    "Navigating": "방향 설정, 비전 제시, 기회 선점",
    "Sensing": "동향 파악, 변화 감지, 징후 포착",
    "Investigating": "원인 규명, 진상 조사, 실태 점검",
}

@dataclass(frozen=True)
class Source:
    source_type: str
    title: str
    url: str
    date: str
    body: str = ""


def public_http_url(url: str) -> bool:
    """SSRF 방지: http(s)·호스트·공개 IP로 해석되는 URL만 허용합니다."""
    try:
        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return False
        host = parsed.hostname.rstrip(".")
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(item[4][0]).is_global for item in addresses)
    except (ValueError, socket.gaierror, OSError):
        return False


def clean_text(value: str, limit: int = MAX_SCRAPED_CHARS) -> str:
    return " ".join(BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True).split())[:limit]


def validate_file(uploaded: Any) -> tuple[str, str | None]:
    if uploaded is None:
        return "", None
    suffix = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
    if suffix not in ALLOWED_EXTENSIONS:
        return "", "허용되지 않는 파일 형식입니다. PDF, DOCX, TXT, MD, CSV만 첨부할 수 있습니다."
    if uploaded.size > MAX_FILE_BYTES:
        return "", "첨부 파일은 최대 10MB까지 업로드할 수 있습니다."
    if suffix not in TEXT_EXTENSIONS:
        return "", "PDF/DOCX는 이 버전에서 자동 추출하지 않습니다. 핵심 내용을 텍스트 입력란에 붙여 넣어주세요."
    try:
        return uploaded.getvalue().decode("utf-8", errors="replace")[:MAX_TEXT_CHARS], None
    except Exception:
        return "", "텍스트 파일을 읽을 수 없습니다. UTF-8 인코딩 파일인지 확인해주세요."


def gemini_text(api_key: str, prompt: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return (response.text or "").strip()


def generate_search_queries(topic: str, purpose: str, purpose_note: str, api_key: str) -> list[str]:
    prompt = f"""보고 주제와 목적을 조사할 짧은 명사형 검색 키워드를 최대 3개 작성하세요.
- 보고 주제: {topic}
- 보고 목적: {purpose}
- 목적 해석 특별 요구: {purpose_note or '없음'}
규칙: 조사·문장 금지, 키워드마다 1~3어절, 쉼표로만 구분, 설명 금지."""
    raw = gemini_text(api_key, prompt)
    items = [re.sub(r"[^0-9A-Za-z가-힣 .&+-]", "", item).strip() for item in raw.split(",")]
    return [item[:80] for item in items if item][:3] or [topic[:80]]


def duplicate(title: str, titles: list[str]) -> bool:
    normalized = re.sub(r"\s+", "", title).lower()
    return not normalized or any(normalized == existing for existing in titles)


def collect_search_results(queries: list[str]) -> list[Source]:
    sources: list[Source] = []
    titles: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ReportResearchBot/1.0)"}
    for query in queries:
        encoded = urllib.parse.quote(query)
        try:
            response = requests.get(f"https://news.search.naver.com/news_search.naver?where=rss&query={encoded}", headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall("./channel/item")[:4]:
                title, url = clean_text(item.findtext("title", "")), item.findtext("link", "")
                raw_date = item.findtext("pubDate", "")
                try: published = parsedate_to_datetime(raw_date).strftime("%Y-%m-%d")
                except (TypeError, ValueError): published = "날짜 미상"
                if public_http_url(url) and not duplicate(title, titles):
                    titles.append(re.sub(r"\s+", "", title).lower())
                    sources.append(Source("네이버 뉴스", title, url, published))
        except (requests.RequestException, ET.ParseError):
            pass
        try:
            results = DDGS().text(query, region="kr-kr", max_results=4)
            for result in results:
                title, url = clean_text(str(result.get("title", ""))), str(result.get("href") or result.get("url") or "")
                if public_http_url(url) and not duplicate(title, titles):
                    titles.append(re.sub(r"\s+", "", title).lower())
                    sources.append(Source("웹 검색", title, url, clean_text(str(result.get("date", "날짜 미상")), 40)))
        except Exception:
            pass
    return sources[:10]


async def fetch_body(session: aiohttp.ClientSession, source: Source) -> Source:
    if not public_http_url(source.url):
        return source
    try:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
        async with session.get(source.url, timeout=timeout, allow_redirects=False) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if response.status != 200 or "text/html" not in content_type:
                return source
            data = await response.content.read(MAX_SCRAPED_CHARS * 5)
            soup = BeautifulSoup(data.decode(response.charset or "utf-8", errors="ignore"), "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
                tag.decompose()
            body = clean_text(soup.get_text(" ", strip=True))
            return Source(source.source_type, source.title, source.url, source.date, body)
    except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeError):
        return source


async def search_and_scrape_async(queries: list[str]) -> list[Source]:
    sources = collect_search_results(queries)
    connector = aiohttp.TCPConnector(limit=5, ssl=True)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ReportResearchBot/1.0)"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        return await asyncio.gather(*(fetch_body(session, item) for item in sources))


def build_context(sources: list[Source]) -> str:
    return "\n".join(
        f"[{index}] {item.source_type} | 작성일: {item.date} | 제목: {item.title}\nURL: {item.url}\n요약/본문: {item.body or '검색 결과 제목 기반'}"
        for index, item in enumerate(sources, 1)
    )[:20_000]


def generate_report(topic: str, purpose: str, target: str, features: list[str], primary: str, notes: dict[str, str], context: str, api_key: str) -> str:
    today = datetime.now().strftime("%Y년 %m월 %d일")
    prompt = f"""당신은 비즈니스 보고서 기획 AI입니다. 오늘은 {today}입니다.

입력: 주제={topic}; 목적={purpose}; 보고 대상={target}; 청중 성향={', '.join(features) or '일반'}
원천 데이터(최우선 근거): {primary or '없음'}
목적 해석 요구: {notes['purpose'] or '없음'}
정보 탐색 요구: {notes['research'] or '없음'}
흐름 설계 요구: {notes['flow'] or '없음'}
검색 근거: {context or '없음'}

규칙:
- 원천 데이터와 검색 근거에서 확인된 사실만 사용하고, 수치·날짜·출처를 만들지 마세요.
- 발행일 기준 과거 정보는 배경·경과·원인, 최근 정보는 현황·동향·향후 방안에 배치하세요.
- 목적 사분면 하나를 선택: Trouble Shooting(현황→이슈→원인/배경→방안→조치→효과), Navigating(동향→개요→전망→목적/목표→방안→계획), Sensing(동향→특징→콘셉트→기대효과→방안), Investigating(배경→현황→경과→결과→전망).
- 목적·목표·배경·현황·동향·이슈·개요·방안·전망·경과·효과·결과·계획·콘셉트·특징·조치 중 4~7개를 청중 성향에 맞게 배열하세요.
- 사용한 검색 근거 번호만 마지막 인덱스에 적으세요. 근거가 없으면 0입니다.

출력 형식:
### 🎯 도출된 목적 사분면: [이름]
* **선정 사유:** ...
### 📌 추천 스토리라인 (보고 목적 맞춤형 정렬 결과)
1. **[요소명]**
   - ...
### 💡 보고 대상 맞춤형 Briefing Tip
* **보고 톤앤매너 & 강조 포인트:** ...
* **논리 전개 핵심 가이드:** ...
* **예상 Q&A 및 대처 방안:** ...
[참조_인덱스: 1, 3]"""
    return gemini_text(api_key, prompt)


def used_source_indices(report: str) -> list[int]:
    match = re.search(r"\[참조_인덱스:\s*([0-9,\s]+)\]", report)
    return [int(value) for value in match.group(1).split(",") if value.strip().isdigit() and int(value) > 0] if match else []


def apply_storyline_studio_design() -> None:
    """React 웹앱의 STORYLINE STUDIO 시각 토큰을 Streamlit에 적용합니다."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;700&family=Pretendard:wght@400;500;600;700;800&display=swap');
    :root { --navy:#102a43; --teal:#087e8b; --canvas:#eef5f7; --card:#f7fbff; --soft:#dcebef; --border:#b9d2da; --muted:#486581; --gold:#f0b429; }
    .stApp { background:var(--canvas); color:var(--navy); font-family:Pretendard,'Apple SD Gothic Neo',sans-serif; }
    [data-testid='stHeader'] { background:transparent; }
    [data-testid='stSidebar'] { background:var(--navy); }
    [data-testid='stSidebar'] * { color:#f7fbff; }
    [data-testid='stSidebar'] [data-testid='stTextInput'] input, [data-testid='stSidebar'] textarea { background:#173b56; border-color:#77c7ce; color:#f7fbff; }
    [data-testid='stSidebar'] [data-testid='stExpander'] { background:#173b56; border:1px solid #2a5874; border-radius:12px; }
    .block-container { max-width:1180px; padding-top:0.8rem; padding-bottom:3rem; }
    .studio-header { background:var(--navy); border-bottom:4px solid var(--teal); margin:0 calc(50% - 50vw) 1.9rem; padding:18px max(20px,calc((100vw - 1180px)/2)); color:#f7fbff; display:flex; align-items:center; justify-content:space-between; }
    .brand { display:flex; gap:10px; align-items:center; font-size:10px; line-height:1.18; font-weight:800; letter-spacing:1.5px; }
    .brand-mark { width:32px; height:32px; display:inline-flex; align-items:center; justify-content:center; border:1px solid #77c7ce; color:#77c7ce; border-radius:9px; font:700 20px Georgia,serif; letter-spacing:0; }
    .status { color:#bdeff2; font-size:12px; font-weight:700; }
    .hero { min-height:276px; border-radius:24px; padding:38px 42px; background:radial-gradient(circle at 87% -20%,#1d4b63 0 17%,transparent 17.4%),var(--navy); color:#f7fbff; margin-bottom:18px; }
    .eyebrow,.section-tag { color:var(--teal); font-size:11px; font-weight:800; letter-spacing:1.2px; margin:0 0 10px; }
    .hero h1 { font-family:'Noto Serif KR',serif; font-size:42px; letter-spacing:-2px; line-height:1.2; margin:0; }
    .hero em { color:#77c7ce; font-style:normal; }
    .hero p:last-child { max-width:570px; color:#dcebef; font-size:15px; line-height:1.7; margin-top:16px; }
    .panel { background:var(--card); border:1px solid var(--border); border-radius:24px; padding:28px; box-shadow:0 12px 30px rgba(16,42,67,.08); margin-bottom:18px; }
    [data-testid='stVerticalBlockBorderWrapper']:has(.input-workbench) { background:var(--card); border:1px solid var(--border); border-radius:24px; box-shadow:0 12px 30px rgba(16,42,67,.08); }
    [data-testid='stVerticalBlockBorderWrapper']:has(.input-workbench) > div { padding:26px; }
    .soft-panel { background:var(--soft); border:1px solid var(--border); box-shadow:none; border-radius:20px; padding:22px; }
    .panel-title { font:700 24px/1.3 'Noto Serif KR',serif; letter-spacing:-1px; margin:0; color:var(--navy); }
    .subcopy { color:var(--navy)!important; font-size:13px!important; font-weight:normal!important; line-height:1.6; }
    .divider-title { border-top:1px solid var(--border); margin-top:27px; padding-top:23px; }
    .required { color:#b42318; font-size:12px; font-weight:700; }
    @media (prefers-color-scheme: light) {
        div[data-testid="stSelectbox"] [data-baseweb="select"],
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] {
            background-color: white !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
            border-radius: 10px !important;
        }
    }
    [data-testid='stTextInput'] label, [data-testid='stTextArea'] label, [data-testid='stSelectbox'] label, [data-testid='stMultiSelect'] label, [data-testid='stFileUploader'] label { color:var(--navy)!important; font-weight:700!important; font-size:13px!important; }
    [data-testid='stTextInput'] input, [data-testid='stTextArea'] textarea, [data-testid='stSelectbox'] [data-baseweb='select'] > div { border:1px solid var(--border)!important; border-radius:10px!important; background:#fff!important; color:var(--navy)!important; }
    [data-testid='stTextInput'] input:focus, [data-testid='stTextArea'] textarea:focus { border-color:var(--teal)!important; box-shadow:0 0 0 3px rgba(8,126,139,.17)!important; }
    [data-testid='stMultiSelect'] [data-baseweb='select'] > div { min-height:46px; }
    .stButton>button, .stFormSubmitButton>button { width:100%; border:0!important; border-radius:10px!important; background:var(--teal)!important; color:#f7fbff!important; font-weight:800!important; min-height:47px; }
    .stButton>button:hover,.stFormSubmitButton>button:hover { background:#066a75!important; }
    [data-testid='stFileUploader'] { border:1px dashed #87aeb9; border-radius:12px; padding:12px; background:#f0f7f8; }
    [data-testid='stFileUploader'] button { color:var(--navy)!important; border-color:var(--navy)!important; }

    /* ANALYSIS PATH 박스 안의 Expander 스타일링 */
    .soft-panel [data-testid='stExpander'] { background:#ffffff!important; border:1px solid var(--border)!important; border-radius:10px!important; margin-top:10px!important; margin-bottom:8px!important; }
    .soft-panel [data-testid='stExpander'] summary { font-weight:700!important; color:var(--navy)!important; font-size:13.5px!important; }

    .metric-row { border-top:1px solid var(--border); padding:14px 0; }
    .metric-row b { display:block; color:var(--muted); font-size:12px; }
    .metric-row span { display:block; margin-top:5px; color:var(--navy); font-size:13px; font-weight:700; line-height:1.5; }
    .result-empty { text-align:center; border:1px dashed var(--border); border-radius:16px; padding:40px 22px; background:#f9fcfd; }
    .result-icon { font-size:28px; }
    .result-empty h3 { font:700 20px/1.4 'Noto Serif KR',serif; color:var(--navy); }
    .result-empty p { color:var(--muted); font-size:14px; line-height:1.7; }
    .source-note { background:#fff2d8; color:#6b4707; border-radius:10px; padding:12px; font-size:13px; line-height:1.55; }

    /* 물음표 Hover CSS 툴팁 구현 */
    .tooltip-container { position: relative; display: inline-block; cursor: pointer; }
    .tooltip-icon { width: 18px; height: 18px; border-radius: 50%; border: 1.5px solid #78909e; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #78909e; font-weight: 800; }
    .tooltip-text {
        visibility: hidden;
        width: 230px;
        background-color: var(--navy);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 11.5px;
        position: absolute;
        z-index: 10;
        bottom: 125%;
        left: 50%;
        margin-left: -115px;
        opacity: 0;
        transition: opacity 0.2s;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .tooltip-container:hover .tooltip-text { visibility: visible; opacity: 1; }

    /* 제목 레이블과 오른쪽 설명 텍스트 수평 정렬 스타일 */
    .title-flex-container { display: flex; justify-content: space-between; align-items: flex-end; width: 100%; margin-bottom: 8px; }
    .right-aligned-subcopy { color: #78909e!important; font-size: 12px!important; font-weight: normal!important; text-align: right; }

    @media(max-width:760px){ .studio-header{padding:16px 18px}.status{display:none}.hero{padding:30px 24px}.hero h1{font-size:34px}.panel{padding:22px}.block-container{padding-left:13px;padding-right:13px;} }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="보고서 스토리라인 기획 에이전트", layout="wide")
    apply_storyline_studio_design()
    st.markdown("<div class='studio-header'><div class='brand'><span class='brand-mark'>S</span><span>STORYLINE<br>STUDIO</span></div><span class='status'>✦ Gemini · 공개 웹 검색 기반 분석</span></div>", unsafe_allow_html=True)

    if "path_notes" not in st.session_state:
        st.session_state.path_notes = {key: "" for key in PATH_HELP}

    with st.sidebar:
        st.header("🔑 API 설정")
        api_key = st.text_input("Gemini API Key", type="password", help="키는 현재 실행에만 사용되며 저장하거나 화면에 표시하지 않습니다.")
        st.caption("Google AI Studio에서 발급한 Gemini API 키를 입력하세요.")

    hero_col, path_col = st.columns([1.7, 0.9], gap="large")
    with hero_col:
        st.markdown("""<div class='hero'><p class='eyebrow'>✦ REPORT DESIGN WORKBENCH</p><h1>📊 보고서 스토리라인<br><em>기획 에이전트</em></h1><p>Gemini와 공개 웹 검색을 활용해 정보 탐색, 보고서 구성 요소 정렬, 스토리 라인 추천 등을 통합 수행합니다.</p></div>""", unsafe_allow_html=True)

    with path_col:
        st.markdown("<div class='panel soft-panel'><p class='section-tag'>ANALYSIS PATH</p><p class='panel-title'>분석 경로</p>", unsafe_allow_html=True)
        for key, (label, caption) in PATH_HELP.items():
            with st.expander(f"{label}  ›"):
                st.caption(caption)
                st.session_state.path_notes[key] = st.text_area("특별 요구사항", st.session_state.path_notes[key], max_chars=500, key=f"note_{key}")
        st.markdown("</div>", unsafe_allow_html=True)

    form_col, brief_col = st.columns([1.7, 0.9], gap="large")
    with form_col:
        with st.container(border=True):
            st.markdown("<div class='input-workbench'><p class='section-tag'>01 / INPUT</p><p class='panel-title'>보고 주제 및 개요 <span class='required'>* 필수 입력</span></p></div>", unsafe_allow_html=True)
            topic_col, purpose_col = st.columns(2)
            with topic_col: topic = st.text_input("보고 주제 및 개요 *", placeholder="예: 통신 시장 동향 및 사업 기회", max_chars=160)
            with purpose_col: purpose = st.text_input("보고 목적 *", placeholder="예: 신규 진출 전략 수립 및 투자 검토", max_chars=220)
            target = st.selectbox("보고 대상 (청중 유형) *", AUDIENCE_OPTIONS)

            st.markdown("""
            <div class='divider-title'>
                <p class='section-tag'>02 / AUDIENCE LENS</p>
                <div class='title-flex-container'>
                    <p class='panel-title' style='margin:0;'>청중 상세 성향 <span class='subcopy'>선택</span></p>
                    <span class='right-aligned-subcopy'>미선택 시 청중 유형의 기본 성향으로 정렬합니다.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            selected_features = []
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox(FEATURE_OPTIONS[0], key="feature_0_col1"):
                    selected_features.append(FEATURE_OPTIONS[0])
                if st.checkbox(FEATURE_OPTIONS[1], key="feature_1_col1"):
                    selected_features.append(FEATURE_OPTIONS[1])
            with col2:
                if st.checkbox(FEATURE_OPTIONS[2], key="feature_2_col2"):
                    selected_features.append(FEATURE_OPTIONS[2])
                if st.checkbox(FEATURE_OPTIONS[3], key="feature_3_col2"):
                    selected_features.append(FEATURE_OPTIONS[3])

            features = selected_features

            st.markdown("""
            <div class='divider-title'>
                <p class='section-tag'>03 / PRIMARY SOURCE</p>
                <div class='title-flex-container'>
                    <p class='panel-title' style='margin:0;'>원천 데이터 <span class='subcopy'>선택</span></p>
                    <span class='right-aligned-subcopy'>입력한 원천 데이터는 분석의 최우선 근거로 반영됩니다.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            primary_text = st.text_area("직접 입력 원천 데이터 또는 참고 URL", max_chars=MAX_TEXT_CHARS, placeholder="회의록, 기사 요약, 참고 URL 등 분석에 반영할 원천 데이터를 입력하세요.")
            st.caption(f"입력 내용은 분석 요청에만 사용합니다. · {len(primary_text):,} / {MAX_TEXT_CHARS:,}")
            uploaded = st.file_uploader("참고 파일 첨부", type=sorted(ALLOWED_EXTENSIONS), help="PDF, DOCX, TXT, MD, CSV / 최대 10MB. 자동 반영은 TXT·MD·CSV만 지원합니다.")
            submitted = st.button("🚀 에이전트 실행", type="primary", use_container_width=True)

    with brief_col:
        feature_summary = ", ".join(features) if features else "청중 유형 기본 성향"
        st.markdown(f"""
        <div class='panel soft-panel'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <p class='section-tag' style='margin:0;'>INPUT REVIEW</p>
                <div class='tooltip-container'>
                    <div class='tooltip-icon'>?</div>
                    <span class='tooltip-text'>입력값이 변경될 때마다 현재 분석 기준을 반영합니다.</span>
                </div>
            </div>
            <p class='panel-title' style='margin-top:6px; margin-bottom:18px;'>분석 기준</p>
            <div class='metric-row'><b>청중</b><span>{html.escape(target)}</span></div>
            <div class='metric-row'><b>근거 우선순위</b><span>입력 원천 데이터 → 공개 검색 근거</span></div>
            <div class='metric-row'><b>보고 성향</b><span>{html.escape(feature_summary)}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # 💡 동적 UI 변경을 위한 st.empty 영역 선언
    output_area = st.empty()

    # 실행 전: PREVIEW 상태로 대기 화면 노출
    if not submitted:
        output_area.markdown("<div class='panel'><div style='display:flex;justify-content:space-between;gap:12px;margin-bottom:16px'><div><p class='section-tag'>OUTPUT / PREVIEW</p><p class='panel-title'>스토리라인 결과 영역</p></div><div style='background:#e2ebf0;color:#274757;font-size:12px;font-weight:700;padding:6px 14px;border-radius:16px;display:flex;align-items:center;gap:6px;height:fit-content'><span style='font-size:14px'>🕒</span> 실행 대기</div></div><div class='result-empty'><div class='result-icon'>▤</div><h3>입력을 바탕으로 설득력 있는 보고 흐름을 설계합니다.</h3><p>실행하면 목적 사분면, 추천 스토리라인, 청중 맞춤형 브리핑 팁과 실제 활용한 검색 근거가 이 영역에 표시됩니다.</p><div style='display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:16px'><span style='background:#e4edf0;color:#2b4d5e;font-size:12px;font-weight:700;padding:5px 14px;border-radius:14px'>Trouble Shooting</span><span style='background:#e4edf0;color:#2b4d5e;font-size:12px;font-weight:700;padding:5px 14px;border-radius:14px'>Navigating</span><span style='background:#e4edf0;color:#2b4d5e;font-size:12px;font-weight:700;padding:5px 14px;border-radius:14px'>Sensing</span><span style='background:#e4edf0;color:#2b4d5e;font-size:12px;font-weight:700;padding:5px 14px;border-radius:14px'>Investigating</span></div></div><div class='source-note' style='margin-top:14px;display:flex;align-items:center;gap:8px'><span>🔍</span><span>유의미한 외부 검색 근거가 없거나 연관성이 낮을 경우, 입력하신 원천 데이터와 내장 AI의 분석을 바탕으로 스토리라인을 정렬합니다.</span></div></div>", unsafe_allow_html=True)
        return

    # 유효성 검사
    if not topic.strip() or not purpose.strip():
        st.error("필수 항목인 보고 주제 및 보고 목적을 입력해주세요."); return
    if not api_key.strip():
        st.warning("사이드바에 Gemini API Key를 입력해주세요."); return
    file_text, error = validate_file(uploaded)
    if error:
        st.error(error); return
    primary = (primary_text.strip() + "\n" + file_text.strip()).strip()[:MAX_TEXT_CHARS]

    try:
        with st.spinner("🔍 [1단계] Gemini가 보고 목적에 맞는 검색 키워드를 정리하고 있습니다..."):
            queries = generate_search_queries(topic, purpose, st.session_state.path_notes["purpose"], api_key.strip())
        st.caption("검색 키워드: " + ", ".join(queries))

        with st.spinner("🌐 [2~3단계] 비동기 스크래퍼가 공개 검색 결과와 웹 본문을 수집하고 있습니다..."):
            sources = asyncio.run(search_and_scrape_async(queries))

        with st.spinner("🧠 [4단계] Gemini가 mySUNI 프레임워크에 맞춰 스토리라인을 설계하고 있습니다..."):
            report = generate_report(topic, purpose, target, features, primary, st.session_state.path_notes, build_context(sources), api_key.strip())

        cleaned = re.sub(r"\[참조_인덱스:\s*[0-9,\s]+\]", "", report).strip()

        # 💡 실행 완료: 동일한 output_area 위치를 OUTPUT / COMPLETE 패널 및 결과물로 즉시 교체
        with output_area.container():
            st.markdown("<div class='panel'><div style='display:flex;justify-content:space-between;gap:12px;margin-bottom:16px'><div><p class='section-tag'>OUTPUT / COMPLETE</p><p class='panel-title'>스토리라인 결과</p></div><div style='background:#d1e7dd;color:#0f5132;font-size:12px;font-weight:700;padding:6px 14px;border-radius:16px;display:flex;align-items:center;gap:6px;height:fit-content'><span style='font-size:14px'>✅</span> 생성 완료</div></div></div>", unsafe_allow_html=True)
            st.success("🎉 스토리라인 정렬이 완료되었습니다!")
            st.markdown(cleaned)

            selected = [sources[index - 1] for index in used_source_indices(report) if index <= len(sources)]
            st.markdown("### 📂 참조 출처 목록")
            if selected:
                for source in selected:
                    st.write(f"• [{source.source_type}] {source.date} | {source.title}")
                    st.link_button("원본 링크 열기", source.url)
            else:
                st.markdown("<div class='source-note'>💡 참고 안내: 유의미한 외부 검색 근거가 없거나 연관성이 낮아, Gemini의 분석과 입력하신 원천 데이터를 바탕으로 스토리라인을 정렬했습니다.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"처리 중 문제가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
