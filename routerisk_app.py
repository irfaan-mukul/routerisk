"""
RouteRisk — Shipment Delay Predictor
Enter shipment details, get a real-time risk assessment powered by live web data and AI.
Run with: python routerisk_app.py
Open: http://localhost:5002
"""

import os
import json
import requests
import anthropic
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

app = Flask(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

SEARCH_PROMPT = """You are a freight risk analyst. Based on this shipment, generate 4 targeted web search queries 
to find CURRENT disruptions, delays, or risks affecting this specific route and mode.

Shipment:
- Mode: {mode}
- Origin: {origin}
- Destination: {destination}
- Carrier: {carrier}
- Ship Date: {ship_date}
- Commodity: {commodity}
- Today's Date: {today}

Return ONLY a JSON array of 4 search query strings. Focus on finding the MOST RECENT information:
1. Current {mode_lower} disruptions on this specific route or region — include the current month and year or "this week" in the query
2. {carrier} recent operational issues or delays — include the current month and year
3. Port/airport congestion or closures at origin or destination — include "latest" or current month
4. Geopolitical or weather events affecting this lane — include "current" or "2026"

Make queries time-specific to surface results from the last 7 days where possible.
Example format: ["query 1", "query 2", "query 3", "query 4"]
Respond with ONLY the JSON array."""

ASSESSMENT_PROMPT = """You are a senior freight risk analyst with 20 years of experience in global logistics.
Assess the delay risk for this shipment based on current intelligence gathered from web searches.

SHIPMENT DETAILS:
- Mode: {mode}
- Origin: {origin}
- Destination: {destination}  
- Carrier: {carrier}
- Ship Date: {ship_date}
- Commodity: {commodity}

CURRENT INTELLIGENCE FROM WEB SEARCHES:
{search_results}

Provide a comprehensive risk assessment as a JSON object:
{{
  "risk_level": "LOW" | "MODERATE" | "HIGH" | "CRITICAL",
  "risk_score": number between 0-100,
  "headline": "one sentence summary of the risk situation",
  "risk_factors": [
    {{
      "category": "WEATHER" | "PORT_CONGESTION" | "CARRIER" | "GEOPOLITICAL" | "ROUTE" | "CUSTOMS" | "CAPACITY" | "SEASONAL",
      "severity": "LOW" | "MODERATE" | "HIGH",
      "title": "short title",
      "description": "2-3 sentence description of this specific risk factor and how it affects this shipment"
    }}
  ],
  "estimated_delay_days": {{
    "min": number,
    "max": number,
    "most_likely": number
  }},
  "recommendations": [
    {{
      "priority": "IMMEDIATE" | "SOON" | "OPTIONAL",
      "action": "specific actionable recommendation"
    }}
  ],
  "data_freshness": "assessment of how current the intelligence is",
  "confidence": "HIGH" | "MODERATE" | "LOW",
  "confidence_reason": "brief explanation of confidence level"
}}

SCORING METHOD — follow this step-by-step process:

STEP 1: SET BASELINE SCORE based on route type
- Domestic air (under 500 miles): baseline 8
- Domestic air (over 500 miles): baseline 12
- International air: baseline 20
- Domestic ocean/inland: baseline 15
- International ocean (same region, e.g. intra-Asia): baseline 25
- International ocean (cross-ocean, e.g. transpacific): baseline 35
- Routes transiting known chokepoints (Suez, Panama Canal, Strait of Malacca): baseline 40

STEP 2: ADJUST FOR COMMODITY RISK
- Low risk (documents, general freight, textiles): no adjustment
- Medium risk (electronics, machinery, food products): +3
- High risk (hazmat, lithium batteries, perishables, pharmaceuticals, high-value): +8
- Temperature controlled or time-sensitive: +5

STEP 3: ADJUST FOR CONFIRMED ACTIVE DISRUPTIONS (from search results only)
Only add points for disruptions that are:
  (a) confirmed in search results from the last 7 days, AND
  (b) directly affect this specific route, carrier, or origin/destination

CRITICAL: Score disruptions by ACTUAL DELAY IMPACT, not just headline severity.
Airport delays of a few hours are NOT equivalent to ocean route diversions adding weeks.

Air freight disruption scoring (delays measured in hours to 1-2 days):
- Minor airport congestion (delays under 4 hours): +3-5
- Significant airport congestion (delays of 4-12 hours): +5-10
- Airport partial closure or severe operational issues: +10-15
- Carrier operational crisis (fleet-wide): +8-12
- Airport full closure: +15-20

Ocean freight disruption scoring (delays measured in days to weeks):
- Minor port congestion (delays of 1-3 days): +5-8
- Significant port congestion (delays of 3-7 days): +8-15
- Canal/strait restriction reducing capacity: +12-18
- Canal/strait diversion requiring rerouting (adds 7+ days): +18-25
- Active military conflict on route: +20-30
- Complete route shutdown or force majeure: +25-35

General disruption scoring (both modes):
- Weather event on route (temporary): +5-10
- Geopolitical tensions (elevated but not active conflict): +5-10
- Active sanctions or trade restrictions affecting this carrier/commodity: +10-15
- Customs backlog or regulatory changes at destination: +5-10

ANTI-STACKING RULE — CRITICAL:
Do NOT count the same underlying disruption multiple times under different names.
- Airport congestion + carrier delays at that same airport + FAA restrictions at that airport = ONE disruption, not three. Score the single most severe aspect only.
- Port congestion + vessel delays at that same port + terminal handling issues = ONE disruption, score the worst aspect.
- A carrier's operational issues caused BY an airport/port problem are not a separate disruption — they are part of the same event.
- Maximum 3-4 truly independent risk factors per assessment. If you find more, consolidate related ones.
- MAXIMUM total disruption adjustment for domestic air routes: +25 points (even with multiple issues, domestic air delays rarely exceed 1-2 days)
- MAXIMUM total disruption adjustment for international air routes: +35 points
- MAXIMUM total disruption adjustment for ocean routes: +45 points

STEP 4: APPLY REDUCTIONS for carrier resilience and alternatives
- Major integrators (FedEx, UPS, DHL Express) on domestic routes: -15
- Short routes under 500 miles with ground fallback: -10
- Major airlines/carriers with daily frequency on this lane: -5
- Multiple alternative carriers readily available on this lane: -5

STEP 5: APPLY RECENCY DISCOUNT
- All disruption data older than 7 days: reduce disruption adjustment by 70%
- All disruption data older than 14 days: reduce to zero (historical context only)
- Ship date more than 14 days away: reduce all transient disruption adjustments by 50% (only structural risks like ongoing conflicts retain full weight)

STEP 6: DETERMINE FINAL SCORE AND LEVEL
- Add baseline + commodity + disruptions - reductions - recency discount
- Minimum score: 5 (nothing is truly zero risk)
- Maximum score: 98
- Map to level: 0-25 = LOW, 26-50 = MODERATE, 51-75 = HIGH, 76-100 = CRITICAL

ABSOLUTE RULE — NO MANUAL OVERRIDES OR CREATIVE WORKAROUNDS:
The final score MUST equal: baseline + commodity + disruptions (CAPPED) - reductions.
That is the ONLY formula. There are no bonus points, no exceptions, no special circumstances.

You may NOT:
- Add extra points beyond the cap for any reason
- Apply the disruption cap and then add more points on top of it
- Invent justifications to exceed the caps (e.g. "both endpoints" or "severity warrants")
- Use phrases like "adjusted to" or "warranted" or "retaining full points" to override the math
- Score disruptions twice by splitting them into sub-categories that each get points

The caps are hard limits:
- Domestic air disruptions: MAXIMUM +25, regardless of how severe
- International air disruptions: MAXIMUM +35, regardless of how severe
- Ocean disruptions: MAXIMUM +45, regardless of how severe

SHOW YOUR MATH EXACTLY LIKE THIS in confidence_reason:
"Baseline [X] + commodity [Y] + disruptions [Z, capped at max] - reductions [W] = [FINAL SCORE]"
The final score in your JSON MUST match this math exactly. Any discrepancy is an error.

CRITICAL RULE — NO DISRUPTIONS FOUND:
If search results contain NO specific, confirmed, recent disruptions on this route, the score MUST stay within the baseline range (baseline + commodity adjustment only, typically 8-35 depending on route type). Do NOT invent or assume risks that are not in the search results. "No news is good news" — a clean search is a positive signal.

In your confidence_reason, briefly show your scoring math: "Baseline X + commodity Y + disruption Z - reductions W = final score."

Be specific and data-driven. Reference actual current events from the search results where possible.
If search results are limited or stale, note that in confidence_reason and keep the score near baseline.
Respond with ONLY the JSON object."""

# ── Web search ────────────────────────────────────────────────────────────────

def web_search(query: str) -> list:
    """Search using Brave API and return results."""
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return []
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": api_key},
            params={"q": query, "count": 5, "search_lang": "en", "country": "us"},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("web", {}).get("results", [])
        return [{"title": r.get("title", ""), "snippet": r.get("description", ""), "url": r.get("url", "")} for r in results]
    except Exception:
        return []


def generate_search_queries(shipment: dict) -> list:
    """Ask Claude to generate targeted search queries for this shipment."""
    client = anthropic.Anthropic()
    today = datetime.now().strftime("%B %d, %Y")
    prompt = SEARCH_PROMPT.format(
        mode=shipment["mode"],
        mode_lower=shipment["mode"].lower(),
        origin=shipment["origin"],
        destination=shipment["destination"],
        carrier=shipment["carrier"],
        ship_date=shipment["ship_date"],
        commodity=shipment["commodity"],
        today=today,
    )
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start != -1 and end > start:
        return json.loads(raw[start:end])
    return []


def assess_risk(shipment: dict, search_results: list) -> dict:
    """Generate risk assessment based on shipment details and search results."""
    client = anthropic.Anthropic()

    results_text = ""
    for i, result in enumerate(search_results, 1):
        results_text += f"\nSearch {i}: {result.get('query', '')}\n"
        for r in result.get("results", []):
            results_text += f"  - {r['title']}: {r['snippet']}\n"

    prompt = ASSESSMENT_PROMPT.format(
        mode=shipment["mode"],
        origin=shipment["origin"],
        destination=shipment["destination"],
        carrier=shipment["carrier"],
        ship_date=shipment["ship_date"],
        commodity=shipment["commodity"],
        search_results=results_text or "No search results available.",
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(raw[start:end])
    raise ValueError("Could not parse risk assessment")


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RouteRisk — Shipment Delay Predictor</title>
<meta property="og:title" content="RouteRisk — Shipment Delay Predictor">
<meta property="og:description" content="Real-time AI risk assessment for global shipments. Enter your route and get an instant delay prediction based on live intelligence.">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #F7F8FA;
    --surface: #FFFFFF;
    --surface-2: #F0F2F5;
    --border: #E2E6ED;
    --border-2: #CBD2DC;
    --text: #0F1923;
    --text-mid: #4A5568;
    --text-light: #8896A8;
    --navy: #0F2B3C;
    --navy-light: #1A3D52;
    --accent: #0066CC;
    --accent-light: #E8F0FC;
    --green: #0A7A4B;
    --green-bg: #E6F7EF;
    --yellow: #B45309;
    --yellow-bg: #FEF3C7;
    --orange: #C2410C;
    --orange-bg: #FEE8DF;
    --red: #B91C1C;
    --red-bg: #FEE2E2;
    --font-display: 'Playfair Display', Georgia, serif;
    --font-body: 'Manrope', system-ui, sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 24px rgba(0,0,0,0.10), 0 1px 4px rgba(0,0,0,0.06);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body { background: var(--bg); color: var(--text); font-family: var(--font-body); min-height: 100vh; }

  /* Top bar */
  .topbar {
    background: var(--navy);
    padding: 0 32px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .topbar-logo { display: flex; align-items: center; gap: 10px; }
  .topbar-icon { width: 32px; height: 32px; background: var(--accent); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 15px; }
  .topbar-name { font-family: var(--font-body); font-size: 17px; font-weight: 800; color: white; letter-spacing: -0.3px; }
  .topbar-name span { color: #60A5FA; }
  .topbar-tag { font-family: var(--font-mono); font-size: 11px; color: #60A5FA; background: rgba(96,165,250,0.12); padding: 4px 10px; border-radius: 4px; letter-spacing: 0.5px; }

  /* Main layout */
  .main { max-width: 1000px; margin: 0 auto; padding: 40px 24px 60px; }

  /* Hero */
  .hero { margin-bottom: 36px; }
  .hero h1 { font-family: var(--font-display); font-size: clamp(28px, 4vw, 40px); color: var(--navy); line-height: 1.2; margin-bottom: 10px; }
  .hero p { font-size: 15px; color: var(--text-mid); font-weight: 400; max-width: 520px; line-height: 1.7; }

  /* Form card */
  .form-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
  }

  .form-card-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  .form-group { display: flex; flex-direction: column; gap: 6px; }
  .form-group.full { grid-column: 1 / -1; }

  label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-mid);
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  input, select {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 11px 14px;
    font-family: var(--font-body);
    font-size: 14px;
    color: var(--text);
    transition: border-color 0.2s, box-shadow 0.2s;
    width: 100%;
    appearance: none;
  }

  input:focus, select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,102,204,0.1);
  }

  input::placeholder { color: var(--text-light); }

  /* Mode selector */
  .mode-selector { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

  .mode-option { position: relative; }
  .mode-option input[type="radio"] { position: absolute; opacity: 0; width: 0; height: 0; }

  .mode-label {
    display: flex; align-items: center; gap: 10px;
    border: 1.5px solid var(--border); border-radius: 10px;
    padding: 14px 16px; cursor: pointer; transition: all 0.2s;
    background: var(--surface);
  }

  .mode-label:hover { border-color: var(--accent); background: var(--accent-light); }
  .mode-option input:checked + .mode-label { border-color: var(--accent); background: var(--accent-light); }

  .mode-icon { font-size: 22px; }
  .mode-text { font-size: 14px; font-weight: 600; color: var(--text); }
  .mode-sub { font-size: 11px; color: var(--text-light); }

  /* Analyze button */
  .btn-analyze {
    width: 100%; padding: 15px;
    background: var(--navy); color: white;
    border: none; border-radius: 10px;
    font-family: var(--font-body); font-size: 15px; font-weight: 700;
    cursor: pointer; transition: all 0.2s;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    letter-spacing: 0.3px; margin-top: 8px;
  }

  .btn-analyze:hover { background: var(--navy-light); transform: translateY(-1px); box-shadow: var(--shadow-lg); }
  .btn-analyze:active { transform: translateY(0); }
  .btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

  /* Loading */
  .loading { display: none; background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 48px 32px; text-align: center; margin-bottom: 24px; box-shadow: var(--shadow); }
  .loading.visible { display: block; }

  .loading-route { display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 28px; }
  .route-node { background: var(--surface-2); border: 2px solid var(--border-2); border-radius: 8px; padding: 8px 16px; font-family: var(--font-mono); font-size: 14px; font-weight: 500; color: var(--navy); }
  .route-line { flex: 1; max-width: 80px; height: 2px; background: var(--border); position: relative; overflow: hidden; }
  .route-line::after { content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, var(--accent), transparent); animation: routeFlow 1.5s ease-in-out infinite; }
  @keyframes routeFlow { to { left: 100%; } }
  .route-plane { font-size: 20px; animation: float 2s ease-in-out infinite; }
  @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }

  .loading-steps { display: flex; flex-direction: column; gap: 10px; max-width: 340px; margin: 0 auto; }
  .loading-step { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--text-light); padding: 8px 12px; border-radius: 8px; transition: all 0.3s; }
  .loading-step.active { background: var(--accent-light); color: var(--accent); font-weight: 500; }
  .loading-step.done { color: var(--green); }
  .step-icon { font-size: 16px; width: 20px; text-align: center; }

  /* Results */
  #results { display: none; }
  #results.visible { display: block; }

  /* Risk gauge */
  .risk-header {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
    display: flex;
    gap: 28px;
    align-items: center;
  }

  .risk-gauge {
    flex-shrink: 0;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
  }

  .risk-gauge.LOW { background: conic-gradient(var(--green) 0deg, var(--green) 90deg, var(--border) 90deg); }
  .risk-gauge.MODERATE { background: conic-gradient(#D97706 0deg, #D97706 180deg, var(--border) 180deg); }
  .risk-gauge.HIGH { background: conic-gradient(var(--orange) 0deg, var(--orange) 252deg, var(--border) 252deg); }
  .risk-gauge.CRITICAL { background: conic-gradient(var(--red) 0deg, var(--red) 360deg); }

  .risk-gauge-inner {
    width: 88px; height: 88px; background: var(--surface);
    border-radius: 50%; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
  }

  .risk-score { font-family: var(--font-mono); font-size: 26px; font-weight: 500; line-height: 1; }
  .risk-score-label { font-size: 10px; color: var(--text-light); text-transform: uppercase; letter-spacing: 1px; }

  .risk-gauge.LOW .risk-score { color: var(--green); }
  .risk-gauge.MODERATE .risk-score { color: #D97706; }
  .risk-gauge.HIGH .risk-score { color: var(--orange); }
  .risk-gauge.CRITICAL .risk-score { color: var(--red); }

  .risk-info { flex: 1; }
  .risk-level-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 6px;
    font-family: var(--font-mono); font-size: 11px; font-weight: 500;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 10px;
  }

  .badge-LOW { background: var(--green-bg); color: var(--green); }
  .badge-MODERATE { background: var(--yellow-bg); color: var(--yellow); }
  .badge-HIGH { background: var(--orange-bg); color: var(--orange); }
  .badge-CRITICAL { background: var(--red-bg); color: var(--red); }

  .risk-headline { font-family: var(--font-display); font-size: 20px; color: var(--navy); margin-bottom: 8px; line-height: 1.3; }
  .risk-delay { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
  .delay-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 12px; font-size: 13px; color: var(--text-mid);
  }
  .delay-chip .delay-val { font-family: var(--font-mono); font-weight: 700; color: var(--navy); }
  .delay-chip.primary { color: rgba(255,255,255,0.7); }
  .delay-chip.primary .delay-val { color: white; }
  .delay-chip.risk-LOW { background: var(--green); border-color: var(--green); }
  .delay-chip.risk-MODERATE { background: #D97706; border-color: #D97706; }
  .delay-chip.risk-HIGH { background: var(--orange); border-color: var(--orange); }
  .delay-chip.risk-CRITICAL { background: var(--red); border-color: var(--red); }



  /* Risk factors */
  .section-header { font-size: 13px; font-weight: 700; color: var(--text-light); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px; margin-top: 28px; display: flex; align-items: center; gap: 8px; }

  .factor-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 10px; box-shadow: var(--shadow); display: flex; gap: 16px; align-items: flex-start; }

  .factor-severity {
    flex-shrink: 0; width: 4px; border-radius: 4px; align-self: stretch; min-height: 40px;
  }
  .factor-severity.LOW { background: var(--green); }
  .factor-severity.MODERATE { background: #D97706; }
  .factor-severity.HIGH { background: var(--red); }

  .factor-cat { font-family: var(--font-mono); font-size: 10px; font-weight: 500; color: var(--text-light); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
  .factor-title { font-size: 14px; font-weight: 700; color: var(--navy); margin-bottom: 6px; }
  .factor-desc { font-size: 13px; color: var(--text-mid); line-height: 1.6; }

  /* Recommendations */


  /* Route summary bar */
  .route-summary {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;
    font-size: 13px; color: var(--text-mid); box-shadow: var(--shadow);
  }
  .route-mode {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--navy); color: white;
    padding: 4px 10px; border-radius: 5px;
    font-family: var(--font-mono); font-size: 11px;
    font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;
  }
  .route-detail { display: flex; align-items: center; gap: 5px; }
  .route-arrow { color: var(--text-light); font-size: 14px; }
  .route-node-text { font-weight: 600; color: var(--navy); }
  .route-sep { color: var(--border-2); font-size: 16px; }
  .route-carrier { color: var(--text-mid); }
  .route-date { font-family: var(--font-mono); font-size: 12px; color: var(--text-light); }

  /* Confidence */
  .confidence-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; margin-bottom: 24px; display: flex; gap: 10px; align-items: flex-start; font-size: 13px; color: var(--text-mid); }

  /* Reset */
  .btn-reset { background: transparent; border: 1.5px solid var(--border); color: var(--text-mid); padding: 11px 24px; border-radius: 8px; font-family: var(--font-body); font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.2s; display: block; margin: 0 auto; }
  .btn-reset:hover { border-color: var(--navy); color: var(--navy); }

  /* How it works */
  .help-toggle { display: flex; align-items: center; justify-content: center; gap: 8px; background: transparent; border: 1.5px solid var(--border); color: var(--accent); padding: 10px 20px; border-radius: 8px; font-family: var(--font-body); font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; margin: 0 auto 24px; letter-spacing: 0.3px; }
  .help-toggle:hover { border-color: var(--accent); background: var(--accent-light); }
  .help-toggle .arrow { transition: transform 0.2s; font-size: 11px; }
  .help-toggle.open .arrow { transform: rotate(180deg); }

  .help-panel { display: none; background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 32px; margin-bottom: 28px; box-shadow: var(--shadow); }
  .help-panel.visible { display: block; }
  .help-section { margin-bottom: 22px; }
  .help-section:last-child { margin-bottom: 0; }
  .help-section h3 { font-size: 14px; font-weight: 700; color: var(--navy); margin-bottom: 8px; }
  .help-section p { font-size: 13px; color: var(--text-mid); line-height: 1.7; margin-bottom: 6px; }
  .help-steps { list-style: none; padding: 0; counter-reset: steps; }
  .help-steps li { counter-increment: steps; display: flex; gap: 12px; margin-bottom: 10px; font-size: 13px; color: var(--text); line-height: 1.6; }
  .help-steps li::before { content: counter(steps); flex-shrink: 0; width: 24px; height: 24px; background: var(--accent-light); color: var(--accent); border: 1px solid #C3D9F5; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: var(--font-mono); font-size: 11px; font-weight: 600; margin-top: 1px; }

  .risk-legend { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-mid); padding: 8px 12px; background: var(--surface-2); border-radius: 8px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

  footer { text-align: center; padding: 32px 0 16px; color: var(--text-light); font-size: 12px; border-top: 1px solid var(--border); margin-top: 40px; }

  @media (max-width: 640px) {
    .form-grid { grid-template-columns: 1fr; }
    .stats-row { grid-template-columns: 1fr 1fr; }
    .risk-header { flex-direction: column; align-items: flex-start; }
    .risk-legend { grid-template-columns: 1fr; }
    .topbar { padding: 0 16px; }
    .main { padding: 24px 16px 40px; }
  }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-logo">
    <div class="topbar-icon">🛰</div>
    <div class="topbar-name">Route<span>Risk</span></div>
  </div>
  <div class="topbar-tag">LIVE INTELLIGENCE</div>
</div>

<div class="main">

  <div class="hero">
    <h1>Shipment Delay Predictor</h1>
    <p>Enter your shipment details and get a real-time risk assessment based on live intelligence — current disruptions, carrier performance, port conditions, and geopolitical events.</p>
  </div>

  <!-- How it works -->
  <button class="help-toggle" id="helpToggle" onclick="toggleHelp()">
    <span>📡</span> How It Works <span class="arrow">▾</span>
  </button>

  <div class="help-panel" id="helpPanel">
    <div class="help-section">
      <h3>What RouteRisk Does</h3>
      <p>RouteRisk combines your shipment details with live web intelligence to predict the likelihood of delays on your specific route. It searches for current disruptions, carrier issues, port congestion, weather events, and geopolitical risks — then produces a risk score and actionable recommendations.</p>
    </div>
    <div class="help-section">
      <h3>How to Use It</h3>
      <ol class="help-steps">
        <li>Select your transportation mode — Air or Ocean.</li>
        <li>Fill in the shipment details: origin, destination, carrier, ship date, and commodity.</li>
        <li>Click <strong>Assess Risk</strong>. The system will search for live intelligence and generate your assessment in 20–40 seconds.</li>
        <li>Review your risk score, contributing factors, and recommendations. Act on any immediate priority items before your shipment departs.</li>
      </ol>
    </div>
    <div class="help-section">
      <h3>Risk Levels</h3>
      <div class="risk-legend">
        <div class="legend-item"><div class="legend-dot" style="background:#0A7A4B;"></div><strong>Low (0–25)</strong> — Normal conditions, minimal delay risk</div>
        <div class="legend-item"><div class="legend-dot" style="background:#D97706;"></div><strong>Moderate (26–50)</strong> — Some risk factors present, monitor closely</div>
        <div class="legend-item"><div class="legend-dot" style="background:#C2410C;"></div><strong>High (51–75)</strong> — Significant disruptions likely, take action</div>
        <div class="legend-item"><div class="legend-dot" style="background:#B91C1C;"></div><strong>Critical (76–100)</strong> — Severe disruptions, immediate action required</div>
      </div>
    </div>
    <div class="help-section">
      <h3>Important Notes</h3>
      <p>RouteRisk provides risk intelligence to support decision-making — it does not guarantee shipment outcomes. Assessments are based on publicly available information at the time of analysis. Always verify critical information with your carrier or freight forwarder before making final decisions. Documents and shipment data are not stored on our servers.</p>
    </div>
  </div>

  <!-- Form -->
  <div class="form-card" id="formSection">
    <div class="form-card-title">📋 Shipment Details</div>

    <div style="margin-bottom:20px;">
      <label style="display:block;margin-bottom:10px;">Transportation Mode</label>
      <div class="mode-selector">
        <div class="mode-option">
          <input type="radio" name="mode" id="modeAir" value="Air Freight" checked>
          <label class="mode-label" for="modeAir">
            <span class="mode-icon">✈️</span>
            <div>
              <div class="mode-text">Air Freight</div>
              <div class="mode-sub">Airports · Airlines</div>
            </div>
          </label>
        </div>
        <div class="mode-option">
          <input type="radio" name="mode" id="modeOcean" value="Ocean Freight">
          <label class="mode-label" for="modeOcean">
            <span class="mode-icon">🚢</span>
            <div>
              <div class="mode-text">Ocean Freight</div>
              <div class="mode-sub">Ports · Shipping Lines</div>
            </div>
          </label>
        </div>
      </div>
    </div>

    <div class="form-grid">
      <div class="form-group">
        <label for="origin">Origin</label>
        <input type="text" id="origin" placeholder="e.g. Shanghai, China or PVG" required>
      </div>
      <div class="form-group">
        <label for="destination">Destination</label>
        <input type="text" id="destination" placeholder="e.g. Houston, TX or IAH" required>
      </div>
      <div class="form-group">
        <label for="carrier">Carrier</label>
        <input type="text" id="carrier" placeholder="e.g. Cathay Pacific or COSCO" required>
      </div>
      <div class="form-group">
        <label for="shipDate">Expected Ship Date</label>
        <input type="date" id="shipDate" required>
      </div>
      <div class="form-group full">
        <label for="commodity">Commodity / Cargo Description</label>
        <input type="text" id="commodity" placeholder="e.g. Industrial parts, electronics, textiles" required>
      </div>
    </div>

    <button class="btn-analyze" id="assessBtn" onclick="runAssessment()">
      🛰 Assess Risk
    </button>
  </div>

  <!-- Loading -->
  <div class="loading" id="loading">
    <div class="loading-route" id="loadingRoute">
      <div class="route-node" id="loadOrigin">—</div>
      <div class="route-line"></div>
      <div class="route-plane" id="loadPlane">✈️</div>
      <div class="route-line"></div>
      <div class="route-node" id="loadDest">—</div>
    </div>
    <div class="loading-steps">
      <div class="loading-step" id="ls1"><span class="step-icon">🔍</span>Generating intelligence queries...</div>
      <div class="loading-step" id="ls2"><span class="step-icon">🌐</span>Searching for route disruptions...</div>
      <div class="loading-step" id="ls3"><span class="step-icon">🚢</span>Checking carrier & port conditions...</div>
      <div class="loading-step" id="ls4"><span class="step-icon">🧠</span>Analyzing risk factors...</div>
      <div class="loading-step" id="ls5"><span class="step-icon">📊</span>Generating risk assessment...</div>
    </div>
  </div>

  <!-- Results -->
  <div id="results">

    <div class="route-summary" id="routeSummary"></div>

    <div class="risk-header">
      <div class="risk-gauge" id="riskGauge">
        <div class="risk-gauge-inner">
          <div class="risk-score" id="riskScore">—</div>
          <div class="risk-score-label">/ 100</div>
        </div>
      </div>
      <div class="risk-info">
        <div class="risk-level-badge" id="riskBadge"></div>
        <div class="risk-headline" id="riskHeadline"></div>
        <div class="risk-delay" id="riskDelay"></div>
      </div>
    </div>



    <div class="section-header">⚠ Risk Factors</div>
    <div id="riskFactors"></div>



    <div class="confidence-card" id="confidenceCard"></div>

    <button class="btn-reset" onclick="resetForm()">← Assess Another Shipment</button>
  </div>

</div>

<footer>RouteRisk &nbsp;·&nbsp; Powered by Claude AI + Live Web Intelligence &nbsp;·&nbsp; For risk guidance only</footer>

<script>
  // Set default ship date to today
  document.getElementById('shipDate').valueAsDate = new Date();

  function toggleHelp() {
    const panel = document.getElementById('helpPanel');
    const toggle = document.getElementById('helpToggle');
    panel.classList.toggle('visible');
    toggle.classList.toggle('open');
  }

  async function runAssessment() {
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const origin = document.getElementById('origin').value.trim();
    const destination = document.getElementById('destination').value.trim();
    const carrier = document.getElementById('carrier').value.trim();
    const shipDate = document.getElementById('shipDate').value;
    const commodity = document.getElementById('commodity').value.trim();

    if (!origin || !destination || !carrier || !shipDate || !commodity) {
      alert('Please fill in all fields.');
      return;
    }

    // Update loading display
    document.getElementById('loadOrigin').textContent = origin.split(',')[0].substring(0, 8);
    document.getElementById('loadDest').textContent = destination.split(',')[0].substring(0, 8);
    document.getElementById('loadPlane').textContent = mode.includes('Air') ? '✈️' : '🚢';

    document.getElementById('formSection').style.display = 'none';
    document.getElementById('loading').classList.add('visible');
    document.getElementById('results').classList.remove('visible');

    // Animate loading steps
    const steps = ['ls1','ls2','ls3','ls4','ls5'];
    steps.forEach((id, i) => setTimeout(() => {
      document.getElementById(id)?.classList.add('active');
      if (i > 0) document.getElementById(steps[i-1])?.classList.remove('active');
    }, i * 2000));

    try {
      const res = await fetch('/assess', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode, origin, destination, carrier, ship_date: shipDate, commodity})
      });
      const data = await res.json();
      document.getElementById('loading').classList.remove('visible');
      if (data.error) { alert('Error: ' + data.error); resetForm(); }
      else renderResults(data, {mode, origin, destination, carrier, shipDate, commodity});
    } catch (err) {
      alert('Something went wrong. Please try again.');
      resetForm();
    }
  }

  function renderResults(data, shipment) {
    const r = data.assessment;
    const level = r.risk_level || 'LOW';

    // Route summary bar
    const modeIcon = shipment.mode.includes('Air') ? '✈️' : '🚢';
    const modeLabel = shipment.mode.includes('Air') ? 'AIR' : 'OCEAN';
    document.getElementById('routeSummary').innerHTML = `
      <span class="route-mode">${modeIcon} ${modeLabel}</span>
      <span class="route-detail">
        <span class="route-node-text">${shipment.origin}</span>
        <span class="route-arrow">→</span>
        <span class="route-node-text">${shipment.destination}</span>
      </span>
      <span class="route-sep">·</span>
      <span class="route-carrier">${shipment.carrier}</span>
      <span class="route-sep">·</span>
      <span class="route-date">${shipment.shipDate}</span>
    `;

    // Risk gauge
    const gauge = document.getElementById('riskGauge');
    gauge.className = 'risk-gauge ' + level;
    document.getElementById('riskScore').textContent = r.risk_score || 0;

    // Badge
    const badge = document.getElementById('riskBadge');
    badge.className = 'risk-level-badge badge-' + level;
    badge.textContent = level.replace('_', ' ');

    document.getElementById('riskHeadline').textContent = r.headline || '';

    // Delay estimate
    const delay = r.estimated_delay_days || {};
    document.getElementById('riskDelay').innerHTML = `
      <div class="delay-chip primary risk-${level}">Expected delay: <span class="delay-val">${delay.most_likely ?? 0} days</span></div>
      <div class="delay-chip">Worst case: <span class="delay-val">${delay.max ?? 0} days</span></div>
    `;

    const factors = r.risk_factors || [];

    // Risk factors
    const catIcons = {WEATHER:'🌧',PORT_CONGESTION:'⚓',CARRIER:'✈',GEOPOLITICAL:'🌍',ROUTE:'🗺',CUSTOMS:'📋',CAPACITY:'📦',SEASONAL:'📅'};
    document.getElementById('riskFactors').innerHTML = factors.map(f => `
      <div class="factor-card">
        <div class="factor-severity ${f.severity}"></div>
        <div>
          <div class="factor-cat">${catIcons[f.category]||'⚠'} ${f.category?.replace('_',' ')}</div>
          <div class="factor-title">${f.title}</div>
          <div class="factor-desc">${f.description}</div>
        </div>
      </div>`).join('');



    // Confidence
    document.getElementById('confidenceCard').innerHTML = `
      <span style="font-size:18px;">🔬</span>
      <div><strong style="color:var(--navy);">Assessment Confidence: ${r.confidence}</strong> — ${r.confidence_reason}</div>`;

    document.getElementById('results').classList.add('visible');
    document.getElementById('results').scrollIntoView({behavior:'smooth'});
  }

  function resetForm() {
    document.getElementById('formSection').style.display = 'block';
    document.getElementById('loading').classList.remove('visible');
    document.getElementById('results').classList.remove('visible');
    document.querySelectorAll('.loading-step').forEach(s => s.classList.remove('active','done'));
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/assess", methods=["POST"])
def assess():
    data = request.get_json()
    required = ["mode", "origin", "destination", "carrier", "ship_date", "commodity"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Missing field: {field}"}), 400

    shipment = {k: data[k] for k in required}

    try:
        # Step 1: Generate targeted search queries
        queries = generate_search_queries(shipment)
        if not queries:
            today_str = datetime.now().strftime("%B %Y")
            queries = [
                f"{shipment['mode']} disruptions {shipment['origin']} {shipment['destination']} {today_str}",
                f"{shipment['carrier']} delays issues {today_str}",
                f"port congestion {shipment['destination']} latest {today_str}",
                f"shipping route risk {shipment['origin']} {shipment['destination']} current",
            ]

        # Step 2: Run web searches
        search_results = []
        for query in queries[:4]:
            results = web_search(query)
            search_results.append({"query": query, "results": results})

        # Step 3: Generate risk assessment
        assessment = assess_risk(shipment, search_results)
        return jsonify({"assessment": assessment, "queries_used": queries})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    debug = os.environ.get("FLASK_ENV") != "production"
    print(f"\nRouteRisk running at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
