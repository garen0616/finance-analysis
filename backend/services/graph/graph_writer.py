import uuid

def ensure_schema(client):
    if not client or not client.enabled:
        return
    text = open("services/graph/graph_schema.cypher").read()
    for stmt in text.split(";"):
        s = stmt.strip()
        if not s:
            continue
        client.run(s)

def write_analysis(client, symbol, analysis):
    if not client or not client.enabled:
        return
    company_id = symbol.upper()
    client.run("MERGE (c:Company {symbol:$symbol})", {"symbol": company_id})
    if analysis.get("tables", {}).get("income"):
        latest = analysis["tables"]["income"][0]
        qid = f"{company_id}-{latest.get('calendarYear')}-{latest.get('period')}"
        client.run("MERGE (q:Quarter {id:$id}) SET q.year=$year,q.period=$period", {"id": qid, "year": latest.get("calendarYear"), "period": latest.get("period")})
        client.run("MATCH (c:Company {symbol:$symbol}), (q:Quarter {id:$qid}) MERGE (c)-[:REPORTS]->(q)", {"symbol": company_id, "qid": qid})
    if analysis.get("highlights"):
        for h in analysis["highlights"][:5]:
            iid = str(uuid.uuid4())
            client.run("MERGE (i:Insight {id:$id}) SET i.text=$text", {"id": iid, "text": h[:240]})
            client.run("MATCH (c:Company {symbol:$symbol}), (i:Insight {id:$id}) MERGE (c)-[:HAS_INSIGHT]->(i)", {"symbol": company_id, "id": iid})
