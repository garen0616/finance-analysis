from statistics import median
from collections import defaultdict

def resolve_peers(client, symbol):
    peers = []
    try:
        peers = client.get_stock_peers(symbol)
    except Exception:
        peers = []
    if not peers:
        prof = client.get_company_profile(symbol)
        if prof.get("companyPeers"):
            peers = [p for p in prof["companyPeers"].split(",") if p]
    return peers[:8]

def compute_peer_medians(client, symbol, peers_override=None):
    peers = peers_override or resolve_peers(client, symbol)
    if symbol in peers:
        peers.remove(symbol)
    metrics = defaultdict(list)
    cleaned = []
    for peer in peers[:8]:
        symbol = peer.get("symbol") if isinstance(peer, dict) else peer
        if not symbol:
            continue
        try:
            inc = client.get_income_q(symbol, 4)
        except Exception:
            continue
        if not inc:
            continue
        latest = inc[0]
        metrics["revenue"].append(latest.get("revenue") or 0)
        metrics["eps"].append(latest.get("epsdiluted") or 0)
        metrics["grossMargin"].append(((latest.get("grossProfit") or 0) / (latest.get("revenue") or 1)) * 100)
        cleaned.append(peer)
    medians = {k: median(v) if v else 0 for k, v in metrics.items()}
    return {"peers": cleaned or peers, "medians": medians}
