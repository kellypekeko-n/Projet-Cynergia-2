"""
Kill Chain Detection for ICS/IIoT Networks
Analyse les séquences d'alertes d'inference.py pour détecter des campagnes
d'attaques multi-étapes, même si chaque alerte individuelle a une faible
confiance.

Alignement MITRE ATT&CK for ICS :
  Phase 1 - Reconnaissance : scanning   (T0840 Network Sniffing)
  Phase 2 - Initial Access  : mitm       (T0830 Adversary-in-the-Middle)
                               backdoor  (T0807 Command-Line Interface)
                               password  (T1110 Brute Force)
  Phase 3 - Exécution/Impact : ransomware (T0826 Loss of Availability)
                                dos/ddos  (T0814 Denial of Service)
                                injection (T0836 Modify Parameter)

Usage:
  python detect_kill_chain.py --alerts alerts.jsonl
  python detect_kill_chain.py --alerts alerts.jsonl --window 600 --min-confidence 0.4
  python detect_kill_chain.py --demo   # Génère des alertes synthétiques pour test
"""
import os, sys, json, argparse, re
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import STEALTHY_CLASSES, MITRE_MAP, METRICS_DIR

# ── Kill chain patterns ────────────────────────────────────────────────────────

KILL_CHAINS = [
    {
        "name":        "Full_ICS_Campaign",
        "description": "Campagne complète : Reconnaissance → Accès → Impact",
        "phases": [
            {"name": "RECON",  "classes": ["scanning"]},
            {"name": "ACCESS", "classes": ["mitm", "backdoor", "password"]},
            {"name": "IMPACT", "classes": ["ransomware", "dos", "ddos", "injection"]},
        ],
        "window_seconds": 600,
        "severity":       "CRITICAL",
        "mitre_chain":    "T0840 → T0830/T0807 → T0826/T0814",
    },
    {
        "name":        "Reconnaissance_Pivot",
        "description": "Pivot furtif : Scan → Accès latéral",
        "phases": [
            {"name": "RECON",  "classes": ["scanning"]},
            {"name": "PIVOT",  "classes": ["mitm", "backdoor", "password"]},
        ],
        "window_seconds": 300,
        "severity":       "HIGH",
        "mitre_chain":    "T0840 → T0830/T0807",
    },
    {
        "name":        "Ransomware_Deployment",
        "description": "Déploiement ransomware : Accès → Impact rapide",
        "phases": [
            {"name": "STAGE",  "classes": ["backdoor", "mitm"]},
            {"name": "IMPACT", "classes": ["ransomware"]},
        ],
        "window_seconds": 180,
        "severity":       "CRITICAL",
        "mitre_chain":    "T0807/T0830 → T0826",
    },
    {
        "name":        "DDoS_Amplification",
        "description": "DoS amplifié : Scan → DDoS coordonné",
        "phases": [
            {"name": "RECON",  "classes": ["scanning"]},
            {"name": "IMPACT", "classes": ["ddos", "dos"]},
        ],
        "window_seconds": 120,
        "severity":       "HIGH",
        "mitre_chain":    "T0840 → T0814",
    },
]

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Alert:
    timestamp: float           # unix epoch
    src_id: str                # source identifier (IP or flow prefix)
    cls: str                   # predicted class
    confidence: float
    mitre_code: str
    raw: dict = field(default_factory=dict)

    def phase_for(self, chain: dict) -> Optional[str]:
        for phase in chain["phases"]:
            if self.cls in phase["classes"]:
                return phase["name"]
        return None


@dataclass
class KillChainEvent:
    chain_name:   str
    description:  str
    severity:     str
    mitre_chain:  str
    src_id:       str
    alerts:       List[Alert]
    detected_at:  float
    window_s:     float

    def to_dict(self):
        return {
            "event_type":    "KILL_CHAIN",
            "chain_name":    self.chain_name,
            "description":   self.description,
            "severity":      self.severity,
            "mitre_chain":   self.mitre_chain,
            "src_id":        self.src_id,
            "detected_at":   datetime.fromtimestamp(self.detected_at, tz=timezone.utc).isoformat(),
            "window_s":      self.window_s,
            "n_alerts":      len(self.alerts),
            "phases_seen":   list({a.cls for a in self.alerts}),
            "min_confidence": round(min(a.confidence for a in self.alerts), 4),
            "avg_confidence": round(sum(a.confidence for a in self.alerts)/len(self.alerts), 4),
            "timeline": [
                {
                    "t":          datetime.fromtimestamp(a.timestamp, tz=timezone.utc).isoformat(),
                    "class":      a.cls,
                    "confidence": round(a.confidence, 4),
                    "mitre":      a.mitre_code,
                }
                for a in sorted(self.alerts, key=lambda x: x.timestamp)
            ],
        }


# ── Parser ────────────────────────────────────────────────────────────────────

def _ts_to_epoch(ts_str: str) -> float:
    """Parse ISO or epoch timestamp → unix float."""
    if ts_str is None:
        return datetime.now(tz=timezone.utc).timestamp()
    try:
        return float(ts_str)
    except (ValueError, TypeError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt).replace(
                tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return datetime.now(tz=timezone.utc).timestamp()


def parse_alert_line(line: str) -> Optional[Alert]:
    """Parse one line from alerts.jsonl → Alert object."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    cls = obj.get("class") or obj.get("cls") or obj.get("prediction")
    if not cls or cls == "normal":
        return None

    ts       = _ts_to_epoch(obj.get("timestamp") or obj.get("ts"))
    conf     = float(obj.get("confidence", 0.5))
    mitre    = obj.get("mitre_code", MITRE_MAP.get(cls, "N/A"))
    src_id   = (obj.get("src_ip") or obj.get("source_ip") or
                obj.get("src") or obj.get("flow_id", "unknown"))

    return Alert(timestamp=ts, src_id=str(src_id), cls=cls,
                 confidence=conf, mitre_code=mitre, raw=obj)


# ── Detector ──────────────────────────────────────────────────────────────────

class KillChainDetector:
    """
    Sliding-window state machine per source ID.
    Maintains a buffer of recent alerts per (src_id) and checks for
    phase completion of each defined kill chain pattern.
    """

    def __init__(self, min_confidence: float = 0.30,
                 max_window: int = 600, verbose: bool = True):
        self.min_confidence = min_confidence
        self.max_window     = max_window
        self.verbose        = verbose
        # src_id → list[Alert]
        self._buffers: Dict[str, List[Alert]] = defaultdict(list)
        self._events:  List[KillChainEvent]   = []
        self._seen:    set                    = set()  # dedup keys

    def ingest(self, alert: Alert) -> List[KillChainEvent]:
        """Process one alert; return any new kill chain events detected."""
        if alert.confidence < self.min_confidence:
            return []

        buf = self._buffers[alert.src_id]
        buf.append(alert)

        # Trim old alerts outside max_window
        cutoff = alert.timestamp - self.max_window
        self._buffers[alert.src_id] = [a for a in buf if a.timestamp >= cutoff]
        buf = self._buffers[alert.src_id]

        new_events = []
        for chain in KILL_CHAINS:
            window = chain["window_seconds"]
            events = self._check_chain(chain, buf, alert.timestamp, window)
            new_events.extend(events)

        self._events.extend(new_events)
        return new_events

    def _check_chain(self, chain: dict, buf: List[Alert],
                     now: float, window: float) -> List[KillChainEvent]:
        """
        Check if the buffer contains all phases of the kill chain
        within the time window, in order (phases[0] before phases[-1]).
        """
        recent = [a for a in buf if a.timestamp >= now - window]
        phases = chain["phases"]
        n_phases = len(phases)

        # For each possible starting alert (phase 0 match), try to build chain
        found = []
        for i, start in enumerate(recent):
            if start.cls not in phases[0]["classes"]:
                continue

            chain_alerts = [start]
            phase_idx    = 1

            for j in range(i + 1, len(recent)):
                a = recent[j]
                if a.timestamp > start.timestamp + window:
                    break
                if a.cls in phases[phase_idx]["classes"]:
                    chain_alerts.append(a)
                    phase_idx += 1
                    if phase_idx == n_phases:
                        break  # all phases matched

            if phase_idx == n_phases:
                # Full chain matched — dedup by (chain, src_id, minute bucket)
                bucket = int(chain_alerts[-1].timestamp // 60)
                key    = (chain["name"], chain_alerts[0].src_id, bucket)
                if key not in self._seen:
                    self._seen.add(key)
                    evt = KillChainEvent(
                        chain_name=chain["name"],
                        description=chain["description"],
                        severity=chain["severity"],
                        mitre_chain=chain["mitre_chain"],
                        src_id=chain_alerts[0].src_id,
                        alerts=chain_alerts,
                        detected_at=chain_alerts[-1].timestamp,
                        window_s=chain_alerts[-1].timestamp - chain_alerts[0].timestamp,
                    )
                    found.append(evt)
        return found

    @property
    def events(self):
        return self._events

    def summary(self):
        by_chain    = defaultdict(int)
        by_severity = defaultdict(int)
        for e in self._events:
            by_chain[e.chain_name]   += 1
            by_severity[e.severity]  += 1
        return {
            "total_events":   len(self._events),
            "by_chain":       dict(by_chain),
            "by_severity":    dict(by_severity),
            "unique_sources": len({e.src_id for e in self._events}),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

def _generate_demo_alerts(out_path: str):
    """Generate synthetic alerts.jsonl for testing."""
    import random
    random.seed(42)
    base_ts = datetime(2026, 6, 15, 8, 0, 0, tzinfo=timezone.utc).timestamp()

    scenarios = [
        # Full ICS campaign from 192.168.1.100
        [("scanning", 0.85,  0),
         ("scanning", 0.72, 15),
         ("mitm",     0.63, 45),
         ("backdoor", 0.58, 90),
         ("ransomware",0.91,180)],
        # DDoS pattern from 10.0.0.55
        [("scanning", 0.79,  0),
         ("ddos",     0.88, 60)],
        # Isolated alerts from 192.168.2.200 (no chain)
        [("dos",      0.55,  0)],
        # Reconnaissance pivot from 172.16.0.10
        [("scanning", 0.91,  0),
         ("password", 0.67, 120),
         ("backdoor", 0.72, 240)],
    ]
    ips = ["192.168.1.100", "10.0.0.55", "192.168.2.200", "172.16.0.10"]

    with open(out_path, "w") as f:
        for ip, scenario in zip(ips, scenarios):
            for cls, conf, delta in scenario:
                ts  = base_ts + delta + random.uniform(-2, 2)
                obj = {
                    "timestamp":  datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "src_ip":     ip,
                    "class":      cls,
                    "confidence": round(conf + random.uniform(-0.05, 0.05), 3),
                    "mitre_code": MITRE_MAP.get(cls, "N/A"),
                }
                f.write(json.dumps(obj) + "\n")
    print(f"  Demo alerts écrites: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    if args.demo:
        demo_path = os.path.join(METRICS_DIR, "demo_alerts.jsonl")
        _generate_demo_alerts(demo_path)
        args.alerts = demo_path

    if not args.alerts or not os.path.exists(args.alerts):
        print(f"[ERROR] Fichier d'alertes introuvable: {args.alerts}")
        print("  Utilisez --demo pour générer des alertes synthétiques.")
        sys.exit(1)

    detector = KillChainDetector(
        min_confidence=args.min_confidence,
        max_window=args.window,
        verbose=not args.quiet,
    )

    print(f"\n=== Kill Chain Detector — Cynergia ICS Framework ===")
    print(f"  Fichier     : {args.alerts}")
    print(f"  Fenêtre max : {args.window}s  |  Confiance min : {args.min_confidence}")
    print(f"  Patterns    : {len(KILL_CHAINS)} kill chains MITRE ATT&CK for ICS\n")

    n_total = n_attack = 0
    with open(args.alerts) as f:
        for line in f:
            a = parse_alert_line(line)
            if a is None:
                continue
            n_total += 1
            if a.cls != "normal":
                n_attack += 1
            events = detector.ingest(a)
            for evt in events:
                ts_str = datetime.fromtimestamp(evt.detected_at, tz=timezone.utc).isoformat()
                print(f"  [{evt.severity}] {evt.chain_name} — src={evt.src_id}")
                print(f"    Séquence : {' → '.join(a.cls for a in sorted(evt.alerts, key=lambda x: x.timestamp))}")
                print(f"    MITRE    : {evt.mitre_chain}")
                print(f"    Durée    : {evt.window_s:.0f}s  |  Détecté : {ts_str}")
                print()

    summ = detector.summary()
    print("─" * 50)
    print(f"RÉSUMÉ : {n_total} alertes lues, {n_attack} attaques")
    print(f"  Kill chains détectés : {summ['total_events']}")
    print(f"  Sources uniques      : {summ['unique_sources']}")
    if summ["by_severity"]:
        for sev, cnt in sorted(summ["by_severity"].items()):
            print(f"    {sev:<10} : {cnt}")

    if args.output:
        out_events = [e.to_dict() for e in detector.events]
        with open(args.output, "w") as f:
            for e in out_events:
                f.write(json.dumps(e) + "\n")
        print(f"\n  Kill chain events sauvegardés : {args.output}")

    if args.metrics:
        with open(args.metrics, "w") as f:
            json.dump({
                "summary":      summ,
                "patterns":     [c["name"] for c in KILL_CHAINS],
                "config": {
                    "min_confidence": args.min_confidence,
                    "window_s":       args.window,
                },
                "events": [e.to_dict() for e in detector.events],
            }, f, indent=2)
        print(f"  Métriques sauvegardées : {args.metrics}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kill Chain Detector for ICS/IIoT")
    p.add_argument("--alerts", type=str, default=None,
                   help="Fichier alerts.jsonl (sortie de inference.py)")
    p.add_argument("--demo", action="store_true",
                   help="Générer des alertes synthétiques pour démonstration")
    p.add_argument("--window", type=int, default=600,
                   help="Fenêtre temporelle max en secondes (défaut: 600)")
    p.add_argument("--min-confidence", type=float, default=0.30,
                   help="Confiance minimale pour inclure une alerte (défaut: 0.30)")
    p.add_argument("--output", type=str, default=None,
                   help="Fichier .jsonl pour sauvegarder les kill chain events")
    p.add_argument("--metrics", type=str, default=None,
                   help="Fichier JSON pour sauvegarder les métriques")
    p.add_argument("--quiet", action="store_true",
                   help="Supprimer les alertes temps réel (seulement résumé final)")
    main(p.parse_args())
