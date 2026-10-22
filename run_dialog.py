import argparse
import json
import os
import statistics
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
STAND = os.environ.get("SERVICE_URL", "http://localhost:8000")
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class RunError(RuntimeError):
    pass


def _call(method, path, body=None, headers=None, timeout=300):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(STAND + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with _opener.open(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RunError(f"{method} {STAND}{path} -> {e.code}: {e.read().decode('utf-8')[:300]}") from e
    except urllib.error.URLError as e:
        raise RunError(f"cannot reach {STAND} ({e.reason})") from e


def doctor(scenarios):
    health = _call("GET", "/health")
    known = set(_call("GET", "/api/_test/defects").get("all_defects", []))
    needed = {d for s in scenarios for c in s["conditions"].values()
              for d in (c.get("defects") or "").split(",") if d}
    missing = sorted(needed - known)
    if missing:
        raise RunError(f"the stand does not know {missing}: update paypilot-stand to the commit named in README")
    if health.get("provider") in (None, "mock"):
        raise RunError("the stand runs LLM_PROVIDER=mock: token and cost numbers need a real model")
    return health


def load_prices():
    return json.loads((ROOT / "prices.json").read_text(encoding="utf-8"))


def _llm_spans(tree):
    return [c for c in tree.get("children", []) if c["name"] == "llm.call"]


def _tool_spans(tree):
    return [c for c in tree.get("children", []) if c["name"].startswith("tool.")]


def _summary(tree):
    found = [c for c in tree.get("children", []) if c["name"] == "agent.summarize"]
    return found[0]["attributes"].get("summary.text", "") if found else None


def span_cost(attrs, prices):
    price = prices["models"].get(attrs.get("gen_ai.request.model", ""))
    if price is None:
        return None
    total_in = attrs.get("gen_ai.usage.input_tokens", 0)
    cached = attrs.get("gen_ai.usage.cache_read_input_tokens", 0)
    written = attrs.get("gen_ai.usage.cache_creation_input_tokens", 0)
    fresh = max(total_in - cached - written, 0)
    usd = (fresh * price["input"] + cached * price["input"] * prices["cache_read_factor"]
           + written * price["input"] * prices["cache_write_factor"]
           + attrs.get("gen_ai.usage.output_tokens", 0) * price["output"]) / 1_000_000
    return usd


def step_row(step, res, tree, prices):
    llm = _llm_spans(tree)
    costs = [span_cost(c["attributes"], prices) for c in llm]
    calls = {}
    for t in _tool_spans(tree):
        key = (t["attributes"]["tool.name"], json.dumps(t["attributes"].get("tool.arguments"), sort_keys=True))
        calls[key] = calls.get(key, 0) + 1
    return {
        "step": step,
        "model_calls": len(llm),
        "retries": sum(1 for c in llm if "retry.attempt" in c["attributes"]),
        "tools": [t["attributes"]["tool.name"] for t in _tool_spans(tree)],
        "identical_max": max(calls.values()) if calls else 0,
        "first_call_input": llm[0]["attributes"].get("gen_ai.usage.input_tokens", 0) if llm else 0,
        "input_tokens": sum(c["attributes"].get("gen_ai.usage.input_tokens", 0) for c in llm),
        "cached_tokens": sum(c["attributes"].get("gen_ai.usage.cache_read_input_tokens", 0) for c in llm),
        "output_tokens": sum(c["attributes"].get("gen_ai.usage.output_tokens", 0) for c in llm),
        "duration_ms": round(tree.get("duration_ms") or 0),
        "trace_cost_usd": None if None in costs else round(sum(costs), 6),
        "usage_cost_usd": res["usage"].get("cost_usd"),
        "replay": tree["attributes"].get("context.replay_active"),
        "summary": _summary(tree),
        "tool_results": [json.dumps(t["attributes"].get("tool.result"), ensure_ascii=False) for t in _tool_spans(tree)],
        "answer": res["answer"],
    }


def run_once(scenario, settings, prices):
    _call("POST", "/api/_test/reset", {})
    headers = {"X-Stand-Settings": json.dumps(settings)} if settings else {}
    session = None
    rows = []
    for step, turn in enumerate(scenario["turns"], 1):
        res = _call("POST", "/chat", {"message": turn["user"], "session_id": session,
                                      "customer_id": scenario["customer_id"]}, headers)
        session = res["session_id"]
        tree = _call("GET", f"/api/_test/traces/{res['request_id']}")
        rows.append(step_row(step, res, tree, prices))
    return rows


def _steps(assertion, rows):
    step = assertion.get("step", "all")
    return rows if step == "all" else [rows[step - 1]]


def check(assertion, rows):
    kind = assertion["kind"]
    if kind == "tool_called":
        hit = [r for r in _steps(assertion, rows) if assertion["tool"] in r["tools"]]
        return bool(hit), f"{assertion['tool']} called on step(s) {[r['step'] for r in hit] or 'none'}"
    if kind == "identical_tool_calls_max":
        worst = max(_steps(assertion, rows), key=lambda r: r["identical_max"])
        return worst["identical_max"] <= assertion["max"], \
            f"step {worst['step']}: {worst['identical_max']} identical call(s), max {assertion['max']}"
    if kind == "first_call_input_tokens_max":
        row = _steps(assertion, rows)[0]
        return row["first_call_input"] <= assertion["max"], \
            f"step {row['step']}: first model call reads {row['first_call_input']} input tokens, max {assertion['max']}"
    if kind == "dialog_tokens_max":
        total = sum(r["input_tokens"] + r["output_tokens"] for r in rows)
        return total <= assertion["max"], f"dialog: {total} tokens, max {assertion['max']}"
    if kind == "dialog_latency_ms_max":
        total = sum(r["duration_ms"] for r in rows)
        return total <= assertion["max"], f"dialog: {total} ms, max {assertion['max']}"
    if kind == "dialog_cost_usd_max":
        total = round(sum(r["trace_cost_usd"] or 0 for r in rows), 6)
        return total <= assertion["max"], f"dialog: ${total} from trace spans, max ${assertion['max']}"
    if kind == "summary_contains":
        summaries = [r for r in rows if r["summary"] is not None]
        if not summaries:
            return False, "no agent.summarize span in the dialog"
        text = summaries[0]["summary"]
        missing = [v for v in assertion["values"] if v.lower() not in text.lower()]
        return not missing, f"step {summaries[0]['step']} summary misses {missing}" if missing \
            else f"step {summaries[0]['step']} summary keeps {assertion['values']}"
    raise RunError(f"unknown assertion kind {kind!r}")


def table(rows):
    head = (f"  {'step':>4} {'calls':>5} {'retry':>5} {'same':>4} {'1st in':>6} {'in':>7} {'cached':>7} {'out':>5} "
            f"{'ms':>7} {'$ trace':>9} {'$ usage':>9} replay")
    lines = [head]
    for r in rows:
        lines.append(f"  {r['step']:>4} {r['model_calls']:>5} {r['retries']:>5} {r['identical_max']:>4} {r['first_call_input']:>6} "
                     f"{r['input_tokens']:>7} {r['cached_tokens']:>7} {r['output_tokens']:>5} {r['duration_ms']:>7} "
                     f"{r['trace_cost_usd'] if r['trace_cost_usd'] is not None else '-':>9} "
                     f"{r['usage_cost_usd'] if r['usage_cost_usd'] is not None else '-':>9} {r['replay']}")
        if r["summary"] is not None:
            lines.append(f"       agent.summarize: {r['summary'][:200]!r}")
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    ms = sum(r["duration_ms"] for r in rows)
    usd = round(sum(r["trace_cost_usd"] or 0 for r in rows), 6)
    usage = round(sum(r["usage_cost_usd"] or 0 for r in rows), 6)
    lines.append(f"  total: {tin} in, {tout} out, {ms} ms, ${usd} from trace spans, ${usage} from /chat usage")
    return "\n".join(lines)


def run(scenario, condition, settings, runs, prices, show_table):
    results = []
    for n in range(1, runs + 1):
        rows = run_once(scenario, settings, prices)
        verdicts = [(a, *check(a, rows)) for a in scenario["assertions"]]
        passed = all(v[1] for v in verdicts)
        results.append({"run": n, "passed": passed, "steps": rows,
                        "verdicts": [{"kind": a["kind"], "ok": ok, "detail": d} for a, ok, d in verdicts]})
        print(f"{scenario['id']} {condition:<8} run {n} {'PASS' if passed else 'FAIL'}")
        if show_table:
            print(table(rows))
        for a, ok, d in verdicts:
            print(f"    {'ok ' if ok else 'BAD'} {a['kind']}: {d}")
    return results


def _median(results, key):
    totals = [sum(s[key] or 0 for s in r["steps"]) for r in results]
    return statistics.median(totals)


def main():
    parser = argparse.ArgumentParser(description="Run a whole dialog live and read its traces as one document.")
    parser.add_argument("--scenario", help="B1..B3; all scenarios when omitted")
    parser.add_argument("--condition", default="all", help="baseline | defect | all")
    parser.add_argument("--runs", type=int, help="runs per condition; default 1, or the scenario's runs with --gate")
    parser.add_argument("--gate", help="daily | release: only scenarios with this gate")
    parser.add_argument("--table", action="store_true", help="print the step-by-step trace table of every run")
    parser.add_argument("--json", help="write raw results to this file")
    args = parser.parse_args()

    scenarios = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "scenarios").glob("*.json"))]
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"].lower() == args.scenario.lower()]
        if not scenarios:
            raise SystemExit(f"no scenario {args.scenario}")
    if args.gate:
        scenarios = [s for s in scenarios if s.get("gate") == args.gate]
        if not scenarios:
            raise SystemExit(f"no scenarios with gate {args.gate}")
    prices = load_prices()
    try:
        health = doctor(scenarios)
        print(f"stand {STAND}: provider {health.get('provider')}")
        report = []
        for scenario in scenarios:
            print(f"\n== {scenario['id']} · {scenario['title']}")
            medians = {}
            for condition, settings in scenario["conditions"].items():
                if args.condition not in ("all", condition):
                    continue
                runs = args.runs or (scenario.get("runs", 1) if args.gate else 1)
                results = run(scenario, condition, settings, runs, prices, args.table)
                medians[condition] = (_median(results, "input_tokens"), _median(results, "duration_ms"),
                                      _median(results, "trace_cost_usd"))
                print(f"   {condition}: {sum(r['passed'] for r in results)}/{len(results)} passed; "
                      f"median {medians[condition][0]} input tokens, {medians[condition][1]} ms, "
                      f"${round(medians[condition][2], 6)}")
                report.append({"scenario": scenario["id"], "condition": condition,
                               "settings": settings, "runs": results})
            if "baseline" in medians and "defect" in medians:
                b, d = medians["baseline"], medians["defect"]
                print(f"   defect / baseline: tokens x{d[0] / b[0]:.2f}, time x{d[1] / b[1]:.2f}, "
                      f"cost x{d[2] / b[2]:.2f}" if all(b) else "   defect / baseline: baseline has a zero")
    except RunError as e:
        raise SystemExit(f"ERROR: {e}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
