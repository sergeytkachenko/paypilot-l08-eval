import argparse
import json
import os
import statistics
import time
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
    needed = {d for s in scenarios for c in s.get("conditions", {}).values()
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
    if price.get("cache_write_inside_read"):
        cached = max(cached - written, 0)
    fresh = max(total_in - cached - written, 0)
    read_factor = price.get("cache_read_factor", prices["cache_read_factor"])
    write_factor = price.get("cache_write_factor", prices["cache_write_factor"])
    usd = (fresh * price["input"] + cached * price["input"] * read_factor
           + written * price["input"] * write_factor
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
        "provider_cost_usd": res["usage"].get("provider_cost_usd"),
        "prompts": [c["attributes"].get("gen_ai.prompt", "") for c in llm],
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


AGGREGATE_KINDS = {"first_call_input_tokens_median_max"}


def check_aggregate(assertion, results):
    kind = assertion["kind"]
    if kind == "first_call_input_tokens_median_max":
        step = assertion["step"]
        values = [r["steps"][step - 1]["first_call_input"] for r in results]
        median = statistics.median(values)
        return median <= assertion["max"], \
            f"step {step}: median first-call input of {len(values)} run(s) is {median:g} (runs {sorted(values)}), max {assertion['max']}"
    raise RunError(f"unknown aggregate assertion kind {kind!r}")


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
            f"{'ms':>7} {'$ trace':>9} {'$ usage':>9} {'$ provider':>10} replay")
    lines = [head]
    for r in rows:
        lines.append(f"  {r['step']:>4} {r['model_calls']:>5} {r['retries']:>5} {r['identical_max']:>4} {r['first_call_input']:>6} "
                     f"{r['input_tokens']:>7} {r['cached_tokens']:>7} {r['output_tokens']:>5} {r['duration_ms']:>7} "
                     f"{_money(r['trace_cost_usd']):>9} {_money(r['usage_cost_usd']):>9} "
                     f"{_money(r['provider_cost_usd']):>10} {r['replay']}")
        if r["summary"] is not None:
            lines.append(f"       agent.summarize: {r['summary'][:200]!r}")
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    ms = sum(r["duration_ms"] for r in rows)
    usd = round(sum(r["trace_cost_usd"] or 0 for r in rows), 6)
    usage = round(sum(r["usage_cost_usd"] or 0 for r in rows), 6)
    provider = round(sum(r["provider_cost_usd"] or 0 for r in rows), 6)
    lines.append(f"  total: {tin} in, {tout} out, {ms} ms, ${usd} from trace spans, "
                 f"${usage} from /chat usage, ${provider} from the provider")
    return "\n".join(lines)


def _money(value):
    return value if value is not None else "-"


def _message_text(content):
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content or ""


def format_prompt(raw):
    if not raw:
        return "  (empty: the stand runs with TRACE_CAPTURE_CONTENT=0)"
    data = json.loads(raw)
    lines = []
    for message in data.get("messages", []):
        role = message.get("role", "?")
        text = _message_text(message.get("content"))
        lines.append(f"  [{role}] {text}" if text else f"  [{role}]")
        for call in message.get("tool_calls") or []:
            lines.append(f"      tool_call {call['function']['name']}({call['function']['arguments']})")
    tools = [t["function"]["name"] for t in data.get("tools", [])]
    lines.append(f"  tools: {', '.join(tools) if tools else '-'}")
    return "\n".join(lines)


def show_prompt(rows, step):
    if not 1 <= step <= len(rows):
        raise RunError(f"--show-prompt {step}: the dialog has {len(rows)} steps")
    prompts = rows[step - 1]["prompts"]
    for number, raw in enumerate(prompts, 1):
        print(f"  -- step {step}, model call {number} of {len(prompts)}: gen_ai.prompt")
        print(format_prompt(raw))


def run_with_retry(scenario, settings, prices, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            return run_once(scenario, settings, prices)
        except RunError as e:
            if attempt == attempts or "-> 500" not in str(e):
                raise
            print(f"    stand answered 500 (often a provider rate limit), retry {attempt}/{attempts - 1}")
            time.sleep(5 * attempt)


def with_overrides(settings, model, reasoning):
    out = dict(settings)
    if model:
        out["model"] = model
    if reasoning:
        out["reasoning"] = reasoning
    return out


def run(scenario, condition, settings, runs, prices, show_table, prompt_step=None):
    results = []
    per_run = [a for a in scenario["assertions"] if a["kind"] not in AGGREGATE_KINDS]
    for n in range(1, runs + 1):
        rows = run_with_retry(scenario, settings, prices)
        verdicts = [(a, *check(a, rows)) for a in per_run]
        passed = all(v[1] for v in verdicts)
        results.append({"run": n, "passed": passed, "steps": rows,
                        "verdicts": [{"kind": a["kind"], "ok": ok, "detail": d} for a, ok, d in verdicts]})
        verdict = ("PASS" if passed else "FAIL") if per_run else ""
        print(f"{scenario['id']} {condition:<8} run {n} {verdict}".rstrip())
        if show_table:
            print(table(rows))
        if prompt_step is not None and n == 1:
            show_prompt(rows, prompt_step)
        for a, ok, d in verdicts:
            print(f"    {'ok ' if ok else 'BAD'} {a['kind']}: {d}")
    aggregates = []
    for a in scenario["assertions"]:
        if a["kind"] in AGGREGATE_KINDS:
            ok, d = check_aggregate(a, results)
            aggregates.append({"kind": a["kind"], "ok": ok, "detail": d})
            print(f"  {'ok ' if ok else 'BAD'} {a['kind']}: {d}")
    return results, aggregates


def strip_prompts(results):
    for result in results:
        for row in result["steps"]:
            row.pop("prompts", None)
    return results


def _median(results, key):
    totals = [sum(s[key] or 0 for s in r["steps"]) for r in results]
    return statistics.median(totals)


def _spread(results, key):
    totals = [sum(s[key] or 0 for s in r["steps"]) for r in results]
    return min(totals), max(totals)


def run_matrix(matrix, scenarios, args, prices):
    by_id = {s["id"]: s for s in scenarios}
    runs = args.runs or matrix.get("runs", 5)
    cells = []
    for model in (matrix["from"], matrix["to"]):
        for sid in matrix["scenarios"]:
            scenario = by_id[sid]
            for condition, settings in scenario["conditions"].items():
                if args.condition not in ("all", condition):
                    continue
                effective = with_overrides(settings, model, args.reasoning)
                print(f"\n-- {model} · {sid} · {condition}")
                results, aggregates = run(scenario, condition, effective, runs, prices, False)
                low, high = _spread(results, "trace_cost_usd")
                verdict = ("PASS" if all(a["ok"] for a in aggregates) else "FAIL") if aggregates \
                    else f"{sum(r['passed'] for r in results)}/{len(results)}"
                cells.append({
                    "model": model, "scenario": sid, "condition": condition, "settings": effective,
                    "passed": sum(r["passed"] for r in results), "runs": len(results), "verdict": verdict,
                    "aggregates": aggregates,
                    "cost_usd_median": round(_median(results, "trace_cost_usd"), 6),
                    "cost_usd_min": round(low, 6), "cost_usd_max": round(high, 6),
                    "provider_usd_median": round(_median(results, "provider_cost_usd"), 6),
                    "time_ms_median": _median(results, "duration_ms"),
                    "input_tokens_median": _median(results, "input_tokens"),
                    "output_tokens_median": _median(results, "output_tokens"),
                    "results": strip_prompts(results)})
    print("\n== B4 · міграція моделі")
    print(f"  {'model':<30} {'scenario':<8} {'condition':<9} {'$ dialog (median, min-max)':<32} "
          f"{'$ provider':>10} {'time s':>7} {'verdicts':>8} {'in / out tokens*':>18}")
    for c in cells:
        dialog = f"{c['cost_usd_median']} ({c['cost_usd_min']}-{c['cost_usd_max']})"
        print(f"  {c['model']:<30} {c['scenario']:<8} {c['condition']:<9} {dialog:<32} "
              f"{c['provider_usd_median']:>10} {c['time_ms_median'] / 1000:>7.1f} "
              f"{c['verdict']:>8} "
              f"{int(c['input_tokens_median'])} / {int(c['output_tokens_median'])}")
    print("  * tokenizers differ: tokens of two models are not comparable one to one")
    return cells


def main():
    parser = argparse.ArgumentParser(description="Run a whole dialog live and read its traces as one document.")
    parser.add_argument("--scenario", help="B1..B4; B1..B3 when omitted")
    parser.add_argument("--condition", default="all", help="baseline | defect | all")
    parser.add_argument("--runs", type=int, help="runs per condition; default 1, or the scenario's runs with --gate; B4 runs 5")
    parser.add_argument("--gate", help="daily | release: only scenarios with this gate")
    parser.add_argument("--table", action="store_true", help="print the step-by-step trace table of every run")
    parser.add_argument("--model", help="model for every /chat request, from the stand's profiles/models.yaml")
    parser.add_argument("--reasoning", choices=["off", "low", "medium", "high"],
                        help="reasoning level for every /chat request (OpenRouter)")
    parser.add_argument("--show-prompt", type=int, metavar="STEP",
                        help="print the prompt of every model call of this step (first run of each condition)")
    parser.add_argument("--json", help="write raw results to this file")
    args = parser.parse_args()

    everything = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "scenarios").glob("*.json"))]
    scenarios = everything
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"].lower() == args.scenario.lower()]
        if not scenarios:
            raise SystemExit(f"no scenario {args.scenario}")
    else:
        scenarios = [s for s in scenarios if "matrix" not in s]
    if args.gate:
        scenarios = [s for s in scenarios if s.get("gate") == args.gate]
        if not scenarios:
            raise SystemExit(f"no scenarios with gate {args.gate}")
    if args.model and any("matrix" in s for s in scenarios):
        raise SystemExit("B4 sets the models itself: drop --model")
    prices = load_prices()
    try:
        health = doctor(scenarios)
        print(f"stand {STAND}: provider {health.get('provider')}")
        report = []
        for scenario in scenarios:
            print(f"\n== {scenario['id']} · {scenario['title']}")
            if "matrix" in scenario:
                cells = run_matrix(scenario["matrix"], everything, args, prices)
                report.append({"scenario": scenario["id"], "cells": cells})
                continue
            medians = {}
            for condition, settings in scenario["conditions"].items():
                if args.condition not in ("all", condition):
                    continue
                runs = args.runs or (scenario.get("runs", 1) if args.gate else 1)
                effective = with_overrides(settings, args.model, args.reasoning)
                results, aggregates = run(scenario, condition, effective, runs, prices, args.table, args.show_prompt)
                medians[condition] = (_median(results, "input_tokens"), _median(results, "duration_ms"),
                                      _median(results, "trace_cost_usd"))
                if aggregates:
                    outcome = "PASS" if all(a["ok"] for a in aggregates) else "FAIL"
                    print(f"   {condition}: {outcome} on the median of {len(results)} run(s); "
                          f"median {medians[condition][0]} input tokens, {medians[condition][1]} ms, "
                          f"${round(medians[condition][2], 6)}")
                else:
                    print(f"   {condition}: {sum(r['passed'] for r in results)}/{len(results)} passed; "
                          f"median {medians[condition][0]} input tokens, {medians[condition][1]} ms, "
                          f"${round(medians[condition][2], 6)}")
                report.append({"scenario": scenario["id"], "condition": condition,
                               "settings": effective, "aggregates": aggregates, "runs": strip_prompts(results)})
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
