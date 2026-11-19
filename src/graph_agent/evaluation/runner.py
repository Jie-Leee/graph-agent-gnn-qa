"""Reproducible question-level evaluation; inference never receives gold labels."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, local

from graph_agent.data.dataset import load_examples, load_questions
from graph_agent.evaluation.metrics import answer_metrics, candidate_recall
from graph_agent.schemas import Prediction

VARIANTS = ("llm", "text", "gat_2_mean", "gat_1_mean", "gat_3_mean", "gat_2_sum", "gat_2_max")
PROTOCOL_VERSION = "e-evaluation-v1"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _finite(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _mode(variant):
    return variant if variant in ("llm", "text") else "gnn"


@dataclass(frozen=True)
class EvaluationOptions:
    data: Path = Path("data/processed")
    raw: Path = Path("data/raw")
    config: Path = Path("configs/experiment.json")
    env: Path = Path(".env")
    output: Path = Path("outputs/evaluation")
    split: str = "validation"
    limit: int = 5
    variants: tuple[str, ...] = ("llm", "text", "gat_2_mean")
    backend: str = "live"
    resume: bool = False
    device: str = "cpu"
    repo_root: Path | None = None
    workers: int = 1


class MockEngine:
    """A deliberately empty predictor for plumbing checks, never a model result."""
    last_trace = None
    requested_model = "MOCK_NO_MODEL"

    def preflight(self):
        pass

    def new_worker(self):
        return MockEngine()

    def identity(self):
        return {"backend": "mock", "implementation": "empty-answers-v1", "requested_model": self.requested_model}

    def run(self, question, variant):
        start = time.perf_counter()
        self.last_trace = {"run_id": uuid.uuid4().hex, "status": "ok", "events": [],
                           "execution_backend": "mock", "usage_reported": False,
                           "route_cache_hit": False, "actual_elapsed_seconds": time.perf_counter() - start,
                           "retrieved_node_ids": [], "context_candidate_ids": []}
        return Prediction(question.question_id, (), "MOCK: empty answer; validates evaluator only.")


class _LockedScorer:
    def __init__(self, scorer, lock):
        self.scorer, self.lock = scorer, lock

    def describe(self):
        return self.scorer.describe()

    def rank(self, question, graph):
        # Share the loaded weights, but serialize GNN inference across workers.
        with self.lock:
            return self.scorer.rank(question, graph)


class LiveEngine:
    """Each worker keeps one AgentRunner for all variants of a question."""
    def __init__(self, options, config):
        from graph_agent.agent.provider import LLMSettings
        self.options, self.config = options, config
        # Validates configuration without making a provider request. Never serialize api_key.
        try:
            self.llm = LLMSettings.from_env(options.env)
        except ValueError:
            raise ValueError("Live backend requires local LLM_API_KEY or DEEPSEEK_API_KEY and valid LLM configuration") from None
        self.requested_model = self.llm.model
        root = options.repo_root or Path.cwd()
        self.checkpoints = {v: root / "checkpoints" / v / "best.pt"
                            for v in options.variants if _mode(v) == "gnn"}
        self.scorers, self.runner = {}, None
        self.resources = None
        self.active_question = None
        self.last_trace = None

    def preflight(self):
        from graph_agent.agent.gnn import CheckpointScorer
        for path in (self.options.raw / "entity2text.txt",):
            if not path.is_file():
                raise ValueError(f"Missing raw name map: {path}; run scripts/download_data.py")
        if any(_mode(v) != "llm" for v in self.options.variants):
            if not (self.options.raw / "FB15K-237" / "train.txt").is_file():
                raise ValueError("Missing visible training graph; run scripts/download_data.py")
        scorer_lock = Lock()
        configured = {item["name"] for item in self.config["gnn"]["configs"]}
        for variant, path in self.checkpoints.items():
            if variant not in configured or not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing selected GNN checkpoint/config: {variant}")
            scorer = CheckpointScorer(path, device=self.options.device)
            self.scorers[variant] = _LockedScorer(scorer, scorer_lock) if self.options.workers > 1 else scorer
        if self.options.workers > 1:
            self._prepare_resources()

    def new_worker(self):
        worker = copy.copy(self)
        worker.runner = None
        worker.active_question = None
        worker.last_trace = None
        return worker

    def identity(self):
        result = {"backend": "live", "provider_backend": "LangChainClient",
                  "requested_model": self.llm.model,
                  "api_base": self.llm.api_base, "temperature": self.llm.temperature,
                  "max_output_tokens": self.llm.max_output_tokens,
                  "max_retries": self.llm.max_retries, "timeout_seconds": self.llm.timeout_seconds,
                  "device": self.options.device,
                  "checkpoints": {name: _digest(path) for name, path in self.checkpoints.items()}}
        raw = self.options.raw
        result["raw_files"] = {name: _digest(raw / name) for name in
                               ("FB15K-237/train.txt", "entity2text.txt", "relation2text.txt")
                               if (raw / name).is_file()}
        return result

    def _prepare_resources(self):
        if self.resources is not None:
            return
        from graph_agent.agent.evidence import EntityCatalog
        from graph_agent.agent.retrieval import GraphRetrievalTool
        from graph_agent.agent.runner import AgentSettings
        from graph_agent.data.loader import read_text_map
        retrieval = self.config["retrieval"]
        settings = AgentSettings(**{key: retrieval[key] for key in
                                   ("max_hops", "max_nodes", "max_edges", "context_token_budget")},
                                 max_tool_calls=self.config["agent"]["max_tool_calls"],
                                 allow_inverse=retrieval.get("allow_inverse", True))
        retriever = None
        if any(_mode(v) != "llm" for v in self.options.variants):
            retriever, catalog = GraphRetrievalTool.from_raw(self.options.raw)
        else:
            catalog = EntityCatalog(read_text_map(self.options.raw / "entity2text.txt"))
        self.resources = retriever, catalog, settings

    def _initialize(self):
        if self.runner is not None:
            return
        from graph_agent.agent.provider import LangChainClient
        from graph_agent.agent.runner import AgentRunner
        self._prepare_resources()
        retriever, catalog, settings = self.resources
        # The tool's mutable last_stats is per worker; the graph index is read-only.
        self.runner = AgentRunner(LangChainClient(self.llm), retriever=copy.copy(retriever),
                                  catalog=catalog, settings=settings, trace_dir=None)

    def run(self, question, variant):
        self._initialize()
        if question != self.active_question:
            self.runner.clear_cache()
            self.active_question = question
        self.runner.scorer = self.scorers.get(variant)
        try:
            return self.runner.run(question, _mode(variant))
        finally:
            self.last_trace = copy.deepcopy(self.runner.last_trace)


def _fingerprint(options, engine, gold_path, questions_path, selected_ids):
    root = options.repo_root or Path.cwd()
    code = {}
    for base in (root / "src", root / "scripts"):
        if base.is_dir():
            for path in sorted(base.rglob("*.py")):
                code[str(path.relative_to(root)).replace("\\", "/")] = _digest(path)
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    config = json.loads(options.config.read_text(encoding="utf-8-sig"))
    payload = {"protocol_version": PROTOCOL_VERSION, "execution_backend": options.backend,
               "split": options.split, "variants": list(options.variants),
               "selected_question_ids": list(selected_ids), "gold_sha256": _digest(gold_path),
               "questions_sha256": _digest(questions_path), "config_sha256": _digest(options.config),
               "config": config, "git_commit": commit, "source_sha256": code,
               "engine": engine.identity()}
    manifest = gold_path.parent / "manifest.json"
    if manifest.is_file():
        payload["dataset_manifest_sha256"] = _digest(manifest)
    return hashlib.sha256(_canonical(payload).encode()).hexdigest(), payload


def _usage(trace, prediction):
    logical_known = trace.get("usage_reported") is True and prediction is not None
    # Copied shared_route_events are logical costs. Only events executed in this
    # call establish actual usage; an unknown cached route must not hide a known
    # current answer completion, or turn unknown usage into zero.
    calls = [event for event in trace.get("events", [])
             if event.get("stage") in ("route", "answer") or "usage_reported" in event]
    actual_known = bool(calls) and all(event.get("usage_reported") is True for event in calls)
    return {"usage_reported": logical_known, "logical_usage_reported": logical_known,
            "actual_usage_reported": actual_known,
            "logical_input_tokens": _finite(getattr(prediction, "input_tokens", None)) if logical_known else None,
            "logical_output_tokens": _finite(getattr(prediction, "output_tokens", None)) if logical_known else None,
            "actual_input_tokens": _finite(trace.get("actual_input_tokens")) if actual_known else None,
            "actual_output_tokens": _finite(trace.get("actual_output_tokens")) if actual_known else None}


def _row(example, prediction, trace, variant, options, fingerprint, group_id, error_type, wall_time):
    success = error_type is None
    predicted = tuple(dict.fromkeys(prediction.answer_ids)) if success else ()
    gold = tuple(example.answer_ids)
    metrics = answer_metrics(predicted, gold) if success else dict(exact_match=0.0, precision=0.0, recall=0.0, f1=0.0)
    mode = _mode(variant)
    served = sorted({event["served_model"] for event in trace.get("events", [])
                     if isinstance(event.get("served_model"), str)})
    resolution = trace.get("resolution") or {}
    row = {"experiment_fingerprint": fingerprint, "question_id": example.question.question_id,
           "question_text": example.question.text, "topic_entity_ids": list(example.question.topic_entity_ids),
           "group_id": group_id, "mode": mode, "variant": variant, "split": options.split,
           "execution_backend": options.backend, "gold_answer_ids": list(gold),
           "predicted_answer_ids": list(predicted),
           "answer_text": prediction.answer_text if success else "",
           **metrics, "candidate_recall": candidate_recall(trace.get("retrieved_node_ids", []), gold) if mode != "llm" else None,
           "context_answer_recall": candidate_recall(trace.get("context_candidate_ids", []), gold) if mode != "llm" else None,
           "graph_unsupported_answer_fraction": len(set(predicted) - set(gold)) / len(predicted) if predicted else None,
           "answer_rate": float(bool(predicted)), "status": "ok" if success else "error", "error_type": error_type,
           "tool_calls": getattr(prediction, "tool_calls", None) if success else None,
           "logical_elapsed_seconds": _finite(getattr(prediction, "elapsed_seconds", None)) if success else None,
           "actual_elapsed_seconds": _finite(trace.get("actual_elapsed_seconds")) if _finite(trace.get("actual_elapsed_seconds")) is not None else wall_time,
           **_usage(trace, prediction), "route_cache_hit": trace.get("route_cache_hit", False),
           "served_models": served, "cited_evidence": [[t.head, t.relation, t.tail] for t in prediction.cited_evidence] if success else [],
           "unresolved_answer_names": resolution.get("unresolved_names", []),
           "ambiguous_answer_names": resolution.get("ambiguous_names", {}),
           "rejected_answer_ids": resolution.get("rejected_ids", []),
           "model_attempted_answer": bool(resolution.get("raw_answer_ids") or resolution.get("raw_answer_names") or predicted),
           "run_id": trace.get("run_id"), "trace_file": None}
    return row


def _aggregate(rows, examples, options, fingerprint, dataset_question_count):
    expected = len(examples)
    groups = {}
    for example in examples:
        q = example.question
        gid = hashlib.sha256(_canonical([list(q.topic_entity_ids), list(example.relation_path)]).encode()).hexdigest()[:20]
        groups.setdefault(gid, []).append(q.question_id)
    result = {}
    for variant in options.variants:
        selected = [r for r in rows if r["variant"] == variant]
        by_id = {r["question_id"]: r for r in selected}
        item = {"variant": variant, "mode": _mode(variant), "execution_backend": options.backend,
                "question_count": expected, "record_count": len(selected),
                "success_count": sum(r["status"] == "ok" for r in selected),
                "failure_count": sum(r["status"] == "error" for r in selected),
                "pending_count": expected - len(selected), "query_group_count": len(groups),
                "usage_known_count": sum(r["usage_reported"] for r in selected),
                "usage_unknown_count": sum(not r["usage_reported"] for r in selected),
                "actual_usage_known_count": sum(r["actual_usage_reported"] for r in selected),
                "actual_usage_unknown_count": sum(not r["actual_usage_reported"] for r in selected)}
        for field in ("exact_match", "precision", "recall", "f1", "answer_rate"):
            item[field] = sum(r[field] for r in selected) / expected
        for field in ("candidate_recall", "context_answer_recall", "graph_unsupported_answer_fraction",
                      "logical_elapsed_seconds", "actual_elapsed_seconds", "tool_calls"):
            values = [r[field] for r in selected if r[field] is not None]
            item[field] = sum(values) / len(values) if values else None
            item[field + "_denominator"] = len(values)
        for field in ("logical_input_tokens", "logical_output_tokens", "actual_input_tokens", "actual_output_tokens"):
            known = [r[field] for r in selected if r[field] is not None]
            item[field] = sum(known) if len(known) == len(selected) and selected else None
            item[field + "_known"] = sum(known)
        for field in ("exact_match", "f1"):
            item["group_macro_" + field] = sum(
                sum(by_id.get(qid, {}).get(field, 0.0) for qid in ids) / len(ids)
                for ids in groups.values()) / len(groups)
        item["served_models"] = sorted({m for r in selected for m in r["served_models"]})
        item["route_mismatch_count"] = len({r["question_id"] for r in selected if r.get("shared_route_consistent") is False})
        result[variant] = item
    complete = all(v["pending_count"] == 0 for v in result.values())
    full_split_complete = complete and expected == dataset_question_count
    formal = (options.backend == "live" and options.split == "test" and full_split_complete
              and not any(v["route_mismatch_count"] for v in result.values()))
    status = "partial" if not complete else ("complete" if full_split_complete else "smoke")
    selected_question_groups = {qid: gid for gid, qids in groups.items() for qid in qids}
    return {"experiment_fingerprint": fingerprint, "execution_backend": options.backend,
            "formal_results": formal, "result_kind": "mock" if options.backend == "mock" else ("formal" if formal else "development"),
            "complete": complete, "full_split_complete": full_split_complete,
            "status": status, "split": options.split, "dataset_question_count": dataset_question_count,
            "selected_question_count": expected,
            "selected_question_ids": [example.question.question_id for example in examples],
            "selected_question_groups": selected_question_groups,
            "variants": result,
            "notes": ["Mock output validates software only and is never formal experimental evidence.",
                      "EM/F1 denominators include failed and, for an interrupted run, pending questions.",
                      "graph_unsupported_answer_fraction is predicted IDs absent from this fixed-KG gold set; it is a QA error proxy, not a real-world hallucination determination.",
                      "Evidence context budget uses UTF-8 bytes, not exact tokenizer tokens.",
                      "Prediction logical cost includes shared routing; actual cost records only calls actually executed in each row. Do not add the two accounting systems.",
                      "Unknown usage is null. Token totals do not prove the exact vendor invoice; retries and interrupted uncommitted calls may cost extra.",
                      "The predefined GNN main configuration is gat_2_mean, never selected using test scores."]}


def _write_summary(output, rows, examples, options, fingerprint, dataset_question_count):
    summary = _aggregate(rows, examples, options, fingerprint, dataset_question_count)
    referenced = {r["trace_file"] for r in rows if r.get("trace_file")}
    summary["uncommitted_trace_count"] = sum(
        str(path.relative_to(output)).replace("\\", "/") not in referenced
        for path in (output / "traces").glob("*.json"))
    _atomic_json(output / "summary.json", summary)
    values = list(summary["variants"].values())
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        for item in values:
            writer.writerow({k: _canonical(v) if isinstance(v, (dict, list)) else v for k, v in item.items()})
    return summary


def _question_batches(tasks, evaluate, engine, workers):
    if workers == 1:
        for task in tasks:
            yield evaluate(task, engine)
        return

    state = local()
    stopped = Event()

    def run_task(task):
        if not hasattr(state, "engine"):
            state.engine = engine.new_worker()
        return evaluate(task, state.engine, stopped)

    pool = ThreadPoolExecutor(max_workers=workers)
    active = set()
    try:
        for _ in range(workers):
            task = next(tasks, None)
            if task is None:
                break
            active.add(pool.submit(run_task, task))
        while active:
            ready, active = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                yield future.result()
                task = next(tasks, None)
                if task is not None:
                    active.add(pool.submit(run_task, task))
    finally:
        stopped.set()
        pool.shutdown(wait=True, cancel_futures=True)


def run_evaluation(options: EvaluationOptions, *, engine=None):
    """Run deterministic selected questions, committing all variants per question.

    Only exact Question objects reach the engine. Gold answers, paths, quality
    labels and evidence are held by this evaluator after inference inputs are
    selected. A mock engine does not call a provider or import GNN dependencies.
    """
    if options.backend not in ("live", "mock") or options.split not in ("validation", "test"):
        raise ValueError("Choose backend live/mock and split validation/test")
    if not isinstance(options.limit, int) or options.limit < 1:
        raise ValueError("limit must be a positive integer; default is a small validation run")
    if type(options.workers) is not int or options.workers < 1:
        raise ValueError("workers must be a positive integer")
    if not options.variants or len(set(options.variants)) != len(options.variants) or set(options.variants) - set(VARIANTS):
        raise ValueError("Choose unique supported variants: " + ",".join(VARIANTS))
    data = Path(options.data)
    gold_path = data if data.is_file() else data / (options.split + ".jsonl")
    questions_path = gold_path.with_name(gold_path.stem + ".questions.jsonl")
    if not questions_path.is_file():
        questions_path = gold_path
    gold = load_examples(gold_path)
    questions = load_questions(questions_path)
    if [(e.question.question_id, e.question.text, e.question.topic_entity_ids) for e in gold] != [
            (q.question_id, q.text, q.topic_entity_ids) for q in questions]:
        raise ValueError("Question-only file and gold file do not describe the same ordered questions")
    examples, questions = gold[:options.limit], questions[:options.limit]
    if not examples:
        raise ValueError("No selected questions")
    output = Path(options.output)
    if output.resolve().is_relative_to(gold_path.parent.resolve()):
        raise ValueError("Evaluation output must not be inside the dataset directory")
    config = json.loads(options.config.read_text(encoding="utf-8-sig"))
    engine = engine or (MockEngine() if options.backend == "mock" else LiveEngine(options, config))
    if options.workers > 1 and not callable(getattr(engine, "new_worker", None)):
        raise ValueError("Concurrent evaluation requires isolated worker engines")
    engine.preflight()
    fingerprint, identity = _fingerprint(options, engine, gold_path, questions_path,
                                         [q.question_id for q in questions])
    output.mkdir(parents=True, exist_ok=True)
    (output / "traces").mkdir(exist_ok=True)
    manifest_path, predictions_path = output / "manifest.json", output / "predictions.jsonl"
    rows = []
    if manifest_path.exists() or predictions_path.exists():
        if not options.resume or not manifest_path.is_file() or not predictions_path.is_file():
            raise ValueError("Output already exists; use --resume with the identical experiment or choose a new output directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment_fingerprint") != fingerprint:
            raise ValueError("Resume rejected: data/config/model/weights/source/git fingerprint changed")
        for number, line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                raise ValueError(f"Incomplete/corrupt predictions row {number}; original file was preserved") from None
            if row.get("experiment_fingerprint") != fingerprint or row.get("execution_backend") != options.backend:
                raise ValueError("Resume rejected: prediction row fingerprint/backend does not match manifest")
            rows.append(row)
        allowed = {(q.question_id, variant) for q in questions for variant in options.variants}
        keys = [(r["question_id"], r["variant"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) - allowed:
            raise ValueError("Resume rejected: duplicate or unexpected prediction keys")
        for q in questions:
            present = {r["variant"] for r in rows if r["question_id"] == q.question_id}
            if present and present != set(options.variants):
                raise ValueError("Resume rejected: incomplete question block; original rows preserved to avoid mixing routes")
    else:
        manifest = {"experiment_fingerprint": fingerprint, "identity": identity,
                    "created_at": datetime.now(timezone.utc).isoformat(), "complete": False,
                    "execution_backend": options.backend, "main_gnn_variant": "gat_2_mean",
                    "resume_granularity": "whole question with every selected variant",
                    "gold_policy": "Only Question is passed to prediction; labels used only for evaluation"}
        _atomic_json(manifest_path, manifest)
        predictions_path.touch()
    manifest.update(dataset_question_count=len(gold), selected_question_count=len(examples),
                    split=options.split, variants=list(options.variants), workers=options.workers,
                    selected_question_ids=[q.question_id for q in questions],
                    selected_question_groups={e.question.question_id: hashlib.sha256(
                        _canonical([list(e.question.topic_entity_ids), list(e.relation_path)]).encode()
                    ).hexdigest()[:20] for e in examples},
                    status="partial", full_split_complete=False, formal_results=False,
                    result_kind="mock" if options.backend == "mock" else "development")
    _atomic_json(manifest_path, manifest)
    completed_ids = {r["question_id"] for r in rows}
    pending = ((example, question) for example, question in zip(examples, questions)
               if question.question_id not in completed_ids)
    try:
        with predictions_path.open("a", encoding="utf-8", newline="\n") as handle:
            def evaluate(task, worker, stopped=None):
                example, question = task
                group_id = manifest["selected_question_groups"][question.question_id]
                batch = []
                for variant in options.variants:
                    if stopped is not None and stopped.is_set():
                        return None
                    started = time.perf_counter()
                    prediction, error_type = None, None
                    # Clear stale state before running, without touching D's route cache.
                    try:
                        worker.last_trace = None
                    except (AttributeError, TypeError):
                        pass
                    try:
                        prediction = worker.run(question, variant)
                        if not isinstance(prediction, Prediction) or prediction.question_id != question.question_id:
                            raise TypeError("Predictor returned an incompatible Prediction")
                    except Exception as error:
                        # Never serialize exception messages: providers may echo credentials.
                        error_type = type(error).__name__
                    wall_time = time.perf_counter() - started
                    trace = copy.deepcopy(worker.last_trace or {})
                    trace_id = uuid.uuid4().hex
                    relative_trace = "traces/" + trace_id + ".json"
                    _atomic_json(output / relative_trace,
                                 {"experiment_fingerprint": fingerprint, "execution_backend": options.backend,
                                  "question_id": question.question_id, "variant": variant, "trace": trace})
                    row = _row(example, prediction, trace, variant, options, fingerprint, group_id, error_type, wall_time)
                    row["requested_model"] = getattr(worker, "requested_model", None)
                    row["trace_file"] = relative_trace
                    route = trace.get("shared_route_events")
                    row["route_fingerprint"] = hashlib.sha256(_canonical(route).encode()).hexdigest() if route is not None else None
                    batch.append(row)
                routes = {r["route_fingerprint"] for r in batch if r["mode"] != "llm" and r["route_fingerprint"]}
                for row in batch:
                    row["shared_route_consistent"] = len(routes) <= 1
                return batch

            with closing(_question_batches(pending, evaluate, engine, options.workers)) as batches:
                for batch in batches:
                    # One complete question is the persistence/resume unit. An interruption
                    # before this point leaves trace files, not mixed partially reused modes.
                    handle.write("".join(_canonical(row) + "\n" for row in batch))
                    handle.flush()
                    os.fsync(handle.fileno())
                    rows.extend(batch)
    finally:
        summary = _write_summary(output, rows, examples, options, fingerprint, len(gold))
        manifest.update(complete=summary["complete"], updated_at=datetime.now(timezone.utc).isoformat(),
                        record_count=len(rows), uncommitted_trace_count=summary["uncommitted_trace_count"],
                        full_split_complete=summary["full_split_complete"], status=summary["status"],
                        formal_results=summary["formal_results"], result_kind=summary["result_kind"])
        _atomic_json(manifest_path, manifest)
    return summary
