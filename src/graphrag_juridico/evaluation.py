"""Avaliação comparativa: LLM puro x RAG vetorial x GraphRAG/KAG, no conjunto de perguntas com gabarito.

Cada resposta é gravada em `resources/evaluation_results.json` assim que chega: se a cota do LLM acabar no
meio, basta rodar de novo que só o que falta é feito. As mesmas respostas alimentam a comparação do
modo demonstração da interface. O relatório (tabela + gráfico) vai para `evaluation/` na raiz do repositório.

Uso:
    python -m graphrag_juridico.evaluation status            # o que já foi gravado
    python -m graphrag_juridico.evaluation run               # roda o que falta (usa banco e LLM)
    python -m graphrag_juridico.evaluation run --only E1,A2  # só algumas perguntas
    python -m graphrag_juridico.evaluation report            # gera evaluation/report.md e metrics.png
"""
import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

from .ask import ask
from .baselines import llm_only, vector_rag
from .config import ROOT
from .contracts import AskResult
from .metrics import Reference, context_norms, evaluate, glossary, norm_label

RESOURCES = Path(__file__).parent / "resources"
QUESTIONS_FILE = RESOURCES / "evaluation_questions.json"
RESULTS_FILE = RESOURCES / "evaluation_results.json"
REPORT_DIR = ROOT / "evaluation"
METHODS: dict[str, tuple[str, Callable[[str], AskResult]]] = {
    "llm": ("LLM puro", llm_only),
    "vector": ("RAG vetorial", vector_rag),
    "graphrag": ("GraphRAG/KAG", lambda q: ask(q, mock=False)),
}
METHOD_COLORS = {"llm": "#BAB0AC", "vector": "#4E79A7", "graphrag": "#F28E2B"}   # mesmas cores em todo lugar
MAX_CHUNK_CHARS = 700   # trechos gravados são cortados: o texto completo fica no banco


@dataclass
class Question:
    id: str
    type: str
    question: str
    expected: list[list[str]] = field(default_factory=list)
    should_abstain: bool = False
    note: str = ""

    @property
    def reference(self) -> Reference:
        return Reference(expected=self.expected, should_abstain=self.should_abstain)


def load_questions() -> list[Question]:
    return [Question(**q) for q in json.loads(QUESTIONS_FILE.read_text("utf-8"))]


def find_question(text: str) -> Question | None:
    """Pergunta do conjunto de avaliação com o mesmo texto (ignora maiúsculas, espaços e '?')."""
    key = " ".join(text.lower().split()).rstrip("?").strip()
    return next((q for q in load_questions() if " ".join(q.question.lower().split()).rstrip("?") == key), None)


def load_results() -> dict[str, dict[str, dict]]:
    """{id da pergunta: {método: {"result": AskResult como dict, "latency": segundos}}}."""
    return json.loads(RESULTS_FILE.read_text("utf-8")) if RESULTS_FILE.exists() else {}


def save_results(results: dict[str, dict[str, dict]]) -> None:
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")


def run_method(method: str, question: str) -> dict:
    """Roda um método e mede a latência. As normas do contexto são calculadas antes de cortar os trechos
    para a gravação. Resposta vazia (falha do provedor) é erro: não é gravada e pode ser refeita."""
    start = time.perf_counter()
    result = METHODS[method][1](question)
    latency = round(time.perf_counter() - start, 2)
    if not result.answer.strip():
        raise RuntimeError(f"{METHODS[method][0]}: o LLM devolveu uma resposta vazia")
    norms = context_norms(result)
    for chunk in result.chunks:
        if len(chunk.text) > MAX_CHUNK_CHARS:
            chunk.text = chunk.text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + " [...]"
    return {"result": result.model_dump(), "latency": latency, "context_norms": norms}


def run_all(only: set[str] | None = None) -> list[str]:
    """Roda o que falta e grava a cada resposta. Para no primeiro erro (ex.: cota esgotada)."""
    results = load_results()
    for q in load_questions():
        if only and q.id not in only:
            continue
        for method, (label, _) in METHODS.items():
            if method in results.get(q.id, {}):
                continue
            try:
                run = run_method(method, q.question)
            except Exception as error:
                print(f"PAROU em {q.id}/{label}: {type(error).__name__}: {str(error)[:200]}")
                return [f"{q.id}/{method}"]
            results.setdefault(q.id, {})[method] = run
            save_results(results)
            print(f"ok {q.id} {label}: {run['latency']:.1f}s")
    return []


def score(results: dict[str, dict[str, dict]] | None = None) -> list[dict]:
    """Uma linha por (pergunta, método) com as métricas e a latência."""
    results = load_results() if results is None else results
    rows = []
    for q in load_questions():
        for method, run in results.get(q.id, {}).items():
            result = AskResult.model_validate(run["result"])
            rows.append({"id": q.id, "type": q.type, "question": q.question, "method": method,
                         "should_abstain": q.should_abstain, "latency": run["latency"],
                         **evaluate(result, q.reference, run.get("context_norms")).as_dict()})
    return rows


def summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Agregados por método (frações de 0 a 1; latência em segundos)."""
    out = {}
    for method in METHODS:
        mine = [r for r in rows if r["method"] == method]
        if not mine:
            continue
        answerable = [r for r in mine if not r["should_abstain"]]
        traps = [r for r in mine if r["should_abstain"]]
        out[method] = {
            "accuracy": mean(r["correct"] for r in mine),
            "trap_abstention": mean(r["abstained"] for r in traps) if traps else float("nan"),
            "wrong_abstention": mean(r["abstained"] for r in answerable) if answerable else float("nan"),
            "citations_valid": mean(r["citations_valid"] for r in mine),
            "grounded_sentences": mean(r["grounded_sentences"] for r in mine),
            "with_unsupported_norms": mean(bool(r["unsupported_norms"]) for r in mine),
            "latency": median(r["latency"] for r in mine),
            "questions": len(mine),
        }
    return out


SUMMARY_LABELS = {
    "accuracy": "Acerto (gabarito)",
    "trap_abstention": "Abstenção correta nas armadilhas",
    "wrong_abstention": "Abstenção indevida nas respondíveis",
    "citations_valid": "Citações válidas",
    "grounded_sentences": "Frases com fonte",
    "with_unsupported_norms": "Respostas com normas sem suporte",
    "latency": "Latência mediana (s)",
}


def _fmt(key: str, value: float) -> str:
    if value != value:  # NaN
        return "-"
    return f"{value:.1f}" if key == "latency" else f"{value:.0%}"


def write_report() -> Path:
    rows = score()
    agg = summary(rows)
    methods = [m for m in METHODS if m in agg]
    REPORT_DIR.mkdir(exist_ok=True)
    lines = ["# Avaliação: LLM puro x RAG vetorial x GraphRAG/KAG", "",
             f"{len({r['id'] for r in rows})} perguntas ({', '.join(sorted({q.type for q in load_questions()}))}). "
             "Métricas automáticas definidas em `src/graphrag_juridico/metrics.py`; perguntas e gabarito em "
             "`src/graphrag_juridico/resources/evaluation_questions.json`.", "",
             "| Métrica | " + " | ".join(METHODS[m][0] for m in methods) + " |",
             "|---|" + "---|" * len(methods)]
    for key, label in SUMMARY_LABELS.items():
        lines.append(f"| {label} | " + " | ".join(_fmt(key, agg[m][key]) for m in methods) + " |")
    lines += ["", "Para \"Abstenção indevida\" e \"Respostas com normas sem suporte\", menor é melhor.", "",
              "![Métricas](metrics.png)", "", "## Por pergunta", "",
              "| Pergunta | Tipo | " + " | ".join(METHODS[m][0] for m in methods) + " |",
              "|---|---|" + "---|" * len(methods)]
    for q in load_questions():
        cells = []
        for m in methods:
            r = next((r for r in rows if r["id"] == q.id and r["method"] == m), None)
            if r is None:
                cells.append("-")
                continue
            mark = "✅" if r["correct"] else "❌"
            extra = " · absteve" if r["abstained"] else ""
            if r["unsupported_norms"]:
                extra += " · normas sem suporte: " + ", ".join(norm_label(n) for n in r["unsupported_norms"])
            cells.append(f"{mark}{extra}")
        lines.append(f"| {q.id}: {q.question} | {q.type} | " + " | ".join(cells) + " |")
    cited = [norm_label(n) for r in rows for n in r["unsupported_norms"]]
    if cited:
        lines += ["", "**Siglas e leis citadas:** " + " · ".join(glossary(cited)) + "."]
    lines += ["", "**Normas sem suporte:** leis citadas na resposta que não aparecem em nenhum trecho, caminho ou "
              "resultado de consulta do contexto (não verificáveis no corpus; indício de alucinação)."]
    report = REPORT_DIR / "report.md"
    report.write_text("\n".join(lines) + "\n", "utf-8")
    _chart(agg, methods, REPORT_DIR / "metrics.png")
    return report


def _chart(agg: dict, methods: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in SUMMARY_LABELS if k != "latency"]
    colors = METHOD_COLORS
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        values = [0 if agg[m][k] != agg[m][k] else agg[m][k] for k in keys]
        ax.bar([x + i * width for x in range(len(keys))], values, width, label=METHODS[m][0], color=colors[m])
    ax.set_xticks([x + width * (len(methods) - 1) / 2 for x in range(len(keys))])
    ax.set_xticklabels([SUMMARY_LABELS[k].replace(" nas ", "\nnas ").replace(" com ", "\ncom ") for k in keys], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação comparativa dos métodos.")
    parser.add_argument("command", choices=["status", "run", "report"])
    parser.add_argument("--only", help="ids separados por vírgula (ex.: E1,A2)")
    args = parser.parse_args()
    if args.command == "run":
        run_all(set(args.only.split(",")) if args.only else None)
    elif args.command == "report":
        print(f"Relatório: {write_report()}")
    else:
        results = load_results()
        for q in load_questions():
            done = [METHODS[m][0] for m in METHODS if m in results.get(q.id, {})]
            print(f"{q.id:3} {len(done)}/{len(METHODS)}  {q.question}")


if __name__ == "__main__":
    main()
