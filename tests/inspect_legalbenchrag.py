import random

from rich.console import Console
from rich.panel import Panel

from src.data import load_legalbench_contractnli

console = Console()

def inspect_random_examples(data, num_examples=3):
    # LegalBench-RAG benchmark jsons have a list of test cases with query + ground_truth snippets.
    test_cases = data.test_cases
    examples = random.sample(test_cases, min(num_examples, len(test_cases)))

    for i, case in enumerate(examples, 1):
        query = case.get("query") or case.get("question")
        ground_truth = case.get("ground_truth") or case.get("snippets") or []

        console.rule(f"[bold yellow]Example {i}[/bold yellow]")
        console.print(Panel.fit(query, title="Query", style="cyan"))

        if not ground_truth:
            console.print("[red]No ground_truth/snippets found for this case[/red]")
            continue

        # Just look at the first snippet
        snippet = ground_truth[0]
        rel_path = snippet.get("file_path") or snippet.get("file") or snippet.get("doc_path")
        span = snippet.get("span")
        if isinstance(span, list) and len(span) == 2:
            start, end = span
        else:
            start = snippet.get("start") or snippet.get("start_char")
            end = snippet.get("end") or snippet.get("end_char")

        console.print(f"[bold]Snippet file:[/bold] {rel_path}")
        console.print(f"[bold]Char span:[/bold] {start} – {end}")

        if rel_path is None or start is None or end is None:
            console.print("[red]Missing file_path/start/end in snippet[/red]")
            continue

        corpus_file = data.corpus_dir / rel_path
        if not corpus_file.exists():
            console.print(f"[red]Corpus file not found:[/red] {corpus_file}")
            continue

        text = corpus_file.read_text(encoding="utf-8", errors="ignore")
        span_text = text[start:end]

        console.print(Panel.fit(span_text, title="Gold span from contract", style="green"))

def main():
    data = load_legalbench_contractnli()
    console.print(f"[bold]Using benchmark file:[/bold] {data.benchmark_path}")
    inspect_random_examples(data, num_examples=3)

if __name__ == "__main__":
    main()
