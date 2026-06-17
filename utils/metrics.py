"""Métricas de avaliação: BLEU e ROUGE.

Implementa funções utilitárias para calcular BLEU (corpus-level) e ROUGE
(médias de precisão/recall/f1 por tipo). As funções tentam usar bibliotecas
robustas quando disponíveis e lançam um erro informativo caso faltem
dependências.

Dependências opcionais:
- `sacrebleu` para BLEU: `pip install sacrebleu`
- `rouge-score` para ROUGE: `pip install rouge-score`
"""
from typing import List, Dict


def compute_bleu(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calcula BLEU em nível de corpus.

    Args:
        references: lista de strings (referências de tamanho N).
        predictions: lista de strings (previsões de tamanho N).

    Returns:
        Dicionário com chave "bleu" contendo o score (float, 0-100).

    Raises:
        ImportError: se `sacrebleu` não estiver instalado.
        ValueError: se tamanhos de listas divergem.
    """
    if len(references) != len(predictions):
        raise ValueError("`references` e `predictions` devem ter o mesmo tamanho")

    try:
        import sacrebleu
    except Exception as e:  # pragma: no cover - runtime dependency check
        raise ImportError(
            "Para calcular BLEU instale 'sacrebleu' (pip install sacrebleu)"
        ) from e

    # sacrebleu espera referências como lista de lista (cada referência é uma lista
    # com as referências possíveis). Aqui usamos uma única referência por sentença.
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return {"bleu": float(bleu.score)}


def compute_rouge(references: List[str], predictions: List[str]) -> Dict[str, Dict[str, float]]:
    """Calcula ROUGE-1, ROUGE-2 e ROUGE-L como média das métricas por par.

    Args:
        references: lista de strings (referências de tamanho N).
        predictions: lista de strings (previsões de tamanho N).

    Returns:
        Dicionário com chaves 'rouge1', 'rouge2', 'rougeL' cada uma contendo um
        dicionário com as chaves 'precision', 'recall', 'fmeasure' (valores entre 0 e 1).

    Raises:
        ImportError: se `rouge_score` não estiver instalado.
        ValueError: se tamanhos de listas divergem.
    """
    if len(references) != len(predictions):
        raise ValueError("`references` e `predictions` devem ter o mesmo tamanho")

    try:
        from rouge_score import rouge_scorer
    except Exception as e:  # pragma: no cover - runtime dependency check
        raise ImportError(
            "Para calcular ROUGE instale 'rouge-score' (pip install rouge-score)"
        ) from e

    keys = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(keys, use_stemmer=True)

    # acumular somas
    agg = {k: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for k in keys}
    n = len(references)

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        for k in keys:
            agg[k]["precision"] += scores[k].precision
            agg[k]["recall"] += scores[k].recall
            agg[k]["fmeasure"] += scores[k].fmeasure

    # média
    for k in keys:
        agg[k]["precision"] = agg[k]["precision"] / n if n else 0.0
        agg[k]["recall"] = agg[k]["recall"] / n if n else 0.0
        agg[k]["fmeasure"] = agg[k]["fmeasure"] / n if n else 0.0

    return agg


__all__ = ["compute_bleu", "compute_rouge"]
