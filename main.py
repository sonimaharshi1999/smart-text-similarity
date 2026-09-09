# Author: Maharshi Soni | License: MIT
"""Smart Text Similarity Engine — Entry point."""

import argparse
import sys

from similarity_engine import SimilarityEngine


def main():
    parser = argparse.ArgumentParser(description="Smart Text Similarity Engine")
    parser.add_argument("--text1", type=str, help="First text to compare")
    parser.add_argument("--text2", type=str, help="Second text to compare")
    parser.add_argument("--batch", type=str, help="CSV file for batch comparison")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (0.0–1.0)")
    args = parser.parse_args()

    engine = SimilarityEngine()

    if args.text1 and args.text2:
        result = engine.compare(args.text1, args.text2)
        verdict = "SIMILAR" if result["combined"] >= args.threshold else "NOT SIMILAR"

        print(f"\nSimilarity Analysis Results:")
        print(f"  Text 1: {args.text1[:80]}")
        print(f"  Text 2: {args.text2[:80]}")
        print(f"\n  TF-IDF Cosine:  {result['tfidf_cosine']:.2f}")
        print(f"  Jaccard Index:   {result['jaccard']:.2f}")
        print(f"  Combined Score:  {result['combined']:.2f}")
        print(f"  Verdict:         {verdict}")

    elif args.batch:
        results = engine.batch_compare(args.batch, args.threshold)
        print(f"\nBatch Comparison Results ({len(results)} pairs):")
        for r in results[:10]:
            print(f"  [{r['score']:.2f}] {r['text1'][:40]}... <-> {r['text2'][:40]}...")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
