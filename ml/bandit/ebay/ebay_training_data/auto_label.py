import csv
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Full scored CSV")
    ap.add_argument("--auto-negative-below", type=float, default=0.15)
    ap.add_argument("--auto-positive-above", type=float, default=0.80)
    ap.add_argument("--review-out", default="review.csv")
    ap.add_argument("--labels-out", default="auto_labels.csv")
    ap.add_argument("--sort", choices=["asc", "desc"], default="asc", help="Sort review band by tfidf_score")
    args = ap.parse_args()

    auto_negative = []
    auto_positive = []
    review = []

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = float(row["tfidf_score"])
            except (ValueError, KeyError):
                continue

            if score < args.auto_negative_below:
                auto_negative.append((row, 0))
            elif score > args.auto_positive_above:
                auto_positive.append((row, 1))
            else:
                review.append(row)

    review.sort(key=lambda r: float(r["tfidf_score"]), reverse=(args.sort == "desc"))

    with open(args.labels_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ebay_id", "ebay_title", "price", "tfidf_score", "wanted"])
        for row, label in auto_negative + auto_positive:
            writer.writerow([row["ebay_id"], row["ebay_title"], row["price"], row["tfidf_score"], label])

    with open(args.review_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ebay_id", "ebay_title", "price", "tfidf_score"])
        for row in review:
            writer.writerow([row["ebay_id"], row["ebay_title"], row["price"], row["tfidf_score"]])

    print(f"Auto-labeled negative: {len(auto_negative)}")
    print(f"Auto-labeled positive: {len(auto_positive)}")
    print(f"Needs review: {len(review)} (sorted by tfidf_score {args.sort})")
    print(f"Auto labels written to: {args.labels_out}")
    print(f"Review band written to: {args.review_out}")

if __name__ == "__main__":
    main()