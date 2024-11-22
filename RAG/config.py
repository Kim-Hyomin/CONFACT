import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG Experiment")

    parser.add_argument("--seed", type=int, default = 2024, help="seed")


    # File paths
    parser.add_argument('--search_query_file', type=str, default="../data/dataset/google_queries.json",
                        help="Path to the JSON file with Google serch queries.")

    parser.add_argument('--all_evidence_url_file', type=str, default="./media_bg_collected/google_evidence_urls.pkl",
                        help="Path to the pickle file containing searched evidence URLs.")

    parser.add_argument('--all_scraped_text_file', type=str, default="./media_bg_collected/google_evidence_text.pkl",
                        help="Path to the pickle file containing scraped evidence text content.")

    parser.add_argument('--credibility_data_file', type=str, default="./media_bg_collected/media_credibility_data.pkl",
                        help="Path to the pickle file with media credibility data.")

    parser.add_argument('--label_demo_file', type=str, default="./media_bg_collected/label_demos.pkl",
                        help="Path to the pickle file with demos for classification.")

    parser.add_argument('--description_demo_file', type=str, default="./media_bg_collected/description_demos.pkl",
                        help="Path to the pickle file with description demos.")

    # Media Checking
    parser.add_argument('--wiki_flag', type=bool, default=True)
    parser.add_argument('--article_flag', type=bool, default=True)
    parser.add_argument('--google_flag', type=bool, default=True)

    # Model and GPU
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path for the AI model to be used.")
    parser.add_argument('--gpu', type=int, default=1,
                        help="GPU device number to use.")


    # RAG
    parser.add_argument("--source", default= "../data/dataset/filtered_data.pkl", type=str, help="Queries and evidence to run the RAG experiment on")

    parser.add_argument("--evidences", nargs='+', default="../data/dataset/intermediate_datasets/sample.json", help="List of evidences for BM25 retrieval")

    parser.add_argument('--sentence_store_path', default="../data/dataset/evidence/evidence.pkl", help='Path to store evidence sentences in a .pkl file')

    parser.add_argument('--k', default=20, type=int, help='Number of  relevant sentences to keep for final evaluation')

    parser.add_argument("--with_MediaBG", type=bool, default = False, help="Whether to consider Media Background")

    parser.add_argument("--method", type=str, choices=["BM25", "filtered", "reranked"], default="BM25", help="Retrieval processing method")

    parser.add_argument("--prompt_type", type=str, choices=["DiscernAndAnswer", "ExplainAndAnswer"], default="DiscernAndAnswer", help="Prompt type for LLM")

    parser.add_argument("--zero_shot", default = False, action='store_true', help="Enable zero-shot mode")


    return parser.parse_args()