# Open Korean Historical Corpus

This repository provides the code and resources for the **Open Korean Historical Corpus**, a large-scale, openly licensed dataset of historical Korean texts.

## About the Corpus

This dataset is a large-scale collection of texts spanning 1,300 years of Korean history.

**Key Features:**

- **Size:** 17.7 million documents and 5.1 billion tokens.
- **Temporal Span:** 7th century to 2025.
- **Sources:** Compiled from 19 distinct institutional archives and public domain collections.
- **Languages:** Covers 6 languages, including:
  - Classical Chinese
  - Middle Korean
  - Early Modern Korean
  - Modern Korean
  - North Korean
  - Japanese
- **Writing Systems:** Includes under-represented scripts like Korean-style Sinitic (Idu) and Hanja-Hangul mixed script.

## Getting the Dataset

The full corpus is available for download on the Hugging Face Hub.

- **Sample (1.3 MB)**: [`./sample.jsonl`](./sample.jsonl)
- **Dataset (28.6 GB)**: https://huggingface.co/datasets/seyoungsong/Open-Korean-Historical-Corpus

## Repository Contents

This repository contains the code used for preprocessing, language and script identification, and classification (including the Idu classifier) described in our paper. Please note that the initial web crawling scripts are not included here to avoid placing an undue burden on the source institutions' servers. Researchers interested in the crawling code may request it from the corresponding authors, subject to an agreement on responsible use.

## License

- **Code:** The code in this repository is released under the **MIT License**.
- **Data:** The Open Korean Historical Corpus is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License**.

## Citation

```
To Be Announced
```
