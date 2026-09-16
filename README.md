# manga2text_tmp

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/manga2text_tmp/blob/main/manga2text_colab.ipynb)

Colab notebook for extracting dialogue text from manga/webtoon PDFs or images.

Pipeline:

`PDF / images -> Koharu RF-DETR -> MangaOCR or PaddleOCR -> language detection -> optional Qwen translation -> JSONL / TXT`
