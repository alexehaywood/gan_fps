esearch -db gds -query '"gse"[Entry Type] AND "GSE25097"[GEO Accession]' | \
efetch -format docsum |
xtract -pattern DocumentSummary -element Accession > Accessions_GSE25097.txt


esearch -db gds -query '"gse"[Entry Type] AND "GSE14520"[GEO Accession]' | \
efetch -format docsum |
xtract -pattern DocumentSummary -element Accession > Accessions_GSE14520.txt

esearch -db gds -query '"gse"[Entry Type] AND "GSE36376"[GEO Accession]' | \
efetch -format docsum |
xtract -pattern DocumentSummary -element Accession > Accessions_GSE36376.txt
