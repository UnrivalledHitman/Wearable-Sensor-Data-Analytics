# Reference verification

Every reference in the current draft carried wrong authors, journal, year or
pages. `references.bib` replaces them with details read from the sources
themselves. Checked on 2026-10-02.

## Old reference number to corrected entry

| Old ref | Cited in the draft as | Corrected BibTeX key | Verified from |
| --- | --- | --- | --- |
| [1] | Alattar and Mohsen, IEEE Access 10, 2022 | `alattar2023survey` | PDF first page (`Related Work/A Survey on Smart Wearable Devices for Healthcare.pdf`) |
| [2] | Alhassan, Hamza and Baroli, IEEE Trans. Learning Technologies, 2023 | `xie2025sharp` | PDF first page and the MDPI article page |
| [3] | Viera, Carvalho, Marinho and Almeida, Sensors 22(3), 2022 | `farabolini2025continuous` | PDF first page and the MDPI article page |
| [4] | D. Kumar, Int. J. of Advanced Research, 2021 | `dilipkumar2021data` | PDF first page |
| [5] | Hossain and Muhammad, IEEE Internet of Things J. 5(2), 2018 | none, see below | PDF first page |
| [6] | M. Khan et al., IEEE Access 9, 2021 | `khan2024realtime` | PDF first page |
| [7] | E. Mahabub et al., J. of Medical Systems 46(4), 2022 | `mahabub2024impact` | PDF first page |
| [8] | R. Parasuraman et al., IEEE Communications Magazine 58(1), 2020 | `vijayan2021review` | PDF first page |
| [9] | T. Ave, Pattern Recognition Letters 145, 2021 | none, see below | PDF first page |
| [10] | Min et al., Advanced Healthcare Materials 10(1), 2021 | `min2023sweat` | PDF first page and the PubMed Central record |

## Sources to drop or replace

- **Old [5].** The text describes "Integration of Wearable IoT Devices with
  Predictive Analytics for Remote Health Monitoring" by Gideon Areo, a
  ResearchGate upload from July 2025 with no journal. No paper matching the
  cited title and venue was found. Replace it with a peer-reviewed source or
  remove the paragraph.
- **Old [9].** The source is an SSRN/ResearchGate upload whose only listed
  author is the account "Research Publication". It cannot be cited reliably.
  Remove it.

## Sources of limited weight

`dilipkumar2021data`, `khan2024realtime` and `mahabub2024impact` are real
articles in journals that a reviewer may not regard as rigorously peer
reviewed. They can stay if cited accurately, but they should not carry the
argument. `uddin2020data` is an editorial.

## Still missing from the bibliography

The draft cites nothing on its actual topic. Before submission the related
work needs, each verified by DOI:

- the WESAD dataset paper (`schmidt2018wesad`, now in the `.bib`);
- 15-25 WESAD stress and affect recognition papers, marking which evaluate
  with leave-one-subject-out and which use random window splits;
- the CNN-LSTM models the abstract calls "state of the art", with their real
  parameter counts;
- lightweight and on-device stress detection work (three starting points are
  in the `.bib`);
- a source for every health claim in the abstract, or remove those claims.

## Rule from here on

No entry goes into `references.bib` unless its DOI, arXiv id or publisher page
has been opened and the authors, venue, year and pages copied from it.
