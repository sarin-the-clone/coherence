# Selected reference books

The full book `.txt` files in this folder are local inputs and are ignored by
Git. The `metrics_*.json` trajectory measurements remain versioned.

PLS-DA can be refitted from the descriptor exports without these texts. The
texts are needed to recompute selected-book/landmark measurements and to run
the model-based `test_semantic_coherence.py` script.

To provide a missing source, follow its Project Gutenberg landing-page link,
choose the plain-text edition, and save it in this folder under the filename
below. Check the source's applicable terms and retain its notices. The IDs and
filenames identify the historical inputs; a current download is not guaranteed
to be byte-identical to the version originally analysed.

| Gutenberg source | Expected local filename |
| --- | --- |
| [108](https://www.gutenberg.org/ebooks/108) | `108_The_Return_of_Sherlock_Holmes.txt` |
| [11323](https://www.gutenberg.org/ebooks/11323) | `11323_Caleb_Williams.txt` |
| [1184](https://www.gutenberg.org/ebooks/1184) | `1184_The_Count_of_Monte_Cristo.txt` |
| [1228](https://www.gutenberg.org/ebooks/1228) | `1228_On_the_Origin_of_Species.txt` |
| [12623](https://www.gutenberg.org/ebooks/12623) | `12623_Robinson_Crusoe.txt` |
| [12669](https://www.gutenberg.org/ebooks/12669) | `12669_Marriage.txt` |
| [1342](https://www.gutenberg.org/ebooks/1342) | `1342_Pride_and_Prejudice.txt` |
| [135](https://www.gutenberg.org/ebooks/135) | `135_Les_Misérables.txt` |
| [24527](https://www.gutenberg.org/ebooks/24527) | `24527_Fragments_of_Science.txt` |
| [2489](https://www.gutenberg.org/ebooks/2489) | `2489_Moby_Dick.txt` |
| [2600](https://www.gutenberg.org/ebooks/2600) | `2600_War_and_Peace.txt` |
| [2701](https://www.gutenberg.org/ebooks/2701) | `2701_Moby_Dick.txt` |
| [3008](https://www.gutenberg.org/ebooks/3008) | `3008_The_Jargon_File.txt` |
| [30486](https://www.gutenberg.org/ebooks/30486) | `30486_Shirley.txt` |
| [34916](https://www.gutenberg.org/ebooks/34916) | `34916_The_Chainbearer.txt` |
| [4300](https://www.gutenberg.org/ebooks/4300) | `4300_Ulysses.txt` |
| [50801](https://www.gutenberg.org/ebooks/50801) | `50801_Narrative_and_Critical_History_of_America.txt` |
| [51649](https://www.gutenberg.org/ebooks/51649) | `51649_The_Mysteries_of_London.txt` |
| [54824](https://www.gutenberg.org/ebooks/54824) | `54824_The_Secret_Doctrine,_Vol._1.txt` |
| [58711](https://www.gutenberg.org/ebooks/58711) | `58711_Traced_and_Tracked.txt` |
| [63912](https://www.gutenberg.org/ebooks/63912) | `63912_The_American_Encyclopedia_of_History,_Biography_and_Travel.txt` |
| [65901](https://www.gutenberg.org/ebooks/65901) | `65901_Heart_of_the_World.txt` |
| [67090](https://www.gutenberg.org/ebooks/67090) | `67090_The_Worm_Ouroboros.txt` |
| [7469](https://www.gutenberg.org/ebooks/7469) | `7469_Daniel_Deronda.txt` |
| [76058](https://www.gutenberg.org/ebooks/76058) | `76058_Aspasia.txt` |
| [84](https://www.gutenberg.org/ebooks/84) | `84_Frankenstein.txt` |
| [996](https://www.gutenberg.org/ebooks/996) | `996_Don_Quixote.txt` |

Earlier Git commits still contain the previously versioned texts. This change
removes them from the current tree and prevents future accidental additions;
it does not rewrite repository history.
