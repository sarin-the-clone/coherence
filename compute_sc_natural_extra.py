"""
Compute SC scalars for additional natural books (groups A–F for parallel runs).

Usage:
    python compute_sc_natural_extra.py --group a
    ...
    python compute_sc_natural_extra.py --group f

Output:
    /tmp/natural_sc_extra_{a,b,c,d,e,f}.json

These are merged with /tmp/natural_books_sc_median.json in the plot scripts.
"""

import argparse
import json
import os
import sys
import time
import zipfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from semantic_coherence import SemanticCoherence, WINDOW_NAMES, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ZIP_PATH   = "natural_texts/Output/gutenberg_benchmark_cache/books.zip"
WINDOWS    = [1, 2, 3, 4]
BATCH_SIZE = 64
MAX_LENGTH = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SCALAR_KEYS = ["off_diag_mean", "novelty_mean", "sim_lag1_mean", "mean_sim_to_past_mean"]

# ---------------------------------------------------------------------------
# Book lists
# ---------------------------------------------------------------------------

BOOKS_A = [
    # Bin 4 (~1M tok)
    {"book_id": 71104, "bin": 4, "title": "English & Scottish Popular Ballads",
     "member": "books/71104.txt"},
    {"book_id": 47790, "bin": 4, "title": "Works of William Cowper",
     "member": "books/47790.txt"},
    {"book_id": 19217, "bin": 4, "title": "History of England Vol. II",
     "member": "books/19217.txt"},
    # Bin 3 (~580k tok)
    {"book_id": 15263, "bin": 3, "title": "The Underground Railroad",
     "member": "books/15263.txt"},
    {"book_id": 55192, "bin": 3, "title": "History of North American Birds",
     "member": "books/55192.txt"},
    {"book_id":  8187, "bin": 3, "title": "Complete Poems of Thomas Moore",
     "member": "books/8187.txt"},
    # Bin 2 (~354k tok)
    {"book_id": 15493, "bin": 2, "title": "The Lancashire Witches",
     "member": "books/15493.txt"},
    {"book_id": 61398, "bin": 2, "title": "The Polar World",
     "member": "books/61398.txt"},
    {"book_id": 57143, "bin": 2, "title": "History of Greece Vol. 02",
     "member": "books/57143.txt"},
    {"book_id": 10574, "bin": 2, "title": "History of England Vol. I",
     "member": "books/10574.txt"},
    # Bin 1 (~194k tok)
    {"book_id": 12455, "bin": 1, "title": "Legends of the Middle Ages",
     "member": "books/12455.txt"},
    {"book_id": 49331, "bin": 1, "title": "The Young Physician",
     "member": "books/49331.txt"},
    {"book_id": 18440, "bin": 1, "title": "Logic: Deductive and Inductive",
     "member": "books/18440.txt"},
    {"book_id": 53486, "bin": 1, "title": "The Three Fates",
     "member": "books/53486.txt"},
    # Bin 0 (~121k tok)
    {"book_id": 54030, "bin": 0, "title": "Ben o' Bill's, the Luddite",
     "member": "books/54030.txt"},
    {"book_id":  1036, "bin": 0, "title": "Joe Wilson and His Mates",
     "member": "books/1036.txt"},
    {"book_id": 28276, "bin": 0, "title": "The End of the Rainbow",
     "member": "books/28276.txt"},
]

BOOKS_B = [
    # Bin 4 (~1M tok)
    {"book_id":  4546, "bin": 4, "title": "Memoirs of Union Civil War Generals",
     "member": "books/4546.txt"},
    {"book_id": 71897, "bin": 4, "title": "History for Ready Reference Vol. 6",
     "member": "books/71897.txt"},
    {"book_id": 40851, "bin": 4, "title": "Abridgment Debates of Congress",
     "member": "books/40851.txt"},
    # Bin 3 (~580k tok)
    {"book_id": 44438, "bin": 3, "title": "The War with Mexico Vol. 2",
     "member": "books/44438.txt"},
    {"book_id":  3875, "bin": 3, "title": "Memoirs of Louis XIV",
     "member": "books/3875.txt"},
    {"book_id":  6589, "bin": 3, "title": "History of Modern Europe 1792-1878",
     "member": "books/6589.txt"},
    {"book_id": 51960, "bin": 3, "title": "Ancient Stone Implements",
     "member": "books/51960.txt"},
    # Bin 2 (~354k tok)
    {"book_id": 48555, "bin": 2, "title": "History of the Order of the Garter",
     "member": "books/48555.txt"},
    {"book_id":  7883, "bin": 2, "title": "Expositions of Holy Scripture",
     "member": "books/7883.txt"},
    {"book_id": 13055, "bin": 2, "title": "History & Collection of Voyages",
     "member": "books/13055.txt"},
    # Bin 1 (~194k tok)
    {"book_id":  5337, "bin": 1, "title": "Thirty-One Years on the Plains",
     "member": "books/5337.txt"},
    {"book_id": 31053, "bin": 1, "title": "History of the Devil",
     "member": "books/31053.txt"},
    {"book_id":  6323, "bin": 1, "title": "The Junior Classics Vol. 4",
     "member": "books/6323.txt"},
    # Bin 0 (~121k tok)
    {"book_id": 34134, "bin": 0, "title": "The Great Mogul",
     "member": "books/34134.txt"},
    {"book_id": 28690, "bin": 0, "title": "The Empire of the East",
     "member": "books/28690.txt"},
    {"book_id": 14149, "bin": 0, "title": "The Pilots of Pomona",
     "member": "books/14149.txt"},
    {"book_id": 18569, "bin": 0, "title": "Hypnerotomachia",
     "member": "books/18569.txt"},
]

BOOKS_C = [
    # Bin 4 (~1M tok)
    {"book_id": 72658, "bin": 4, "title": "Principles of Modern Surgery",
     "member": "books/72658.txt"},
    {"book_id": 15475, "bin": 4, "title": "The Mahabharata Vol. 2",
     "member": "books/15475.txt"},
    # Bin 3 (~580k tok)
    {"book_id": 42808, "bin": 3, "title": "The Native Races of the Pacific States",
     "member": "books/42808.txt"},
    {"book_id": 46752, "bin": 3, "title": "History of the Jews Vol. 6",
     "member": "books/46752.txt"},
    {"book_id": 71209, "bin": 3, "title": "The Historians' History of the World",
     "member": "books/71209.txt"},
    # Bin 2 (~354k tok)
    {"book_id": 74316, "bin": 2, "title": "Geology Vol. 1",
     "member": "books/74316.txt"},
    {"book_id":  8726, "bin": 2, "title": "The Three Cities Trilogy: Rome",
     "member": "books/8726.txt"},
    {"book_id": 15293, "bin": 2, "title": "Influences of Geographic Environment",
     "member": "books/15293.txt"},
    {"book_id": 60736, "bin": 2, "title": "History of the LDS Church",
     "member": "books/60736.txt"},
    # Bin 1 (~194k tok)
    {"book_id": 70835, "bin": 1, "title": "Shakespeare's Environment",
     "member": "books/70835.txt"},
    {"book_id": 11212, "bin": 1, "title": "Modern India",
     "member": "books/11212.txt"},
    {"book_id":  4522, "bin": 1, "title": "The Fight for the Republic in China",
     "member": "books/4522.txt"},
    {"book_id": 19690, "bin": 1, "title": "Apologia pro vita sua",
     "member": "books/19690.txt"},
    # Bin 0 (~121k tok)
    {"book_id": 33472, "bin": 0, "title": "With the World's Great Travellers Vol. 2",
     "member": "books/33472.txt"},
    {"book_id": 40838, "bin": 0, "title": "The Empire Makers",
     "member": "books/40838.txt"},
    {"book_id": 18459, "bin": 0, "title": "Hypnerotomachia: Strife of Love in Dream",
     "member": "books/18459.txt"},
]

BOOKS_D = [
    {"book_id": 6930, "bin": 4, "title": "The Faerie Queene — Volume 01", "member": "books/6930.txt"},
    {"book_id": 51789, "bin": 4, "title": "Narrative and Critical History of America, Vol. 6 (of 8", "member": "books/51789.txt"},
    {"book_id": 35354, "bin": 4, "title": "Moral Theology  A Complete Course Based on St. Thomas A", "member": "books/35354.txt"},
    {"book_id": 571, "bin": 4, "title": "The 1995 CIA World Factbook", "member": "books/571.txt"},
    {"book_id": 61626, "bin": 4, "title": "The Secret Doctrine, Vol. 4 of 4 The Synthesis of Scien", "member": "books/61626.txt"},
    {"book_id": 664, "bin": 4, "title": "The Gutenberg Webster\'s Unabridged Dictionary: Section", "member": "books/664.txt"},
    {"book_id": 1662, "bin": 4, "title": "The 1997 CIA World Factbook", "member": "books/1662.txt"},
    {"book_id": 44525, "bin": 4, "title": "Report on the Radiolaria Collected by H.M.S. Challenger", "member": "books/44525.txt"},
    {"book_id": 59468, "bin": 4, "title": "The Scottish Highlands, Highland Clans and Highland Reg", "member": "books/59468.txt"},
    {"book_id": 53305, "bin": 4, "title": "A system of practical medicine. By American authors. Vo", "member": "books/53305.txt"},
    {"book_id": 74739, "bin": 4, "title": "A military dictionary", "member": "books/74739.txt"},
    {"book_id": 668, "bin": 4, "title": "The Gutenberg Webster\'s Unabridged Dictionary: Section", "member": "books/668.txt"},
    {"book_id": 59553, "bin": 4, "title": "The Southern Literary Messenger, Volume I., 1834-35", "member": "books/59553.txt"},
    {"book_id": 48105, "bin": 4, "title": "Dictionary of Quotations from Ancient and Modern, Engli", "member": "books/48105.txt"},
    {"book_id": 6800, "bin": 4, "title": "The Works of Frederick Schiller", "member": "books/6800.txt"},
    {"book_id": 68302, "bin": 4, "title": "History for ready reference, Volume 3, Greece to Nibelu", "member": "books/68302.txt"},
    {"book_id": 50503, "bin": 4, "title": "Forty-Eighth Annual Report of the Bureau of American Et", "member": "books/50503.txt"},
    {"book_id": 55059, "bin": 4, "title": "Modern English biography, volume 1 (of 4), A-H", "member": "books/55059.txt"},
    {"book_id": 4200, "bin": 4, "title": "The Diary of Samuel Pepys — Complete", "member": "books/4200.txt"},
    {"book_id": 66791, "bin": 3, "title": "The City of the Saints, and Across the Rocky Mountains ", "member": "books/66791.txt"},
    {"book_id": 968, "bin": 3, "title": "Martin Chuzzlewit", "member": "books/968.txt"},
    {"book_id": 61649, "bin": 3, "title": "Pugilistica: The History of British Boxing, Volume 2 (o", "member": "books/61649.txt"},
    {"book_id": 44009, "bin": 3, "title": "Warren Commission (09 of 26): Hearings Vol. IX (of 15)", "member": "books/44009.txt"},
    {"book_id": 53625, "bin": 3, "title": "Shakspeare and His Times [Vol. 1 of 2]  Including the B", "member": "books/53625.txt"},
    {"book_id": 46804, "bin": 3, "title": "Bentley\'s Miscellany, Volume II", "member": "books/46804.txt"},
    {"book_id": 51393, "bin": 3, "title": "Toadstools, mushrooms, fungi, edible and poisonous; one", "member": "books/51393.txt"},
    {"book_id": 1399, "bin": 3, "title": "Anna Karenina", "member": "books/1399.txt"},
    {"book_id": 49008, "bin": 3, "title": "The Works of William Shakespeare [Cambridge Edition] [V", "member": "books/49008.txt"},
    {"book_id": 45283, "bin": 3, "title": "Systematic Theology (Volume 3 of 3)", "member": "books/45283.txt"},
    {"book_id": 75947, "bin": 3, "title": "An introduction to the study of mammals living and exti", "member": "books/75947.txt"},
    {"book_id": 61792, "bin": 3, "title": "Olympic Victor Monuments and Greek Athletic Art", "member": "books/61792.txt"},
    {"book_id": 59386, "bin": 3, "title": "The history of human marriage", "member": "books/59386.txt"},
    {"book_id": 58658, "bin": 3, "title": "History of Central America, Volume 1, 1501-1530  The Wo", "member": "books/58658.txt"},
    {"book_id": 54977, "bin": 3, "title": "The Sovereignty of the Sea  An Historical Account of th", "member": "books/54977.txt"},
    {"book_id": 40619, "bin": 3, "title": "Camilla; or, A Picture of Youth", "member": "books/40619.txt"},
    {"book_id": 51514, "bin": 3, "title": "The historians\' history of the world in twenty-five vo", "member": "books/51514.txt"},
    {"book_id": 55195, "bin": 3, "title": "The historians\' history of the world in twenty-five vo", "member": "books/55195.txt"},
    {"book_id": 747, "bin": 3, "title": "Anomalies and Curiosities of Medicine", "member": "books/747.txt"},
    {"book_id": 39580, "bin": 3, "title": "A History of the Inquisition of the Middle Ages; volume", "member": "books/39580.txt"},
    {"book_id": 45737, "bin": 3, "title": "A Popular Handbook to the National Gallery, Volume I, F", "member": "books/45737.txt"},
    {"book_id": 73369, "bin": 3, "title": "The historians\' history of the world in twenty-five vo", "member": "books/73369.txt"},
    {"book_id": 54652, "bin": 3, "title": "History of Gujarát Gazetteer of the Bombay Presidency, ", "member": "books/54652.txt"},
    {"book_id": 65341, "bin": 3, "title": "Treatise on Poisons In relation to medical jurisprudenc", "member": "books/65341.txt"},
    {"book_id": 31412, "bin": 3, "title": "Old and New London, Volume I  A Narrative of Its Histor", "member": "books/31412.txt"},
    {"book_id": 55368, "bin": 3, "title": "Nazi Conspiracy and Aggression, Volume 01 (of 11)", "member": "books/55368.txt"},
    {"book_id": 52056, "bin": 3, "title": "The Mysteries of London, v. 3/4", "member": "books/52056.txt"},
    {"book_id": 28496, "bin": 3, "title": "Introduction to the Science of Sociology", "member": "books/28496.txt"},
    {"book_id": 30312, "bin": 3, "title": "Carmen Ariza", "member": "books/30312.txt"},
    {"book_id": 59828, "bin": 3, "title": "The String of Pearls; Or, The Barber of Fleet Street. A", "member": "books/59828.txt"},
    {"book_id": 50801, "bin": 3, "title": "Narrative and Critical History of America, Vol. 1 (of 8", "member": "books/50801.txt"},
    {"book_id": 47476, "bin": 3, "title": "Poems of American History", "member": "books/47476.txt"},
    {"book_id": 46994, "bin": 3, "title": "The pears of New York", "member": "books/46994.txt"},
    {"book_id": 57060, "bin": 3, "title": "London Labour and the London Poor, Vol. 3", "member": "books/57060.txt"},
    {"book_id": 67330, "bin": 3, "title": "A History of Magic and Experimental Science, Volume 2 (", "member": "books/67330.txt"},
    {"book_id": 17439, "bin": 3, "title": "Mother\'s Remedies  Over One Thousand Tried and Tested ", "member": "books/17439.txt"},
    {"book_id": 1184, "bin": 3, "title": "The Count of Monte Cristo", "member": "books/1184.txt"},
    {"book_id": 2988, "bin": 3, "title": "Mark Twain: A Biography. Complete", "member": "books/2988.txt"},
    {"book_id": 49843, "bin": 3, "title": "The History of Java, v. 1-2", "member": "books/49843.txt"},
    {"book_id": 69073, "bin": 3, "title": "The uncivilized races of men in all countries of the wo", "member": "books/69073.txt"},
    {"book_id": 20450, "bin": 3, "title": "The Lives of the Fathers, Martyrs, and Other Principal ", "member": "books/20450.txt"},
    {"book_id": 5668, "bin": 3, "title": "The Arabian Nights Entertainments - Complete", "member": "books/5668.txt"},
    {"book_id": 28500, "bin": 3, "title": "All About Coffee", "member": "books/28500.txt"},
    {"book_id": 55998, "bin": 3, "title": "London Labour and the London Poor, Vol. 1", "member": "books/55998.txt"},
]

BOOKS_E = [
    {"book_id": 41633, "bin": 4, "title": "A Christian Directory, Part 1: Christian Ethics", "member": "books/41633.txt"},
    {"book_id": 9600, "bin": 4, "title": "The Works of John Greenleaf Whittier, Complete", "member": "books/9600.txt"},
    {"book_id": 49351, "bin": 4, "title": "The Pictorial Field-Book of the Revolution, Vol. 1 (of ", "member": "books/49351.txt"},
    {"book_id": 2760, "bin": 4, "title": "Celebrated Crimes (Complete)", "member": "books/2760.txt"},
    {"book_id": 67258, "bin": 4, "title": "A Selection of Cases on the Law of Torts", "member": "books/67258.txt"},
    {"book_id": 8419, "bin": 4, "title": "The Journals of Lewis and Clark, 1804-1806", "member": "books/8419.txt"},
    {"book_id": 53275, "bin": 4, "title": "The Every-day Book and Table Book, v. 1 (of 3)  or Ever", "member": "books/53275.txt"},
    {"book_id": 44608, "bin": 4, "title": "The Bābur-nāma in English (Memoirs of Bābur)", "member": "books/44608.txt"},
    {"book_id": 9170, "bin": 4, "title": "The Three Cities Trilogy, Complete Lourdes, Rome and Pa", "member": "books/9170.txt"},
    {"book_id": 44851, "bin": 4, "title": "Thirty Years\' View (Vol. 1 of 2)  or, A History of the", "member": "books/44851.txt"},
    {"book_id": 663, "bin": 4, "title": "The Gutenberg Webster\'s Unabridged Dictionary: Section", "member": "books/663.txt"},
    {"book_id": 63127, "bin": 4, "title": "Records of the Kirk of Scotland  containing the Acts an", "member": "books/63127.txt"},
    {"book_id": 15476, "bin": 4, "title": "The Mahabharata of Krishna-Dwaipayana Vyasa, Volume 3 B", "member": "books/15476.txt"},
    {"book_id": 6046, "bin": 4, "title": "Works of John Bunyan — Volume 01", "member": "books/6046.txt"},
    {"book_id": 66861, "bin": 4, "title": "History for ready reference, Volume 2, El Dorado to Gre", "member": "books/66861.txt"},
    {"book_id": 39733, "bin": 4, "title": "Cooley\'s Cyclopædia of Practical Receipts and Collater", "member": "books/39733.txt"},
    {"book_id": 57383, "bin": 4, "title": "Project Gutenberg Edition of The Memoirs of Four Civil ", "member": "books/57383.txt"},
    {"book_id": 19218, "bin": 4, "title": "The History of England in Three Volumes, Vol.III.  From", "member": "books/19218.txt"},
    {"book_id": 56200, "bin": 4, "title": "Queen Anna\'s New World of Words; or, Dictionarie of th", "member": "books/56200.txt"},
    {"book_id": 60555, "bin": 3, "title": "The Parochial History of Cornwall, Volume 1 (of 4)", "member": "books/60555.txt"},
    {"book_id": 4361, "bin": 3, "title": "Memoirs of General William T. Sherman — Complete", "member": "books/4361.txt"},
    {"book_id": 44008, "bin": 3, "title": "Warren Commission (08 of 26): Hearings Vol. VIII (of 15", "member": "books/44008.txt"},
    {"book_id": 75040, "bin": 3, "title": "The wonders of the world", "member": "books/75040.txt"},
    {"book_id": 44315, "bin": 3, "title": "Plutarch\'s Lives, Volume 4 (of 4)", "member": "books/44315.txt"},
    {"book_id": 67459, "bin": 3, "title": "The Reign of William Rufus and the Accession of Henry t", "member": "books/67459.txt"},
    {"book_id": 46807, "bin": 3, "title": "Lyman\'s History of old Walla Walla County, Vol. 1  Emb", "member": "books/46807.txt"},
    {"book_id": 51291, "bin": 3, "title": "Narrative and Critical History of America, Vol. 4 (of 8", "member": "books/51291.txt"},
    {"book_id": 60786, "bin": 3, "title": "History of Greece, Volume 12 (of 12)", "member": "books/60786.txt"},
    {"book_id": 44002, "bin": 3, "title": "Warren Commission (02 of 26): Hearings Vol. II (of 15)", "member": "books/44002.txt"},
    {"book_id": 29878, "bin": 3, "title": "The History of Woman Suffrage, Volume V", "member": "books/29878.txt"},
    {"book_id": 821, "bin": 3, "title": "Dombey and Son", "member": "books/821.txt"},
    {"book_id": 70548, "bin": 3, "title": "Parodies of the works of English & American authors, vo", "member": "books/70548.txt"},
    {"book_id": 766, "bin": 3, "title": "David Copperfield", "member": "books/766.txt"},
    {"book_id": 57111, "bin": 3, "title": "Warren Commission (13 of 26): Hearings Vol. XIII (of 15", "member": "books/57111.txt"},
    {"book_id": 4269, "bin": 3, "title": "Complete Plays of John Galsworthy", "member": "books/4269.txt"},
    {"book_id": 3623, "bin": 3, "title": "The Golden Bough: A Study of Magic and Religion", "member": "books/3623.txt"},
    {"book_id": 69425, "bin": 3, "title": "Paradisi in sole paradisus terrestris, or, A garden of ", "member": "books/69425.txt"},
    {"book_id": 25545, "bin": 3, "title": "Children\'s Literature A Textbook of Sources for Teache", "member": "books/25545.txt"},
    {"book_id": 42506, "bin": 3, "title": "Chronicles (1 of 6): The Description of Britaine", "member": "books/42506.txt"},
    {"book_id": 44213, "bin": 3, "title": "Harper\'s Pictorial Library of the World War, Volume XI", "member": "books/44213.txt"},
    {"book_id": 15191, "bin": 3, "title": "The Commercial Products of the Vegetable Kingdom  Consi", "member": "books/15191.txt"},
    {"book_id": 47767, "bin": 3, "title": "The Propaganda for Reform in Proprietary Medicines, Vol", "member": "books/47767.txt"},
    {"book_id": 71433, "bin": 3, "title": "The complete works of John Gower, volume 3", "member": "books/71433.txt"},
    {"book_id": 75871, "bin": 3, "title": "Isis unveiled, Volume 2 (of 2), Theology", "member": "books/75871.txt"},
    {"book_id": 44007, "bin": 3, "title": "Warren Commission (07 of 26): Hearings Vol. VII (of 15)", "member": "books/44007.txt"},
    {"book_id": 54488, "bin": 3, "title": "The Secret Doctrine, Vol. 2 of 4  The Synthesis of Scie", "member": "books/54488.txt"},
    {"book_id": 10625, "bin": 3, "title": "A Concise Dictionary of Middle English from A.D. 1150 t", "member": "books/10625.txt"},
    {"book_id": 62148, "bin": 3, "title": "The Preacher\'s Complete Homiletic Commentary of the Bo", "member": "books/62148.txt"},
    {"book_id": 7440, "bin": 3, "title": "The Koran (Al-Qur\'an)", "member": "books/7440.txt"},
    {"book_id": 24869, "bin": 3, "title": "The Rámáyan of Válmíki, translated into English verse", "member": "books/24869.txt"},
    {"book_id": 58250, "bin": 3, "title": "Index of the Project Gutenberg Works of Thomas Hardy", "member": "books/58250.txt"},
    {"book_id": 43195, "bin": 3, "title": "Chaucerian and Other Pieces Being a Supplement to the C", "member": "books/43195.txt"},
    {"book_id": 3332, "bin": 3, "title": "The Variation of Animals and Plants under Domestication", "member": "books/3332.txt"},
    {"book_id": 3567, "bin": 3, "title": "Memoirs of Napoleon Bonaparte — Complete", "member": "books/3567.txt"},
    {"book_id": 22036, "bin": 3, "title": "Recollections of Forty Years in the House, Senate and C", "member": "books/22036.txt"},
    {"book_id": 3090, "bin": 3, "title": "Complete Original Short Stories of Guy De Maupassant", "member": "books/3090.txt"},
    {"book_id": 32736, "bin": 3, "title": "The Bible Story", "member": "books/32736.txt"},
    {"book_id": 69074, "bin": 3, "title": "The uncivilized races of men in all countries of the wo", "member": "books/69074.txt"},
    {"book_id": 75731, "bin": 3, "title": "The regimental roll of honour and war record of the Art", "member": "books/75731.txt"},
    {"book_id": 74764, "bin": 3, "title": "Bibliographical history of electricity & magnetism, chr", "member": "books/74764.txt"},
    {"book_id": 3253, "bin": 3, "title": "The Papers and Writings of Abraham Lincoln, Complete", "member": "books/3253.txt"},
    {"book_id": 67691, "bin": 3, "title": "Pharmacographia  A history of the principal drugs of ve", "member": "books/67691.txt"},
    {"book_id": 44028, "bin": 3, "title": "The Every Day Book of History and Chronology Embracing ", "member": "books/44028.txt"},
]

BOOKS_F = [
    {"book_id": 42766, "bin": 4, "title": "Report of the Committee Appointed to Investigate the Ra", "member": "books/42766.txt"},
    {"book_id": 44526, "bin": 4, "title": "Report on the Radiolaria Collected by H.M.S. Challenger", "member": "books/44526.txt"},
    {"book_id": 665, "bin": 4, "title": "The Gutenberg Webster\'s Unabridged Dictionary: Section", "member": "books/665.txt"},
    {"book_id": 43097, "bin": 4, "title": "Chaucer\'s Works, Volume 6 —  Introduction, Glossary, a", "member": "books/43097.txt"},
    {"book_id": 666, "bin": 4, "title": "The Gutenberg Webster\'s Unabridged Dictionary: Section", "member": "books/666.txt"},
    {"book_id": 6048, "bin": 4, "title": "Works of John Bunyan — Volume 03", "member": "books/6048.txt"},
    {"book_id": 53143, "bin": 4, "title": "Henley\'s Twentieth Century Formulas, Recipes and Proce", "member": "books/53143.txt"},
    {"book_id": 27889, "bin": 4, "title": "Familiar Quotations A Collection of Passages, Phrases, ", "member": "books/27889.txt"},
    {"book_id": 40588, "bin": 4, "title": "The Kathá Sarit Ságara; or, Ocean of the Streams of Sto", "member": "books/40588.txt"},
    {"book_id": 3672, "bin": 4, "title": "The 2000 CIA World Factbook", "member": "books/3672.txt"},
    {"book_id": 669, "bin": 4, "title": "The Gutenberg Webster\'s Unabridged Dictionary: Section", "member": "books/669.txt"},
    {"book_id": 39225, "bin": 4, "title": "Modern Machine-Shop Practice, Volumes I and II", "member": "books/39225.txt"},
    {"book_id": 51491, "bin": 4, "title": "Church History (Volumes 1-3)", "member": "books/51491.txt"},
    {"book_id": 70343, "bin": 4, "title": "History for ready reference, Volume 5, Tunnage to Zyp a", "member": "books/70343.txt"},
    {"book_id": 3136, "bin": 4, "title": "The Complete Project Gutenberg Writings of Charles Dudl", "member": "books/3136.txt"},
    {"book_id": 11275, "bin": 4, "title": "The Anti-Slavery Examiner, Omnibus", "member": "books/11275.txt"},
    {"book_id": 53884, "bin": 4, "title": "A Biblical and Theological Dictionary  explanatory of t", "member": "books/53884.txt"},
    {"book_id": 58529, "bin": 4, "title": "Hobson-Jobson  A Glossary of Colloquial Anglo-Indian Wo", "member": "books/58529.txt"},
    {"book_id": 11615, "bin": 4, "title": "The Grammar of English Grammars", "member": "books/11615.txt"},
    {"book_id": 73702, "bin": 3, "title": "The Cambridge natural history, Vol. 07 (of 10)", "member": "books/73702.txt"},
    {"book_id": 54824, "bin": 3, "title": "The Secret Doctrine, Vol. 1 of 4  The Synthesis of Scie", "member": "books/54824.txt"},
    {"book_id": 31885, "bin": 3, "title": "Bible Myths and their Parallels in other Religions  Bei", "member": "books/31885.txt"},
    {"book_id": 32600, "bin": 3, "title": "History of the Reign of Philip the Second, King of Spai", "member": "books/32600.txt"},
    {"book_id": 60557, "bin": 3, "title": "The Parochial History of Cornwall, Volume 3 (of 4)", "member": "books/60557.txt"},
    {"book_id": 3300, "bin": 3, "title": "An Inquiry into the Nature and Causes of the Wealth of ", "member": "books/3300.txt"},
    {"book_id": 1023, "bin": 3, "title": "Bleak House", "member": "books/1023.txt"},
    {"book_id": 34916, "bin": 3, "title": "The Chainbearer; Or, The Littlepage Manuscripts", "member": "books/34916.txt"},
    {"book_id": 20587, "bin": 3, "title": "Integration of the Armed Forces, 1940-1965", "member": "books/20587.txt"},
    {"book_id": 50436, "bin": 3, "title": "Protestantism and Catholicity compared in their effects", "member": "books/50436.txt"},
    {"book_id": 60766, "bin": 3, "title": "The Discoverie of Witchcraft", "member": "books/60766.txt"},
    {"book_id": 46595, "bin": 3, "title": "The Chronicles of Crime or The New Newgate Calendar. v.", "member": "books/46595.txt"},
    {"book_id": 3331, "bin": 3, "title": "The Diary of Samuel Pepys", "member": "books/3331.txt"},
    {"book_id": 28148, "bin": 3, "title": "Cotton is King, and Pro-Slavery Arguments  Comprising t", "member": "books/28148.txt"},
    {"book_id": 7467, "bin": 3, "title": "The Newcomes: Memoirs of a Most Respectable Family", "member": "books/7467.txt"},
    {"book_id": 10681, "bin": 3, "title": "Roget\'s Thesaurus of English Words and Phrases", "member": "books/10681.txt"},
    {"book_id": 19488, "bin": 3, "title": "The Life of Joan of Arc, Vol. 1 and 2", "member": "books/19488.txt"},
    {"book_id": 55497, "bin": 3, "title": "The historians\' history of the world in twenty-five vo", "member": "books/55497.txt"},
    {"book_id": 47746, "bin": 3, "title": "Campfire and Battlefield An Illustrated History of the ", "member": "books/47746.txt"},
    {"book_id": 75147, "bin": 3, "title": "The collected stories of Elisabeth Sanxay Holding from ", "member": "books/75147.txt"},
    {"book_id": 58216, "bin": 3, "title": "History of Atchison County, Kansas", "member": "books/58216.txt"},
    {"book_id": 67792, "bin": 3, "title": "A History of Magic and Experimental Science, Volume 1 (", "member": "books/67792.txt"},
    {"book_id": 52105, "bin": 3, "title": "Shell-shock and other neuropsychiatric problems", "member": "books/52105.txt"},
    {"book_id": 56213, "bin": 3, "title": "Nazi Conspiracy and Aggression, Volume 02 (of 11)", "member": "books/56213.txt"},
    {"book_id": 56162, "bin": 3, "title": "The Illustrated Dictionary of Gardening, Division 1; A ", "member": "books/56162.txt"},
    {"book_id": 54169, "bin": 3, "title": "A History of North American Birds; Land Birds; Vol. 3 o", "member": "books/54169.txt"},
    {"book_id": 51649, "bin": 3, "title": "The Mysteries of London, v. 4/4", "member": "books/51649.txt"},
    {"book_id": 41983, "bin": 3, "title": "The Book of Curiosities", "member": "books/41983.txt"},
    {"book_id": 66555, "bin": 3, "title": "Psyche: The Cult of Souls and Belief in Immortality amo", "member": "books/66555.txt"},
    {"book_id": 27977, "bin": 3, "title": "Austral English", "member": "books/27977.txt"},
    {"book_id": 43671, "bin": 3, "title": "A History of Epidemics in Britain, Volume 2 (of 2)  Fro", "member": "books/43671.txt"},
    {"book_id": 47757, "bin": 3, "title": "A Manual of Philippine Birds", "member": "books/47757.txt"},
    {"book_id": 22515, "bin": 3, "title": "The Modern Scottish Minstrel, Volumes 1-6.  The Songs o", "member": "books/22515.txt"},
    {"book_id": 72770, "bin": 3, "title": "The plums of New York", "member": "books/72770.txt"},
    {"book_id": 75996, "bin": 3, "title": "A text-book of veterinary anatomy", "member": "books/75996.txt"},
    {"book_id": 49352, "bin": 3, "title": "The Pictorial Field-Book of the Revolution, Vol. 2 (of ", "member": "books/49352.txt"},
    {"book_id": 12606, "bin": 3, "title": "The Great Speeches and Orations of Daniel Webster  With", "member": "books/12606.txt"},
    {"book_id": 68693, "bin": 3, "title": "History of the inductive sciences, from the earliest to", "member": "books/68693.txt"},
    {"book_id": 12410, "bin": 3, "title": "The Travels of Marco Polo — Volume 2", "member": "books/12410.txt"},
    {"book_id": 66785, "bin": 3, "title": "The Poems of Felicia Hemans", "member": "books/66785.txt"},
    {"book_id": 29090, "bin": 3, "title": "The Complete Poetical Works of Samuel Taylor Coleridge.", "member": "books/29090.txt"},
    {"book_id": 2848, "bin": 3, "title": "Antiquities of the Jews", "member": "books/2848.txt"},
    {"book_id": 16955, "bin": 3, "title": "Three translations of the Koran (Al-Qur\'an) side by si", "member": "books/16955.txt"},
    {"book_id": 64037, "bin": 3, "title": "The Life of Jesus Critically Examined (4th ed.)", "member": "books/64037.txt"},
]

OUTPUT_BY_GROUP = {"a": "/tmp/natural_sc_extra_a.json",
                   "b": "/tmp/natural_sc_extra_b.json",
                   "c": "/tmp/natural_sc_extra_c.json",
                   "d": "/tmp/natural_sc_extra_d.json",
                   "e": "/tmp/natural_sc_extra_e.json",
                   "f": "/tmp/natural_sc_extra_f.json"}
BOOKS_BY_GROUP  = {"a": BOOKS_A, "b": BOOKS_B, "c": BOOKS_C,
                   "d": BOOKS_D, "e": BOOKS_E, "f": BOOKS_F}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["a", "b", "c", "d", "e", "f"], required=True,
                        help="Which group of books to process")
    args = parser.parse_args()

    output_path = OUTPUT_BY_GROUP[args.group]
    books       = BOOKS_BY_GROUP[args.group]

    print(f"Group  : {args.group.upper()}  ({len(books)} books)")
    print(f"Device : {DEVICE}")
    print(f"Output : {output_path}")
    print(f"Zip    : {ZIP_PATH}\n")

    results = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)

    pending = [b for b in books if str(b["book_id"]) not in results]
    print(f"{len(results)} already done, {len(pending)} to compute\n")

    print(f"Loading {DEFAULT_MODEL} ...")
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL, trust_remote_code=True, dtype=torch.float16,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s\n")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for i, binfo in enumerate(books):
            bid = str(binfo["book_id"])
            if bid in results:
                print(f"[{i+1}/{len(books)}] skip  {binfo['title'][:50]}")
                continue

            print(f"\n[{i+1}/{len(books)}] Bin {binfo['bin']}  {binfo['title'][:50]}")
            t_book = time.perf_counter()

            text = zf.read(binfo["member"]).decode("utf-8", errors="replace")
            n_chars = len(text)
            print(f"  {n_chars:,} chars  ≈ {n_chars/4e6:.2f}M tokens")

            sc = SemanticCoherence(
                text=text,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
                tokenizer=tokenizer,
                model=model,
            )

            entry = {
                "book_id":  binfo["book_id"],
                "title":    binfo["title"],
                "bin":      binfo["bin"],
                "n_chars":  n_chars,
                "n_tokens": int(n_chars / 4.0),
            }

            for w in WINDOWS:
                t_w = time.perf_counter()
                sc.compute_window(w)
                wname = WINDOW_NAMES[w]
                prefix_key = f"sc_{wname}_"
                scalars = {k: sc.results.get(f"{prefix_key}{k}") for k in SCALAR_KEYS}
                n_win = sc.results.get(f"{prefix_key}n_windows", 0)
                entry[wname] = scalars
                print(f"  {wname}: {n_win} windows  "
                      f"odm={scalars['off_diag_mean']:.3f}  "
                      f"nov={scalars['novelty_mean']:.3f}  "
                      f"sl1={scalars['sim_lag1_mean']:.3f}  "
                      f"({time.perf_counter()-t_w:.0f}s)")

            results[bid] = entry
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  saved ({time.perf_counter()-t_book:.0f}s total)")

    print(f"\nDone. {output_path}")


if __name__ == "__main__":
    main()
