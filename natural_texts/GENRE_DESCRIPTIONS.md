# Genre Descriptions for PCA Analysis

## 18 Final Genre Categories

### Fiction & Literature (8 categories)

1. **General Fiction**
   - General novels, literary fiction
   - Includes: Classics of Literature, British Literature, American Literature
   - Examples: "Moby Dick", "Pride and Prejudice"

2. **Historical Fiction**
   - Historical novels
   - Examples: "A Tale of Two Cities", "Ivanhoe"

3. **Romance**
   - Romance novels
   - Examples: "Jane Eyre", "Wuthering Heights"

4. **Mystery/Crime**
   - Detective stories, crime fiction, thrillers
   - Examples: Sherlock Holmes, Agatha Christie

5. **Sci-Fi/Fantasy**
   - Science fiction and fantasy novels
   - Examples: H.G. Wells, Jules Verne

6. **Short Stories**
   - Collections of short fiction
   - Anthologies

7. **Poetry & Drama**
   - Poetry collections, plays, dramatic works
   - Examples: Shakespeare, Milton's "Paradise Lost"

8. **Children/YA**
1.  **General Fiction**
    -   General novels, literary fiction
    -   Includes: Classics of Literature, British Literature, American Literature
    -   Examples: "Moby Dick", "Pride and Prejudice"

2.  **Historical Fiction**
    -   Historical novels
    -   Examples: "A Tale of Two Cities", "Ivanhoe"

3.  **Romance**
    -   Romance novels
    -   Examples: "Jane Eyre", "Wuthering Heights"

4.  **Mystery/Crime**
    -   Detective stories, crime fiction, thrillers
    -   Examples: Sherlock Holmes, Agatha Christie

5.  **Sci-Fi/Fantasy**
    -   Science fiction and fantasy novels
    -   Examples: H.G. Wells, Jules Verne

6.  **Short Stories**
    -   Collections of short fiction
    -   Anthologies

7.  **Poetry & Drama**
    -   Poetry collections, plays, dramatic works
    -   Examples: Shakespeare, Milton's "Paradise Lost"

8.  **Children/YA**
    -   Children's literature and young adult fiction
    -   Includes: Children's Anthologies
    -   Examples: "Anne of Green Gables"

### Non-Fiction Narrative (4 categories)

9.  **Adventure** (960 books)
    -   Adventure narratives, exploration stories
    -   Examples: "Robinson Crusoe", adventure tales

10. **Biography** (1,820 books)
    -   Life stories, memoirs, autobiographies
    -   Examples: Presidential biographies, historical figures

11. **History** (3,373 books)
    -   Historical works, historical analysis
    -   **Should include**: Early Modern history, Medieval/Middle Ages, Warfare, Classical Antiquity, Slavery, United States history, Archaeology & Anthropology
    -   Examples: Historical accounts, period studies

12. **Travel** (878 books)
    -   Travel narratives, exploration accounts
    -   Geographical descriptions

### Intellectual & Academic (5 categories)

13. **Science**
    - Scientific works, natural sciences
    - **Should include**: Chemistry/Biochemistry, Psychology/Psychiatry, Nature/Gardening/Animals, Technology
    - Examples: Darwin, scientific treatises

14. **Philosophy**
    - Philosophical works, ethics, logic
    - Examples: Plato, Aristotle, modern philosophy

15. **Religion**
    - Religious texts, theology, spiritual works
    - Examples: "The Book of Mormon", religious commentaries

16. **Mythology**
    - Myths, legends, folklore
    - Examples: Greek mythology, Norse legends

17. **Reference**
    - Dictionaries, encyclopedias, how-to guides
    - Educational materials, language learning
    - **Should include**: Technology guides, instructional materials
    - Examples: Webster's Dictionary, CIA World Factbook

### Expression & Communication (2 categories)

18. **Essays & Letters**
    - Essay collections, correspondence, non-fiction prose
    - Examples: Emerson, philosophical essays

19. **Humor**
    - Humorous works, satire, comedy
    - Examples: Mark Twain's humorous works

### Miscellaneous

20. **Other**
    - **Should contain only**: Art, Cooking & Drinking, Law & Criminology, and truly uncategorizable works
    - **Should NOT contain**: History, Science, or items that fit other categories
    - Currently needs reclassification

## Known Issues

### Items Misclassified in "Other" (handled by `refine_genres.py`):
- **Psychiatry/Psychology** → Moved to **Science**
- **History - Early Modern** → Moved to **History**
- **Science - Chemistry/Biochemistry** → Moved to **Science**
- **Archaeology & Anthropology** → Moved to **History**
- **History - Medieval/Middle Ages** → Moved to **History**
- **Slavery** → Moved to **History**
- **Nature/Gardening/Animals** → Moved to **Science**
- **Classical Antiquity** → Moved to **History**
- **Technology** → Moved to **Science**
- **United States** → Moved to **History**
- **Children's Anthologies** → Moved to **Children/YA**
- **British Literature** → Moved to **General Fiction**

## Distribution Summary
- **Total books**: ~19,626
- **Largest category**: General Fiction
- **Smallest category**: Philosophy
- **Most problematic**: Other (significantly reduced by `refine_genres.py`)
