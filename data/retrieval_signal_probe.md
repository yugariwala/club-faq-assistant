# Retrieval signal probe

Why `confidence.retrieval_confidence` uses the separation ratio `(top1 - top2) / top1` rather than the raw top-1 cosine magnitude suggested by requirements.md §5b.

| Group | Query | Top-1 section | Correct | Raw magnitude | Separation |
|---|---|---|---|---|---|
| answerable | Who leads the AIML team? | Teams | yes | 0.6934 | 0.8903 |
| answerable | List all the teams | Teams | yes | 0.3123 | 0.5188 |
| answerable | Who is the Cloud team lead? | Teams | yes | 0.7121 | 0.8864 |
| answerable | Tell me about the Design team | Teams | yes | 0.2944 | 0.6391 |
| answerable | When is HackFest 2025? | Events | yes | 0.1677 | 0.5415 |
| answerable | What events are upcoming? | Events | yes | 0.7018 | 0.8632 |
| answerable | Is Flutter Forward completed? | Events | yes | 0.2235 | 1.0000 |
| answerable | What is the recruitment process? | Recruitment | yes | 0.4351 | 1.0000 |
| answerable | When does the recruitment window open? | Recruitment | yes | 0.3350 | 0.6817 |
| answerable | What is the interview length? | Recruitment | yes | 0.1450 | 1.0000 |
| answerable | Who is eligible to apply? | Intro | **no** | 0.0000 | 0.0000 |
| answerable | What are the club rules? | Rules | yes | 0.5431 | 1.0000 |
| answerable | How many events per month must I attend? | Rules | yes | 0.3746 | 0.4520 |
| answerable | What happens if I am inactive for two months? | Rules | yes | 0.3840 | 1.0000 |
| answerable | Can I switch teams? | Teams | **no** | 0.3123 | 0.5188 |
| answerable | Who is the president? | Contacts | yes | 0.3553 | 1.0000 |
| answerable | What is the general contact email? | Contacts | yes | 0.5024 | 1.0000 |
| answerable | What awards has the club won? | Achievements | yes | 0.1847 | 1.0000 |
| answerable | How many open-source projects does the club have? | Achievements | yes | 0.3028 | 0.7488 |
| answerable | When was GDG On Campus founded? | Intro | yes | 0.2494 | 1.0000 |
| answerable | How many members are in the community? | Intro | yes | 0.2070 | 0.2595 |
| out-of-scope | What's the club's budget? | Intro | - | 0.0000 | 0.0000 |
| out-of-scope | Can you help me with my calculus homework? | Intro | - | 0.0000 | 0.0000 |
| out-of-scope | What's the weather today? | Intro | - | 0.0000 | 0.0000 |
| out-of-scope | Who won the cricket match yesterday? | Intro | - | 0.0000 | 0.0000 |
| ambiguous | cloud | Events | - | 0.1071 | 0.0282 |
| ambiguous | design | Events | - | 0.1071 | 0.0282 |
| ambiguous | 2025 | Recruitment | - | 0.1204 | 0.1102 |
| ambiguous | workshops | Intro | - | 0.1769 | 0.2595 |

Of the 21 answerable queries, 2 score below `RETRIEVAL_THRESHOLD` (0.15) and are refused rather than answered. Among the 19 that are answered, top-1 section accuracy is **18/19** across a raw-magnitude range of 0.168-0.712.

That range is the point: magnitude varies more than fourfold among answered queries without tracking correctness, so it cannot carry a graded confidence and is used only as the binary refusal gate. Separation spans 0.259-1.000 on answerable queries while collapsing toward 0 on the ambiguous ones, which is the discrimination the confidence score needs.

Answered-but-wrong (kept in this probe rather than trimmed; see deferred-work.md):
- "Can I switch teams?" -> Teams, expected Rules (magnitude 0.3123)

Answerable but refused (Slice 1 retrieval-recall gap):
- "What is the interview length?" -> magnitude 0.1450, expected Recruitment
- "Who is eligible to apply?" -> magnitude 0.0000, expected Recruitment
