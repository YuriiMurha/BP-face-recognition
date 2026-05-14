# Writing Guide for Thesis Chapters

## Anti-AI Detection Rules

### BANNED Words/Phrases
- delve, tapestry, landscape, foster, pivotal, underscore, nuanced, multifaceted
- synergy, realm, underpins, unraveling, unveiling, leveraging
- furthermore, moreover, consequently, subsequently, henceforth, thereby, wherein
- "serves as a testament", "plays a crucial role", "pivotal moment"
- "In recent years," "In the ever-evolving" "The field of" as paragraph openers
- "Additionally," "Furthermore," as sentence starters
- "features a" → use "has a"; "represents a" → use "is a"

### ALLOWED Technical Terms
- implement, optimize, prioritize — these are correct technical vocabulary
- paradigm — acceptable in technical context (max 3-4 per chapter)
- key — normal English word
- but, it is — basic English, not AI signals

### Writing Patterns
- Mix short (5-8 word) and long (30+ word) sentences
- Don't force rule-of-three patterns
- Use straight quotes: " not " "
- Some contractions OK: "doesn't" not "does not"

### Citations
- Use [CITE: topic] placeholders
- Never invent author names or years
- Example: [CITE: FaceNet paper] not (Schroff et al., 2015)

## Detection Script
Run: `python "C:/Users/yuram/.openclaw/workspace/skills/humanize-ai-text/scripts/detect.py" "<filepath>"`
Ignore false positives: markdown formatting, "but", "it is", "key", "paradigm" in technical context.
Fix real issues: banned words, copula avoidance, filler phrases.
