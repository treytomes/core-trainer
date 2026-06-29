---
description: Create a journal entry documenting learning, accomplishments, next steps, and reflections
---

# Journal Entry Skill

When the user invokes `/journal`, create a new journal entry in `.claude/journal/` documenting the current session's work.

## Instructions

1. Determine the current date and create a filename: `YYYY-MM-DD.md` (or append to existing file if one exists for today)

2. If creating a new entry, use this template:

```markdown
# Journal Entry - [Date]

## Accomplishments
- What was built, fixed, or implemented
- What tasks were completed

## Learning
- New concepts or technologies learned
- Insights gained about the codebase or problem domain
- Challenges overcome and how

## Next Steps
- Planned work for future sessions
- Open questions or decisions needed
- Ideas for improvements or features

## Reflections
- What went well
- What could be improved
- Any blockers or concerns
```

3. If appending to an existing entry for today, add a new timestamped section:

```markdown
---

## Update - [Time]

[Same sections as above]
```

4. Fill in each section based on the conversation context:
   - Review what was accomplished in this session
   - Note any learning moments or insights shared
   - Identify logical next steps based on current state
   - Add thoughtful reflections on the process

5. Write the journal entry to `.claude/journal/[date].md`

6. Confirm to the user that the journal entry has been created/updated

## Notes

- Keep entries concise but meaningful
- Focus on context that would help future sessions
- Include specific file paths and technical details where relevant
- Don't just list tasks - capture the *why* and *how*
- If the user provides specific content to journal, incorporate it
