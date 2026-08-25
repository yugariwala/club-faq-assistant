"""Fixed knowledge base for the GDG On Campus Club FAQ Assistant.

Single source of truth for KB storage. Content is reproduced verbatim from
`requirements.md` §2 ("Knowledge Base") -- do not edit wording/values here
without updating requirements.md first (see spec Boundaries: "Ask First").

Each entry is tagged with its section label. These labels are the exact
strings surfaced to users as `source_section` in answers/citations.
"""

KB_ENTRIES: list[dict] = [
    {
        "section": "Intro",
        "content": (
            "GDG On Campus is a community of 150+ tech enthusiasts. "
            "Founded in 2022. Organizes workshops, hackathons, and speaker sessions."
        ),
    },
    {
        "section": "Teams",
        "content": (
            "AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel), "
            "App Dev (Lead: Arjun Mehta), Cloud (Lead: Sneha Gupta), "
            "Cybersecurity (Lead: Vikram Singh), Design (Lead: Ananya Reddy)"
        ),
    },
    {
        "section": "Events",
        "content": (
            "Intro to GenAI Workshop (Sept 15, Upcoming), HackFest 2025 (Oct 10, Upcoming), "
            "Cloud Study Jam (Sept 20, Upcoming), Flutter Forward (Aug 30, Completed), "
            "CyberCTF Challenge (Nov 5, Upcoming), Design Thinking Bootcamp (Sept 25, Upcoming)"
        ),
    },
    {
        "section": "Recruitment",
        "content": (
            "Application Form → Technical Assessment (1 week) → Interview (15 min) "
            "→ Results (1 week) → Onboarding (2 weeks). "
            "Window: Sept 1–15, 2025. Eligibility: 1st to 3rd year."
        ),
    },
    {
        "section": "Rules",
        "content": (
            "Minimum 2 events/month to stay active. Inactive for 2 months = alumni status. "
            "Team switching once per semester. At least 1 project contribution per semester."
        ),
    },
    {
        "section": "Contacts",
        "content": (
            "President: Aditya Kumar (president@gdgoncampus.com), VP: Meera Joshi, "
            "Tech Head: Rohan Desai, General: info@gdgoncampus.com"
        ),
    },
    {
        "section": "Achievements",
        "content": (
            "Best Community Award at DevFest 2024, 12 open-source projects (500+ GitHub stars), "
            "25+ workshops in 2024–25, partnerships with 3 college clubs."
        ),
    },
]
