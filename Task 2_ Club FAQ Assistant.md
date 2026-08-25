## 🤖 AIML Task: Club FAQ Assistant

## Problem Statement

Build an AI-powered chatbot for the club that can answer questions, perform actions, and track interactions — all grounded in the provided club information.

The chatbot should go beyond simple Q&A. It should understand conversational context, avoid making up answers, and be capable of executing simple tasks like an AI agent.

## Expected Behavior:

User: "What teams are available in the club?" Bot: Lists available teams with details, citing the source. User: "Register me for the GenAI workshop" Bot: Collects required info → performs the action → confirms.

## Club Information

Use the following as your knowledge base. You may structure and store this data in any format you see fit.

Club Introduction — GDG On Campus is a community of 150+ tech enthusiasts. Founded in 2022. Organizes workshops, hackathons, and speaker sessions.

Teams — AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel), App Dev (Lead: Arjun Mehta), Cloud (Lead: Sneha Gupta), Cybersecurity (Lead: Vikram Singh), Design (Lead: Ananya Reddy)

Events — Intro to GenAI Workshop (Sept 15, Upcoming), HackFest 2025 (Oct 10, Upcoming), Cloud Study Jam (Sept 20, Upcoming), Flutter Forward (Aug 30, Completed), CyberCTF Challenge (Nov 5, Upcoming), Design Thinking Bootcamp (Sept 25, Upcoming)

Recruitment — Application Form → Technical Assessment (1 week) → Interview (15 min) → Results (1 week) → Onboarding (2 weeks). Window: Sept 1–15, 2025. Eligibility: 1st to 3rd year.

Rules — Minimum 2 events/month to stay active. Inactive for 2 months = alumni status. Team switching once per semester. At least 1 project contribution per semester.

Contacts — President: Aditya Kumar (president@gdgoncampus.com), VP: Meera Joshi, Tech Head: Rohan Desai, General: info@gdgoncampus.com

Achievements — Best Community Award at DevFest 2024, 12 open-source projects (500+ GitHub stars), 25+ workshops in 2024–25, partnerships with 3 college clubs.


## Requirements

## 1. Core Chatbot

Build a chatbot that answers user questions based on the club information above. Use any LLM API or open-source model. If the answer isn't available in the data, the bot must say so —

never fabricate answers.

## 2. Smart Features

The chatbot should go beyond basic Q&A:

- Multi-turn memory — understand references to previous messages in the same session ("Who leads it?" should resolve based on context)

- Source citation — tell the user where the answer came from

- Confidence scoring — indicate how confident the bot is in its response

- Intent classification — categorize what the user is asking (FAQ, event inquiry, action request, etc.) and show it in the UI

## 3. Agentic Actions

The bot should not only answer but also do things when asked. Implement at least 2 actions such as event registration, feedback submission, status check, or reminder setup. The bot should gather missing information conversationally, persist the data, and confirm the completed action.

## 4. Dashboard

Build a simple dashboard to verify everything works end-to-end — chat stats, intent breakdown, actions log, and any unanswered queries.
