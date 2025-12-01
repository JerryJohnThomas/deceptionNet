# Synthetic Werewolf Conversations

This folder contains hand-crafted and programmatically generated examples that mirror the raw dataset schema used in the project. All conversation text and summaries are written in English so they can feed directly into imitation pipelines without translation.

## Contents
- 1.json � Villagers win after a Day 1 seer reveal.
- 2.json � Werewolves win by baiting a fake medic.
- 3.json .. 502.json � 500 additional synthetic games generated with varied night kills, vote patterns, and talk snippets. Every game includes the four key roles: Villager, Werewolf, Seer, Doctor, plus full judge/task/audio sections.

## Notes
- The structure matches the public MindGames raw logs (judge, 	ask, game_state, 	ask_time, udio, ideo, origin).
- Role identifiers stay within the competition conventions (Villager, Werewolf, Seer, Doctor) so downstream converters can remap Doctor to Medic if needed.
- Timings and voting patterns are internally consistent for short two-day scenarios, making the set useful for smoke tests, parser unit tests, or early-stage imitation prototyping.

Feel free to expand or regenerate the set by tweaking the generator script in the repo history.


> Across the 502 synthetic logs in generated/chatgpt/:

  - Village wins (The good side wins): 256 games → 51.0 %
  - Werewolf wins (Werewolves Win): 246 games → 49.0 %
  - No other outcomes present.

  Nice near-even split for testing.