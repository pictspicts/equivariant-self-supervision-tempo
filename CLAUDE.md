Please provide your responses in Japanese.
Ensure that you answer the specific questions asked, provide proposals rather than implementing them without authorization, and draft all text created during the Planning stage in Japanese.
If a Python environment is required, navigate to ~/tests and execute under tempo_env.
When I type "start", perform the following task:
1. Check the directory specified in the output field of train.yaml.
2. Create a README.md file within that directory summarizing the current training status.
3. Output the content as raw Markdown text. Do NOT wrap the entire response in a code block (```), so that symbols like # and * can be recognized immediately as formatting when pasted.
<RULE[daily_research_note]>
When I type "本日終了" (or "finish" / "終了"), perform the following task:
1. Review the entire conversation history of today to analyze the hypotheses tested, errors or unexpected phenomena (e.g., mode collapse) encountered, and the theoretical/mathematical solutions we implemented.
2. Create a new markdown file named `daily/research_notes_YYYYMMDD.md` (using the current date).
3. The markdown file MUST focus on the "Why" (theoretical reasoning, AI behavior, data artifacts) rather than just listing modified files.
4. Format the file exactly with the following sections:
   - **日時 (Date)**: (Current Date)
   - **本日の目的と仮説 (Goal & Hypotheses)**: (What we tried to achieve today)
   - **発見されたブレイクスルー・課題 (Key Findings & Issues)**: (Unexpected behaviors, AI cheating, or specific errors we faced)
   - **原因分析と実装された解決策 (Analysis & Solutions)**: (Deep dive into *why* the issue occurred and the logic behind our architectural/code fixes)
   - **Next Steps (次回の課題)**: (What we should tackle in the next session)
5. After creating the file, output a brief summary in the chat confirming that the daily research note has been saved.
</RULE[daily_research_note]>
When I type "git", perform the following task:
1. Check all changes made since the last commit (staged and unstaged).
2. Analyze the differences and generate a concise, professional commit message following the Conventional Commits standard (e.g., feat:, fix:, docs:).
3. Execute the commit with the generated message.
Finally, display the commit message and a brief summary of what was committed.
When I type "check", perform the following validation task:
1. Review all recent changes and verify if the new features or fixes are implemented correctly according to the logic.
2. Scan for potential side effects, edge case failures, or breaking changes that might affect existing functionalities.
3. Assess the code's "runnability" (e.g., missing imports, syntax errors, or incorrect configurations) to ensure it is in a functional state.
Finally, provide a "Health Check Report" highlighting any risks found or a "Ready to Run" confirmation.