# Non-scientific tool notes

- Initial apply_patch could not create the audit helper. PowerShell wrote the helper before scientific execution.
- A functions.exec JavaScript string parse error occurred while preparing the summary generator. No nested tool or scientific command ran; corrected string construction was used.
- The first final deliverable-list PowerShell command had an invalid foreach-to-pipeline expression (empty pipe element). It failed during parsing and read/wrote nothing. A corrected read-only listing was used.

Scientific commands, model hashes, the one-shot CAL/CONFIRM result, report, and execution summary are unchanged. All scientific and auxiliary-check failures remain in their original stage logs.