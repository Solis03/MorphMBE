# Target Roughness Control Requirements

No controller is implemented in Phase 0.

No structured, software-controllable growth variables were discovered in the
machine-readable benchmark metadata. Filename text contains ambiguous process
tokens such as ramp-down temperature labels and GaSb duration labels, but these
are not curated variables and are not allowed as model inputs.

Candidate variables requiring expert curation before any control software:

- ramp-down temperature: unit appears to be C in some filenames; no trusted
  structured range is available;
- deposition or process-stage duration: unit appears to be minutes in some
  filenames; no trusted structured range is available;
- material/process stage labels such as GaSb or AlSb: categorical context only
  until curated.

Future control software must define which variables are available to software,
operator-approved units, observed historical ranges, safety limits, and any hard
constraints. Expert approval is required before using those variables for
recommendations.

The future interface must accept a target Rq in nm, output predicted Rq in nm,
provide recommended actions only in advisory mode by default, require human
confirmation, reject out-of-distribution states, record uncertainty, and log the
full experiment context. Autonomous mode requires a separate safety review.
