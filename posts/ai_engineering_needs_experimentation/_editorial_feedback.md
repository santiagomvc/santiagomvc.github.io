# Editorial feedback — "AI Engineering needs experimentation"

Editor, 2026-09-19. Higher-level review of the draft: is the take right, what Paul Graham
and Scott Alexander would say, and what is missing at each level. Sentence-level findings
are in `~/Documents/team-hq/state/reports/editor-blog.md`.

## Verdict

The take is right. It is not new: it is the position of most people who publish about
building LLM products. It is worth a post only if you add something the reader cannot get
from those people: your case, a boundary test, or the reason the other view is winning.

## Is it right?

Yes, with two soft spots.

1. **The draft describes a weaker version of the other camp than the one good engineers
   hold.** "A few knobs, almost deterministic" is not what a good software engineer
   believes. That engineer runs A/B tests, canaries, feature flags and dashboards. That is
   experimentation with a different vocabulary: online, on production traffic, on usage
   metrics. The real split is not experiment versus configure. It is offline experiment on
   a held-out set (data science) versus online experiment on live traffic (web). Stated
   that way, both camps have a method and both methods miss something. That is a more
   interesting post than "they are wrong".
2. **"Wrong" has no boundary.** Your 2024 post said rules first, ML when rules fall short.
   The same rule applies here. For "summarize this email" the component view works. The
   question a reader wants answered is where the line is. One candidate test: if you can
   write a deterministic acceptance check for one output, treat the model as a component;
   if judging one output needs a person or a model-judge, you are in evaluation, and
   evaluation is experimentation. Your call whether that matches your experience.

## What Paul Graham would say

From his essays, not from him. Three questions.

1. **What did you find out?** The draft is a position. His essays are built around a
   discovery. The discovery in your draft is the line "why are customers not reading our
   outputs". That is something you learned; the rest is what you already believed. Build
   the post around the learning.
2. **Why do smart people believe the thing you call wrong?** The draft says the view is
   growing and never says why. Candidates: the model is now bought, not trained, so the
   visible half of the data-science job disappeared and people concluded the whole job
   did; data-science teams had a reputation for notebooks that never shipped; vendors sell
   "API plus prompt equals product". Name the cause and the reader trusts the diagnosis.
3. **Say the surprising true thing.** The surprising claim available to you: training died
   and the experiment loop survived. The model went from an artifact you produce to a
   component you buy, and every step of the loop except training still applies:
   understand the data, define the metric, look at the errors, change something, measure
   again. The software camp saw "no training" and read it as "no experiment". That
   sentence is your thesis. It is not in the draft.

## What Scott Alexander would say

1. **Steelman first.** He would write the best version of the component view before
   disagreeing with it, and it would be the one in "Is it right?" point 1. The draft gives
   the other side one clause.
2. **Conflict of interest.** A data scientist arguing that data-science skills are
   essential. He would state that up front, in one sentence, and then argue. It answers
   the objection before the reader raises it.
3. **Attribution.** "Unsatisfied customers" has many causes: latency, cost, wrong product,
   bad UX. The draft attributes all of it to missing experimentation. He would ask how you
   know, and the answer is the example paragraph you have not written yet.
4. **Isolated demand for rigor.** You admit the data-science loop "wasn't up to the
   standards of academic research". So the two camps differ in degree of rigor, not in
   kind. A continuum is a weaker claim than a split, and a truer one.

## What is missing, by level

| Level | Gap |
|---|---|
| Sentence | In `editor-blog.md`. |
| Structure | In the earlier plan: claim in paragraph 1, example in place of the abstract failure, short "what to do" list before the closing line, link to the 2024 post. |
| Argument | No mechanism for why the other view spreads; no boundary for when it suffices; no evidence beyond "in my experience". |
| Discourse | The term "AI Engineer" was popularized to describe the API-using software role, not the data scientist (Swyx, "The Rise of the AI Engineer", 2023). Your post argues that role must import the data-science loop. Say so and the reader has a reference point. Hamel Husain, Shreya Shankar, Eugene Yan and Chip Huyen's "AI Engineering" book argue the evaluation-first position at length. A reader who knows them will ask why the post does not mention them. |
| Reader | What you have that they do not: WhatsApp retail bots in Colombia and legislative documents in the US, with customers who did not read the outputs. That is the material for the example. |
| Method | Hill climbing on a technical metric risks Goodhart's law: the metric improves and the product does not. Your 2024 post separated model metrics from usage metrics; one sentence here on the same split closes the gap. |
| Vocabulary | "Deterministic" is the wrong word for the other camp's belief. They know the model is stochastic; they handle it with temperature 0, structured outputs and retries. Their belief is "good enough out of the box". Argue against what they believe. |

## Unverified

Everything attributed to Graham, Alexander, Swyx, Husain, Shankar, Yan and Huyen is from
memory. No source was fetched. Verify before any of it goes into the post.

## Questions for you

1. Does the boundary test (deterministic acceptance check versus judged output) match what
   you saw? If not, what is your line?
2. Is there a moment where you learned something, rather than confirmed something?
   "Customers were not reading our outputs" reads like one. What happened next?
3. Do you want to name the Swyx definition and the evaluation-first writers, or stay
   independent of the discourse?
4. Do you want the one-sentence bias disclosure?
5. Do you accept "training died and the experiment loop survived" as the thesis, in your
   words, or is that not what you believe?
