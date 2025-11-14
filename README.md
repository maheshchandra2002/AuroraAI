# Member Message Question‑Answering Service

This repository contains a **simple question‑answering API** designed to
answer natural‑language questions about member activities.  The service
consumes messages from a publicly available API and attempts to infer
answers without requiring any special credentials or additional data
sources.

## Features

* **Single endpoint**: send a question to `/ask` and receive a short answer.
* **No training required**: the service uses heuristics and TF‑IDF similarity
  to derive answers directly from member messages.
* **Automatic data retrieval**: all messages are fetched from the remote
  `/messages` API; no database setup is required.
* **Extensible**: modular functions make it straightforward to add new
  question types or swap out the similarity backend for embeddings.

## Quick start

1. **Clone** this repository and change into the project directory:

   ```bash
   git clone https://github.com/<your‑username>/qa_service.git
   cd qa_service
   ```

2. **Install** the dependencies.  It's recommended to use a virtual
   environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run** the application locally using Uvicorn:

   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **Query** the API.  Open another terminal or use a tool like `curl` to ask
   questions:

   ```bash
   curl "http://localhost:8000/ask?question=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"
   curl "http://localhost:8000/ask?question=How%20many%20cars%20does%20Vikram%20Desai%20have%3F"
   curl "http://localhost:8000/ask?question=What%20are%20Amira’s%20favorite%20restaurants%3F"
   ```

   Each request returns a JSON object with an `answer` field.  For example:

   ```json
   { "answer": "Layla Kawaguchi is planning a trip to London on December 10, 2025." }
   ```

## Design notes

### Approach implemented

The core of the application resides in `main.py`.  When a question arrives
the service fetches all member messages by paginating through the
`/messages` endpoint provided by the exercise (see
`https://november7-730026606190.europe-west1.run.app/docs`).  Once the
messages are available the service:

1. **Identifies the subject** of the question by scanning for known
   member names.  This is performed by the `detect_person` function which
   performs a case‑insensitive match against the `user_name` field in
   each message.
2. **Routes** the question to a specific handler based on simple heuristics:
   * *When …?* questions are handled by `answer_when_question`.  The
     function looks for phrases like "trip to Paris" in the question, finds
     corresponding messages mentioning trips, parses a date (using
     `dateparser` if installed) and returns a sentence such as
     "Layla Kawaguchi is planning a trip to London on December 10, 2025".
   * *How many …?* questions currently support counting the number of
     cars owned.  `answer_how_many_question` searches for the keyword
     (e.g. "car") in the member’s messages and extracts a number expressed as
     digits or as words ("two", "three" etc.).
   * Questions asking about *favourite restaurants* are handled by
     `answer_favorite_question`, which looks for messages containing both
     "favorite" and "restaurant" and attempts to parse the list of
     restaurants.
3. If none of the specialised handlers return an answer the service
   falls back to a **similarity search**.  A TF‑IDF vector is built from
   the question and all messages using scikit‑learn’s
   `TfidfVectorizer`, and the message with the highest cosine similarity
   to the question is returned verbatim.  This simple retrieval
   baseline ensures that the client receives a relevant sentence even
   when the heuristics do not apply.

### Alternative approaches considered

* **Vector search with embeddings**: Instead of TF‑IDF one could compute
  dense embeddings (e.g. with sentence transformers) for each message and
  the incoming question, then use cosine similarity on the embeddings.
  Dense vectors capture semantic meaning better than bag‑of‑words and
  would likely provide more accurate matches.  This approach was
  discarded here to avoid pulling large models and because the
  environment does not include pre‑trained embedding libraries by default.

* **Large language models (LLMs)**: A more sophisticated solution would
  employ an LLM (such as GPT‑4) to summarise relevant messages and
  generate natural responses.  Combining an embedding‑based retriever
  with an LLM forms a Retrieval‑Augmented Generation (RAG) system.  Such
  a system could answer arbitrary questions beyond the simple patterns
  implemented here.  However, LLM inference incurs cost and latency and
  often requires API keys.  For an assessment exercise a heuristic
  approach is sufficient.

* **Traditional rule‑based extraction**: We considered writing a large
  set of regular expressions tailored to each type of question.  While
  accurate for a narrow domain, this strategy scales poorly and is
  brittle in the face of varied wording.  Instead we opted to
  implement a few simple patterns (date, number, favourites) and rely on
  vector similarity when no pattern matches.

### Deployment

The service can be deployed to any platform that supports running a
Python web application.  On cloud platforms such as **Heroku**,
**Render**, **Railway** or **AWS Elastic Beanstalk** you can configure a
web dyno with Python 3.11+, install dependencies via `requirements.txt`
and start the server with the command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Ensure that outbound HTTP requests are permitted so the application can
reach the messages API.

## Data insights and anomalies

While developing the service we inspected a portion of the **3349**
messages exposed by the `/messages` API to understand their structure
and content.  Several noteworthy observations were made:

* **Unrealistic requests**: Many messages involve extravagant tasks such
  as booking private jets, arranging yachts in Monaco, or organising
  exclusive dinners at Michelin‑starred restaurants.  These tasks hint
  at a synthetic or aspirational dataset rather than everyday
  concierge requests.

* **Varied formatting**: Dates appear in multiple forms within the
  messages—some include explicit ISO timestamps (e.g. `2025-05-05T07:47:20.159073+00:00`),
  while others refer to relative times (“next weekend”, “this Friday”).  To
  handle this the service uses `dateparser` with a preference for future
  dates.

* **Names with mixed whitespace and accents**: User names contain
  accented characters (e.g. "Amélie", "Müller") and sometimes extra spaces.
  Matching names must be case‑insensitive and tolerant of such
  variations.

* **Missing or irrelevant information**: Not all questions will find an
  answer.  For example, in the subset analysed we did not find
  messages explicitly stating how many cars **Vikram Desai** owns.  The
  system therefore gracefully informs the user when it cannot answer
  rather than hallucinating a response.

These observations shaped the design of the heuristics and highlight
opportunities for future improvements such as normalising names,
standardising dates or enriching the dataset.

## License

This project is released under the MIT License.  See `LICENSE` for details.